import tensorflow as tf
from utility import common
from utility import utils
slim = tf.contrib.slim


class darknet53(object):
    """framework for feature extraction"""

    def __init__(self, inputs):
        self.outputs = self.forward(inputs)

    def res_block(self, inputs, filters):
        """
        implement residual block.
        """
        shortcut = inputs
        inputs = common.conv2d(inputs, filters, 1)
        inputs = common.conv2d(inputs, filters * 2, 3)

        inputs = inputs + shortcut
        return inputs

    def forward(self, inputs):
        inputs = common.conv2d(inputs, 32, 3, strides=1)
        inputs = common.conv2d(inputs, 64, 3, strides=2)
        inputs = self.res_block(inputs, 32)
        inputs = common.conv2d(inputs, 128, 3, strides=2)
        # res_block2
        for i in range(2):
            inputs = self.res_block(inputs, 64)

        inputs = common.conv2d(inputs, 256, 3, strides=2)
        # res_block8
        for i in range(8):
            inputs = self.res_block(inputs, 128)

        feature_1 = inputs
        inputs = common.conv2d(inputs, 512, 3, strides=2)

        for i in range(8):
            inputs = self.res_block(inputs, 256)

        feature_2 = inputs
        inputs = common.conv2d(inputs, 1024, 3, strides=2)

        for i in range(4):
            inputs = self.res_block(inputs, 512)
        return feature_1, feature_2, inputs


class yolov3(object):
    def __init__(self, num_classes, anchors, batch_norm_decay=0.9, leaky_relu=0.1):
        self.ANCHORS = anchors
        self.BN_DECAY = batch_norm_decay
        self.LEAKY_RELU = leaky_relu
        self.NUM_CLASSES = num_classes
        self.feature_maps = []

    def yolo_block(self, inputs, filters):
        """ conv-1-conv-3*2-conv-1(feature)-conv3(inputs)"""
        inputs = common.conv2d(inputs, filters * 1, 1)
        inputs = common.conv2d(inputs, filters * 2, 3)
        inputs = common.conv2d(inputs, filters * 1, 1)
        inputs = common.conv2d(inputs, filters * 2, 3)
        inputs = common.conv2d(inputs, filters * 1, 1)
        feature = inputs
        inputs = common.conv2d(inputs, filters * 2, 3)
        return feature, inputs

    def detection_layer(self, inputs, anchors):
        """Get the final score map"""
        num_anchors = len(anchors)
        feature_map = slim.conv2d(inputs, num_anchors * (5 + self.NUM_CLASSES), 1, stride=1,
                                  normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer())
        return feature_map

    def upsample(self, inputs, out_shape):
        """upsample a low dimensions feature map to a higher dimensions"""
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        inputs = tf.identity(inputs, name='upsampled')
        return inputs

    def inference(self, inputs, is_training=False):
        """
        Inference of yolo v3.
        :param inputs: a 4-D tensor of size[N,H,W,C]
        :param is_training: training or testing
        :param reuse:
        :return:
        """
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.BN_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,
        }
        # Set activation and bn and initializer.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common.conv2d]):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.LEAKY_RELU)):
                with tf.variable_scope('darknet-53'):
                    feature_1, feature_2, inputs = darknet53(inputs).outputs
                with tf.variable_scope('yolo-v3'):
                    feature, inputs = self.yolo_block(inputs, 512)
                    feature_map_1 = self.detection_layer(inputs, self.ANCHORS[6:9])
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inputs = common.conv2d(feature, 256, 1)
                    upsample_size = feature_2.get_shape().as_list()
                    inputs = self.upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, feature_2], axis=3)

                    feature, inputs = self.yolo_block(inputs, 256)
                    feature_map_2 = self.detection_layer(inputs, self.ANCHORS[3:6])
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inputs = common.conv2d(feature, 128, 1)
                    upsample_size = feature_1.get_shape().as_list()
                    inputs = self.upsample(inputs, upsample_size)
                    inputs = tf.concat([inputs, feature_1], axis=3)

                    feature, inputs = self.yolo_block(inputs, 128)
                    feature_map_3 = self.detection_layer(inputs, self.ANCHORS[0:3])
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')
            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        """reorganize feature map from final layers
        :return
        x_y_offset: a meshgrid of offset corresponding to final grid.
        boxes: [N,H,W,num_anchors,4(x_centers,y_centers,w,h)]

        """
        num_anchors = len(anchors)  # num_anchors=3
        grid_size = feature_map.shape.as_list()[1:3]
        # the downscale image in height and weight
        stride = tf.cast(self.img_size // grid_size, tf.float32)  # [h,w] -> [y,x]
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors,
                                               5 + self.NUM_CLASSES])       # shape:[N,grid_H,W,num_anchors,5+classes]

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.NUM_CLASSES],
                                                                    axis=-1)  # e.m.box_centers [N,H,W,num_anchors,2]
        box_centers = tf.nn.sigmoid(box_centers)

        # design a offset matrix
        grid_x = tf.range(grid_size[1], dtype=tf.int32)  # x shape(w,)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)  # y shape(h,)

        a, b = tf.meshgrid(grid_x, grid_y)  # shape(h,w)
        x_offset = tf.reshape(a, (-1, 1))  # shape(h*w,)
        y_offset = tf.reshape(b, (-1, 1))  # shape(h*w,)
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)  # shape(h*w,2)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])  # (h,w,1,2)
        x_y_offset = tf.cast(x_y_offset,tf.float32)

        box_centers = box_centers + x_y_offset  # predicted centers + the grid offset
        box_centers = box_centers * stride[::-1]  # rescale to original scale

        box_sizes = tf.exp(box_sizes) * anchors  # anchors ->[w,h]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, prob_logits

    def reshape(self, x_y_offset, boxes, confs, probs):
        grid_size = x_y_offset.shape.as_list()[:2]
        boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])  # anchors number=3
        confs = tf.reshape(confs, [-1, grid_size[0] * grid_size[1] * 3, 1])
        probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, self.NUM_CLASSES])

        return boxes, confs, probs

    def predict(self, feature_maps):
        """
        Note: compute the receptive field and get boxes,confs and class_probs
        given feature_maps
        feature_maps -> [None, 13, 13, 3*(4+1+num_class)],
                        [None, 26, 26, 255],
                        [None, 52, 52, 255],
        :return
        boxes [N,/sigma(3*grid_w*h),4]
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_maps_anchors = [(feature_map_1, self.ANCHORS[6:9]),
                                (feature_map_2, self.ANCHORS[3:6]),
                                (feature_map_3, self.ANCHORS[0:3]), ]

        results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_maps_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            boxes, conf_logits, prob_logits = self.reshape(*result)  # flatten feature map i.e boxes(N,3*grid_w*h,4)

            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)  # shape : [N,/sigma(3*grid_w*h),4]
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x0 = center_x - width / 2.
        y0 = center_y - height / 2.
        x1 = center_x + width / 2.
        y1 = center_y + height / 2.

        boxes = tf.concat([x0, y0, x1, y1], axis=-1)
        return boxes, confs, probs

    def comput_loss(self, y_pred, y_true, ignore_thresh=0.5, max_box_per_image=8):
        """Note: compute the loss
        Arguments:y_pred, list -> [feature_map_1, feature_map_2, feature_map_3]
                                        the shape of [None, 13, 13, 3*(NUM_CLASS+5)], etc
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.,
        total_loss = 0.
        ANCHORS = [self.ANCHORS[6:9], self.ANCHORS[3:6], self.ANCHORS[0:3]]

        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], ANCHORS[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]

        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def loss_layer(self, feature_map_i, y_true, anchors):
        # size in [h,w] format ! don't get messed up
        grid_size = tf.shape(feature_map_i)[1:3]
        grid_size_ = feature_map_i.shape.as_list()[1:3]

        y_true = tf.reshape(y_true, [-1, grid_size[0], grid_size[1], 3, 5 + self.NUM_CLASSES])

        # downscale ratio in height and width
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)
        # shape: take H*W input image and 13*13 feature mpa for example:
        # [N,13,13,3,1]
        object_mask = y_true[..., 4:5]
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

        # shape:[V,2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N,13, 13, 3, V]
        iou = self.iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        best_iou = tf.reduce_max(iou, axis=-1)
        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th, numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0), x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)

        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                    y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss = tf.reduce_sum(conf_loss_pos * conf_loss_neg) / N

        # shape: [N, 13, 13, 3, 1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                                           logits=pred_prob_logits)
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        """
        calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: only care about the size match
        """
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1,V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N,13, 13, 3, 1, 2] &[1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)

        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area)

        return iou

# ANCHORS = utils.get_anchors('../data/anchors.txt',416,416)
# images = tf.placeholder(dtype=tf.float32, shape=(1, 416, 416, 3))
# model=yolov3(1,ANCHORS)
# pred_feature_map = model.inference(images, is_training=tf.placeholder(tf.bool))
# # loss = model.comput_loss(pred_feature_map, y_true)
# y_pred = model.predict(pred_feature_map)
# c = 1
