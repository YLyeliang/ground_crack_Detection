import tensorflow as tf
from utility import common

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
        feature_map = slim.conv2d(inputs, num_anchors * (5 + self.NUM_CLASSES), 1, strides=1,
                                  normalizer_fn=None, activation_fn=None, biases_initializer=tf.zeros_initializer())
        return feature_map

    def upsample(self, inputs, out_shape):
        """upsample a low dimensions feature map to a higher dimensions"""
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))
        inputs = tf.identity(inputs, name='upsampled')
        return inputs

    def inference(self, inputs, is_training=False, reuse=False):
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
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common.conv2d], reuse=reuse):
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
        """reorganize feature map from final layers"""
        num_anchors = len(anchors)  # num_anchors=3
        grid_size = feature_map.shape.as_list()[1:3]
        # the downscale image in height and weight
        stride = tf.cast(self.img_size // grid_size, tf.float32)  # [h,w] -> [y,x]
        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors,
                                               5 + self.NUM_CLASSES])  # shape:[N,grid_H,W,num_anchors,5+classes]

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.NUM_CLASSES],
                                                                    axis=-1)  # e.m.box_centers [N,H,W,num_anchors,2]
        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)

        a, b = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])

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
        feature_maps -> [None, 13, 13, 255],
                        [None, 26, 26, 255],
                        [None, 52, 52, 255],
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        feature_maps_anchors = [(feature_map_1, self.ANCHORS[6:9]),
                                (feature_map_2, self.ANCHORS[3:6]),
                                (feature_map_3, self.ANCHORS[0:3]), ]

        results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_maps_anchors]
        boxes_list, confs_list, probs_list = [], [], []

        for result in results:
            boxes, conf_logits, prob_logits = self.reshape(*result)

            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)

            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        boxes = tf.concat(boxes_list, axis=1)


b = tf.placeholder(dtype=tf.float32, shape=(1, 512, 512, 3))
dark = darknet53(b)
c = 1
