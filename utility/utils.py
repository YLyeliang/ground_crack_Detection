import colorsys
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw
from collections import Counter


def resize_image_correct_bbox(image, boxes, image_h, image_w):
    origin_image_size = tf.to_float(tf.shape(image)[0:2])
    image = tf.image.resize_images(image, size=[image_h, image_w])

    # correct bbox
    xx1 = boxes[:, 0] * image_w / origin_image_size[1]
    yy1 = boxes[:, 1] * image_h / origin_image_size[0]
    xx2 = boxes[:, 2] * image_w / origin_image_size[1]
    yy2 = boxes[:, 3] * image_h / origin_image_size[0]
    idx = boxes[:, 4]

    boxes = tf.stack([xx1, yy1, xx2, yy2, idx], axis=1)
    return image, boxes


def bbox_iou(pred, true):
    """ compute iou between pred box and true boxes"""
    intersect_mins = np.maximum(pred[:, 0:2], true[:, 0:2])
    intersect_maxs = np.minimum(pred[:, 2:4], true[:, 2:4])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_area = np.prod(pred[:, 2:4] - pred[:, 0:2], axis=1)
    true_area = np.prod(true[:, 2:4] - true[:, 0:2], axis=1)

    iou = intersect_area / (pred_area + true_area - intersect_area)

    return iou


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure python NMS baseline.
    :param boxes: shape [-1,4]
    :param scores: shape[-1,1]
    :param max_boxes: representing the maximum of boxes to be selected by non_max_suppression.
    :param iou_thresh: representing the iou_threshold for deciding to keep boxes
    :return:
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # from big to small

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # intersection
        xx1 = np.maximum(x1[i], x1[order[1:]])  # compare the coordinates of box having biggest score  with remains.
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # box having biggest score's iou with remain boxes

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
    """
    Non-Maximum suppression
    :param boxes: shape [1,/sigma(3*w*h), 4]
    :param scores: shape [1, 10647, num_classes]
    :return
    boxes: shape [keep_boxes_number,4]
    """
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # picked bboxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:, i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:, i][indices]
        if len(filter_boxes) == 0: continue
        # do non_max suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores, max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32') * i)
    if len(picked_boxes) == 0: return None, None, None

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def evaluate(y_pred, y_true, iou_thresh=0.5, score_thresh=0.3):
    """

    :param y_pred: a tensor contains [boxes,confs,probs] boxes:[N,/sigma(3*w*h),4]
    :param y_true: [3,N,grid_h,grid_w,3,4+1+num_classes]
    :param iou_thresh:
    :param score_thresh:
    :return:
    """
    num_images = y_true[0].shape[0]  # N
    num_classes = y_true[0][0][..., 5:].shape[-1]
    true_labels_dict = {i: 0 for i in range(num_classes)}  # {class: count}
    pred_labels_dict = {i: 0 for i in range(num_classes)}
    true_positive_dict = {i: 0 for i in range(num_classes)}

    for i in range(num_images):  # ith of batch_size N
        true_labels_list, true_boxes_list = [], []
        for j in range(3):  # three feature maps     jth feature map of 3
            true_probs_temp = y_true[j][i][..., 5:]
            true_boxes_temp = y_true[j][i][..., 0:4]

            object_mask = true_probs_temp.sum(
                axis=-1) > 0  # keep those num probs >0 i.e keep cells that have object in true labels.

            true_probs_temp = true_probs_temp[object_mask]
            true_boxes_temp = true_boxes_temp[object_mask]

            true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
            true_boxes_list += true_boxes_temp.tolist()

        if len(true_labels_list) != 0:
            for cls, count in Counter(true_labels_list).items(): true_labels_dict[cls] += count

        pred_boxes = y_pred[0][i:i + 1]
        pred_confs = y_pred[1][i:i + 1]
        pred_probs = y_pred[2][i:i + 1]

        pred_boxes, pred_scores, pred_labels = nms(pred_boxes, pred_confs * pred_probs, num_classes,
                                                   max_boxes=10, score_thresh=score_thresh, iou_thresh=iou_thresh)

        true_boxes = np.array(true_boxes_list)
        box_centers, box_sizes = true_boxes[:, 0:2], true_boxes[:, 2:4]

        true_boxes[:, 0:2] = box_centers - box_sizes / 2.
        true_boxes[:, 2:4] = true_boxes[:, 0:2] + box_sizes
        pred_labels_list = [] if pred_labels is None else pred_labels.tolist()

        if len(pred_labels_list) != 0:
            for cls, count in Counter(pred_labels_list).items(): pred_labels_dict[cls] += count
        else:
            continue

        detected = []
        for k in range(len(pred_labels_list)):
            # compute iou between predicted box and ground_truth boxes
            iou = bbox_iou(pred_boxes[k:k + 1], true_boxes)
            m = np.argmax(iou)  # Extract index of largest overlap
            if iou[m] >= iou_thresh and pred_labels_list[k] == true_labels_list[m] and m not in detected:
                true_positive_dict[true_labels_list[m]] += 1
                detected.append(m)

    recall = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
    precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)

    return recall, precision


def get_anchors(anchors_path, image_h, image_w):
    """ loads the anchors from a txt file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(), dtype=np.float32)
    anchors = anchors.reshape(-1, 2)
    anchors[:, 1] = anchors[:, 1] * image_h
    anchors[:, 0] = anchors[:, 0] * image_w
    return anchors


def draw_boxes(image, boxes, scores, labels, classes, detection_size,
               font='./data/font/FiraMono-Medium.otf', show=True):
    """
    draw boxes on image
    :param boxes: shape of [num, 4]
    :param scores: shape of [num, ]
    :param labels: shape of [num, ]
    :param classes: class names.
    :param detection_size:
    :param show:
    :return: image
    """
    if boxes is None: return image
    draw = ImageDraw.Draw(image)
    # draw settings
    font = ImageFont.truetype(font=font, size=np.floor(2e-2 * image.size[1]).astype('int32'))
    hsv_tuples = [(x / len(classes), 0.9, 1.0) for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    for i in range(len(labels)):  # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" % (label, score)
        text_size = draw.textsize(bbox_text, font)
        # convert_to_original_size
        detection_size, original_size = np.array(detection_size), np.array(image.size)
        ratio = original_size / detection_size
        bbox = list((bbox.reshape(2, 2) * ratio).reshape(-1))

        draw.rectangle(bbox, outline=colors[labels[i]])
        text_origin = bbox[:2] - np.array([0, text_size[1]])
        draw.rectangle([tuple(text_origin), tuple(text_origin + text_size)], fill=colors[labels[i]])
        # # draw bbox
        draw.text(tuple(text_origin), bbox_text, fill=(0, 0, 0), font=font)

    image.show() if show else None
    return image
