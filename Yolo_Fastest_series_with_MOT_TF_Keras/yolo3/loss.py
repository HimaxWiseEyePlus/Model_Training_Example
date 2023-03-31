# -*- coding=utf-8 -*-
#!/usr/bin/python3

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from yolo3.postprocess import yolo3_decode


def box_iou(b1, b2):
    """
    Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def box_diou(b_true, b_pred, use_ciou=False):
    """
    Calculate DIoU/CIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, None, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    use_ciou: bool flag to indicate whether to use CIoU loss type

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, t)
    """
    b_true = K.expand_dims(b_true, 0)
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh/2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred = K.expand_dims(b_pred, -2)
    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh/2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]
    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # box center distance
    center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)

    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    if use_ciou:
        # calculate param v and alpha to extend to CIoU
        v = 4*K.square(tf.math.atan2(b_true_wh[..., 0], b_true_wh[..., 1]) - tf.math.atan2(
            b_pred_wh[..., 0], b_pred_wh[..., 1])) / (math.pi * math.pi)
        # a trick: here we add an non-gradient coefficient w^2+h^2 to v to customize it's back-propagate,
        #          to match related description for equation (12) in original paper
        #
        #
        #          v'/w' = (8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (h/(w^2+h^2))          (12)
        #          v'/h' = -(8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (w/(w^2+h^2))
        #
        #          The dominator w^2+h^2 is usually a small value for the cases
        #          h and w ranging in [0; 1], which is likely to yield gradient
        #          explosion. And thus in our implementation, the dominator
        #          w^2+h^2 is simply removed for stable convergence, by which
        #          the step size 1/(w^2+h^2) is replaced by 1 and the gradient direction
        #          is still consistent with Eqn. (12).
        # v *= tf.squeeze(tf.stop_gradient(tf.square(b_pred_wh[..., 0]) + tf.square(b_pred_wh[..., 1])), axis=-1)

        alpha = tf.stop_gradient(v / (1.0 - iou + v + K.epsilon()))
        diou = diou - alpha * v

    return diou, iou


class YoloLoss(tf.keras.layers.Layer):
    def __init__(self, anchors, num_classes, ignore_thresh, name=None):
        super(YoloLoss, self).__init__(name=name)
        '''
        Parameters
        ----------
        anchors: array, shape=(N, 2), wh
        num_classes: integer
        ignore_thresh: float, the iou threshold whether to ignore object confidence loss
        '''
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh

    def yolo3_loss(self, output, y_true_box, iou_thresh_mask_l, true_class_probs_l):
        '''
        YOLOv3 loss function.

        Parameters
        ----------
        output: list of tensor, the output of yolo_body or tiny_yolo_body
        y_true_box: list of array, ground truth box
        iou_thresh_mask_l: IoU loss mask
        true_class_probs_l: class label

        Returns
        -------
        loss: tensor, shape=(1,)
        '''
        num_layers = len(self.anchors)//3  # default setting

        yolo_outputs = output
        y_true_box = y_true_box

        if num_layers == 3:
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            scale_x_y = [None, None, None]
        else:
            anchor_mask = [[3, 4, 5], [0, 1, 2]]
            scale_x_y = [None, None]

        input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, tf.float32)

        loss = 0
        total_location_loss = 0
        total_confidence_loss = 0
        total_class_loss = 0

        batch_size = K.shape(yolo_outputs[0])[0]  # batch size, tensor
        batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

        raw_pred_all = []
        pred_box_all = []
        ciou_all = []
        iou_all = []

        for i in range(num_layers):

            _, raw_pred, pred_xy, pred_wh = yolo3_decode(yolo_outputs[i],
                                                         self.anchors[anchor_mask[i]], self.num_classes, input_shape, scale_x_y=scale_x_y[i], calc_loss=True)
            pred_box = K.concatenate([pred_xy, pred_wh])

            ciou = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            iou = tf.TensorArray(tf.float32, size=1, dynamic_size=True)

            def iou_loop_body(b, ciou, iou):
                true_box = y_true_box[b]
                ciou_b, iou_b = box_diou(true_box, pred_box[b], True)
                ciou = ciou.write(b, ciou_b)
                iou = iou.write(b, iou_b)
                return b+1, ciou, iou
            _, ciou, iou = tf.while_loop(
                lambda b, *args: b < batch_size, iou_loop_body, [0, ciou, iou])

            ciou = ciou.stack()
            iou = iou.stack()

            raw_pred_all.append(raw_pred)
            pred_box_all.append(pred_box)
            ciou_all.append(ciou)
            iou_all.append(iou)

        for i in range(num_layers):

            iou_thresh_mask = iou_thresh_mask_l[i]
            true_class_probs = true_class_probs_l[i]

            pred_box = pred_box_all[i]
            raw_pred = raw_pred_all[i]

            best_iou = K.max(iou_all[i], axis=-1)

            ignore_mask = best_iou < self.ignore_thresh
            ignore_mask = K.expand_dims(ignore_mask, -1)

            ignore_match_mask = best_iou > self.ignore_thresh
            ignore_match_mask = K.expand_dims(ignore_match_mask, -1)
            class_id_match = tf.stop_gradient(tf.math.logical_and(K.expand_dims(
                K.max(K.sigmoid(raw_pred[..., 5:]), axis=-1) < 0.25, -1), ignore_match_mask))

            ignore_mask = K.cast(tf.math.logical_or(
                ignore_mask, class_id_match), tf.float32)
            #ignore_mask = K.cast(ignore_mask, tf.float32)

            ciou_loss = 1 - ciou_all[i]
            ciou_loss = ciou_loss * iou_thresh_mask
            ciou_loss = K.sum(ciou_loss, axis=-1, keepdims=True)

            iou_thresh_mask = K.max(iou_thresh_mask, axis=-1, keepdims=True)
            true_objectness_probs = iou_thresh_mask
            confidence_loss = true_objectness_probs * K.binary_crossentropy(true_objectness_probs, raw_pred[..., 4:5], from_logits=True) + \
                (1-true_objectness_probs) * K.binary_crossentropy(true_objectness_probs,
                                                                  raw_pred[..., 4:5], from_logits=True) * ignore_mask

            class_loss = K.binary_crossentropy(
                true_class_probs, raw_pred[..., 5:], from_logits=True)
            class_loss = true_objectness_probs * class_loss

            class_sum = K.sum(true_class_probs, axis=-1, keepdims=True)
            class_no_flag = K.cast(class_sum == 0, tf.float32)
            class_sum = class_sum + class_no_flag
            ciou_loss = ciou_loss / class_sum
            ciou_loss = K.sum(ciou_loss) / batch_size_f
            location_loss = ciou_loss

            confidence_loss = K.sum(confidence_loss) / batch_size_f
            class_loss = K.sum(class_loss) / batch_size_f

            loss += location_loss + confidence_loss + class_loss
            total_location_loss += location_loss
            total_confidence_loss += confidence_loss
            total_class_loss += class_loss

        return loss, total_location_loss, total_confidence_loss, total_class_loss

    def call(self, args):
        num_layers = len(self.anchors)//3  # default setting
        output = args[:num_layers]
        y_true_box = args[num_layers]
        iou_thresh_mask = args[num_layers+1:num_layers*2+1]
        true_class_probs = args[num_layers*2+1:num_layers*3+1]

        loss, total_location_loss, total_confidence_loss, total_class_loss = self.yolo3_loss(
            output, y_true_box, iou_thresh_mask, true_class_probs)
        self.add_loss(loss)

        return loss, total_location_loss, total_confidence_loss, total_class_loss

    def get_config(self):
        config = super().get_config().copy()
        config.update({'anchors': self.anchors,
                       'num_classes': self.num_classes,
                       'ignore_thresh': self.ignore_thresh,
                       'name': self.name})

        return config
