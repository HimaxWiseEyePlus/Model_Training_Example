#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create YOLOv3 models with different backbone & head
"""
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa

from yolo3.models.yolo_fastest_xl import yolo_fastest_xl_body, yolo_fastest_body_body, yolo_fastest_body

from yolo3.loss import YoloLoss

from common.model_utils import add_metrics, get_pruning_model, get_clustering_model, get_qat_model
from core.config import cfg

# A map of model type to construction info list for YOLOv3
#
# info list format:
#   [model_function, backbone_length]
#
yolo3_model_map = {
}

# A map of model type to construction info list for Tiny YOLOv3
#
# info list format:
#   [model_function, backbone_length]
#
yolo3_tiny_model_map = {
    'yolo_fastest_xl': [yolo_fastest_xl_body, 268],
    'yolo_fastest_body': [yolo_fastest_body_body, 268],
    'yolo_fastest': [yolo_fastest_body, 267],
}


def get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_tensor=None, input_shape=None):
    # prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=cfg.YOLO.input_shape, name='image_input')

    # Tiny YOLOv3 model has 6 anchors and 2 feature layers
    if num_feature_layers == 2:
        if model_type in yolo3_tiny_model_map:
            model_function = yolo3_tiny_model_map[model_type][0]
            backbone_len = yolo3_tiny_model_map[model_type][1]
            model_body = model_function(
                input_tensor, num_anchors//2, num_classes)
        else:
            raise ValueError('This model type is not supported now')

    # YOLOv3 model has 9 anchors and 3 feature layers
    elif num_feature_layers == 3:
        if model_type in yolo3_model_map:
            model_function = yolo3_model_map[model_type][0]
            backbone_len = yolo3_model_map[model_type][1]
            model_body = model_function(
                input_tensor, num_anchors//3, num_classes)
        else:
            raise ValueError('This model type is not supported now')
    else:
        raise ValueError('model type mismatch anchors')

    return model_body, backbone_len


class SGDWCustomModel(Model):
    def train_step(self, data):
        x = data
        decay_var_list = []
        v = []
        for w in self.trainable_variables:
            if w.name.find("batch_normalization") >= 0:
                v.append(w)
            elif w.name.find("bias") >= 0:
                v.append(w)
            else:
                decay_var_list.append(w)

        # Compute gradients
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.losses

        # Update weights
        self.optimizer.minimize(
            loss, var_list=self.trainable_variables, decay_var_list=decay_var_list, tape=tape)
        #self.optimizer.minimize(loss, var_list=self.trainable_variables, tape=tape)
        # Update metrics (includes the metric that tracks the loss)
        return {m.name: m.result() for m in self.metrics}


def get_yolo3_train_model(model_type, anchors, num_classes, optimizer, weights_path=None, model_pruning=False, pruning_end_step=10000, max_boxes=200, model_clustering = False, model_qat = False, reload_path = None):
    '''create the training model, for YOLOv3'''
    num_anchors = len(anchors)
    # YOLOv3 model has 9 anchors and 3 feature layers but
    # Tiny YOLOv3 model has 6 anchors and 2 feature layers,
    # so we can calculate feature layers number to get model type
    num_feature_layers = num_anchors//3

    # feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]

    y_true_box = [Input(shape=(max_boxes, 4), name='y_true_box')]
    iou_thresh_mask = [Input(shape=(None, None, 3, max_boxes), name='iou_thresh_mask{}'.format(
        l)) for l in range(num_feature_layers)]
    true_class_probs = [Input(shape=(None, None, 3, num_classes), name='true_class_probs{}'.format(
        l)) for l in range(num_feature_layers)]

    model_body, backbone_len = get_yolo3_model(
        model_type, num_feature_layers, num_anchors, num_classes)

    # model_body.save("logs/fastest_AOS_COCO_person_re2/fastest_person_gray_160.h5")
    # model_body.summary()
    # exit()

    print('Create {} {} model with {} anchors and {} classes.'.format(
        'Tiny' if num_feature_layers == 2 else '', model_type, num_anchors, num_classes))
    print('model layer number:', len(model_body.layers))

    if weights_path:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    if model_pruning:
        model_body = get_pruning_model(
            model_body, begin_step=0, end_step=pruning_end_step)
    
    if model_clustering:
        model_body = get_clustering_model(model_body)

    if model_qat:
        model_body = get_qat_model(model_body)

    predictions = YoloLoss(anchors, num_classes, ignore_thresh=cfg.YOLO.ignore_thresh, name="predictions")(
        [*model_body.output, *y_true_box, *iou_thresh_mask, *true_class_probs])

    if isinstance(optimizer, tfa.optimizers.SGDW):
        print("SGDWCustomModel")
        model = SGDWCustomModel(inputs=[
                                model_body.input, *y_true_box, *iou_thresh_mask, *true_class_probs], outputs=predictions)
    else:
        model = Model(inputs=[model_body.input, *y_true_box, *
                      iou_thresh_mask, *true_class_probs], outputs=predictions)

    loss_dict = {'loss': model.output[0], 'location_loss': model.output[1],
                 'confidence_loss': model.output[2], 'class_loss': model.output[3]}
    add_metrics(model, loss_dict)

    model.compile(optimizer=optimizer, loss=None)

    if reload_path != None:
        model.load_weights(reload_path)

    return model, model_body
