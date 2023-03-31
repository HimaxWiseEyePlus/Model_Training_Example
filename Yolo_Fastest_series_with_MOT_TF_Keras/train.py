#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain the YOLO model for your own dataset.
"""
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os
import argparse
import numpy as np
import tensorflow.keras.backend as K
# from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN
# from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.api.sparsity import keras as sparsity


from yolo3.model import get_yolo3_train_model
from yolo3.data import yolo3_data_generator_wrapper

from common.utils import get_classes, get_anchors, get_dataset, optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import MOTSaveCallBack

from core.config import cfg

# Try to enable Auto Mixed Precision on TF 2.0
# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
# os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tf.compat.v1.disable_eager_execution()

optimize_tf_gpu(tf, K)


def main(args):
    annotation_file = args.annotation_file
    log_dir = os.path.join('logs', args.log_path)
    classes_path = args.classes_path
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    anchors = get_anchors(args.anchors_path)
    num_anchors = len(anchors)

    # callbacks for training process
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False,
                          write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                                 monitor='val_loss',
                                 mode='min',
                                 verbose=1,
                                 save_weights_only=False,
                                 save_best_only=False,
                                 period=1)
    terminate_on_nan = TerminateOnNaN()
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')

    callbacks = [
        logging,
        checkpoint,
        terminate_on_nan,
        # early_stopping,
    ]

    # get train&val dataset
    dataset = get_dataset(annotation_file)
    if args.val_annotation_file:
        val_dataset = get_dataset(args.val_annotation_file)
        num_train = len(dataset)
        num_val = len(val_dataset)
        dataset.extend(val_dataset)
    else:
        val_split = args.val_split
        num_val = int(len(dataset)*val_split)
        num_train = len(dataset) - num_val

    # assign multiscale interval
    if args.multiscale:
        rescale_interval = args.rescale_interval
    else:
        rescale_interval = -1  # Doesn't rescale

    # model input shape check
    input_shape = args.model_image_size
    assert (input_shape[0] % 32 == 0 and input_shape[1] %
            32 == 0), 'model_image_size should be multiples of 32'

    # get different model type & train&val data generator
    if args.model_type.startswith('yolo_fastest'):
        get_train_model = get_yolo3_train_model
        data_generator = yolo3_data_generator_wrapper

    else:
        raise ValueError('Unsupported model type')

    # prepare model pruning config
    pruning_end_step = np.ceil(
        1.0 * num_train / args.batch_size).astype(np.int32) * args.total_epoch
    if args.model_pruning:
        pruning_callbacks = [sparsity.UpdatePruningStep(
        ), sparsity.PruningSummaries(log_dir=log_dir, profile_batch=0)]
        callbacks = callbacks + pruning_callbacks

    if args.model_pruning or args.model_clustering:
        motsave_callback = MOTSaveCallBack(
            args.save_epoch_interval, args.model_pruning, args.model_clustering, log_dir)
        callbacks.append(motsave_callback)

    steps_per_epoch = max(1, num_train//args.batch_size)
    decay_steps = steps_per_epoch * args.total_epoch
    initial_steps = steps_per_epoch * args.init_epoch

    optimizer = get_optimizer(args.optimizer, args.learning_rate, total_epoch=args.total_epoch,
                              decay_type=args.decay_type, decay_steps=decay_steps, initial_steps=initial_steps)

    if args.gpu_num >= 2:
        devices_list = ["/gpu:{}".format(n) for n in range(args.gpu_num)]
        strategy = tf.distribute.MirroredStrategy(devices=devices_list)
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model, model_body = get_train_model(args.model_type, anchors, num_classes, weights_path=args.weights_path,
                                    optimizer=optimizer, model_pruning=args.model_pruning, pruning_end_step=pruning_end_step, model_clustering=args.model_clustering, model_qat=args.model_qat, reload_path = args.reload_path)
    else:
        model, model_body = get_train_model(args.model_type, anchors, num_classes, weights_path=args.weights_path,
                                optimizer=optimizer, model_pruning=args.model_pruning, pruning_end_step=pruning_end_step, model_clustering=args.model_clustering, model_qat=args.model_qat, reload_path = args.reload_path)

    print('Train on {} samples, val on {} samples, with batch size {}, input_shape {}.'.format(
        num_train, num_val, args.batch_size, input_shape))
    model.fit(data_generator(dataset[:num_train], args.batch_size, input_shape, anchors, num_classes, rescale_interval, multi_anchor_assign=args.multi_anchor_assign, augment=True),
              steps_per_epoch=max(1, num_train//args.batch_size),
              validation_data=data_generator(dataset[num_train:], args.batch_size, input_shape,
                                             anchors, num_classes, rescale_interval, multi_anchor_assign=args.multi_anchor_assign, augment=False),
              validation_steps=max(1, num_val//args.batch_size),
              epochs=args.total_epoch,
              initial_epoch=args.init_epoch,
              workers=6,
              use_multiprocessing=True,
              max_queue_size=128,
              callbacks=callbacks)

    # Finally store model
    if args.model_pruning:
        model = sparsity.strip_pruning(model)
    elif args.model_clustering:
        model = tfmot.clustering.keras.strip_clustering(model)

    model.save(os.path.join(log_dir, 'trained_final.h5'))
    model_body.save(os.path.join(log_dir, 'trained_model_body_final.h5'))
    
    if args.model_qat:
        converter = tf.lite.TFLiteConverter.from_keras_model(model_body)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        
        qat_tflite_model = converter.convert()
        qat_model_file = os.path.join(log_dir, cfg.MOT.qat_mode+ '.tflite')
        # Save the model.
        with open(qat_model_file, 'wb') as f:
            f.write(qat_tflite_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='yolo_fastest_xl',
                        help='YOLO model type: yolo_fastest_xl/..., default=%(default)s')
    parser.add_argument('--anchors_path', type=str, required=False, default=os.path.join('cfg', 'yolo-fastest-xl-anchors.txt'),
                        help='path to anchor definitions, default=%(default)s')
    parser.add_argument('--model_image_size', type=str, required=False, default='320x320',
                        help="Initial model image input size as <height>x<width>, default=%(default)s")
    parser.add_argument('--weights_path', type=str, required=False, default=None,
                        help="Pretrained model/weights file for fine tune")
    parser.add_argument('--reload_path', type=str, required=False, default=None,
                        help="reload weights file")

    # Data options
    parser.add_argument('--annotation_file', type=str, required=False, default='./tools/dataset_label/linux/train2017.txt',
                        help='train annotation txt file, default=%(default)s')
    parser.add_argument('--log_path', type=str, required=True, default='./',
                        help='log path, default=%(default)s')
    parser.add_argument('--val_annotation_file', type=str, required=False, default='./tools/dataset_label/linux/val2017.txt',
                        help='val annotation txt file, default=%(default)s')
    # parser.add_argument('--val_split', type=float, required=False, default=0.1,
    #                     help="validation data persentage in dataset if no val dataset provide, default=%(default)s")
    parser.add_argument('--classes_path', type=str, required=False, default=os.path.join('configs', 'coco_classes.txt'),
                        help='path to class definitions, default=%(default)s')

    # Training options
    parser.add_argument('--batch_size', type=int, required=False, default=32,
                        help="Batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='sgd', choices=['adam', 'rmsprop', 'sgd','sgdw','sgd0'],
                        help="optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
                        help="Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default='piecewise_constant', choices=[None, 'none', 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
                        help="Learning rate decay type, default=%(default)s")
    parser.add_argument('--init_epoch', type=int, required=False, default=0,
                        help="Initial training epochs for fine tune training, default=%(default)s")
    parser.add_argument('--total_epoch', type=int, required=False, default=270,
                        help="Total training epochs, default=%(default)s")
    parser.add_argument('--multiscale', default=False, action="store_true",
                        help='Whether to use multiscale training')
    parser.add_argument('--rescale_interval', type=int, required=False, default=10,
                        help="Number of iteration(batches) interval to rescale input size, default=%(default)s")
    parser.add_argument('--multi_anchor_assign', default=False, action="store_true",
                        help="Assign multiple anchors to single ground truth")
    parser.add_argument('--gpu_num', type=int, required=False, default=1,
                        help='Number of GPU to use, default=%(default)s')

    # MOT
    parser.add_argument('--model_pruning', default=False, action="store_true",
                        help='Use model pruning for optimization')
    parser.add_argument('--model_clustering', default=False, action="store_true",
                        help='Use model clustering for optimization')
    parser.add_argument('--model_qat', default=False, action="store_true",
                        help='Use model QAT for optimization')
    parser.add_argument('--save_epoch_interval', type=int, required=False, default=1,
                        help="Number of iteration(epochs) interval to do save, default=%(default)s")

    args = parser.parse_args()
    height, width = args.model_image_size.split('x')
    args.model_image_size = (int(height), int(width))

    main(args)
