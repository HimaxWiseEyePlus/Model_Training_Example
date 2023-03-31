#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO_v3 Darknet Model Defined in Keras."""

from hashlib import new
import tensorflow as tf 
from tensorflow.keras.layers import Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Dropout, Resizing
from tensorflow.keras.models import Model

from yolo3.models.layers import compose, DarknetConv2D, DarknetConv2D_BN_Leaky, Darknet_Depthwise_Conv2D_BN_Leaky
from common.backbones.layers import CustomBatchNormalization
from core.config import cfg
    

def yolo_fastext_conv_block_withpad(input_filter, output_filter):
    return compose(
        DarknetConv2D_BN_Leaky(input_filter, (1, 1),
                               padding='valid', strides=(1, 1)),
        ZeroPadding2D(((1, 0), (1, 0))),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (3, 3), padding='valid', strides=(2, 2)),
        DarknetConv2D(output_filter, (1, 1), padding='valid',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization()
    )


def yolo_fastext_conv_block(input_filter, output_filter):
    return compose(
        DarknetConv2D_BN_Leaky(input_filter, (1, 1),
                               padding='valid', strides=(1, 1)),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (3, 3), padding='same', strides=(1, 1)),
        DarknetConv2D(output_filter, (1, 1), padding='valid',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization()
    )


def yolo_fastest_xl_body(inputs, num_anchors, num_classes):
    '''Create YOLO Fastest xl model CNN body in keras.'''

    head = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        DarknetConv2D_BN_Leaky(16, (3, 3), padding='valid', strides=(2, 2)),
    )(inputs)

    block_1 = yolo_fastext_conv_block(16, 8)(head)
    block_2 = yolo_fastext_conv_block(16, 8)(block_1)
    block_2_drop = Dropout(0.2)(block_2)
    block_2_add = Add()([block_1, block_2_drop])

    block_3 = yolo_fastext_conv_block_withpad(48, 16)(block_2_add)
    block_4 = yolo_fastext_conv_block(64, 16)(block_3)
    block_4_drop = Dropout(0.2)(block_4)
    block_4_add = Add()([block_3, block_4_drop])
    block_5 = yolo_fastext_conv_block(64, 16)(block_4_add)
    block_5_drop = Dropout(0.2)(block_5)
    block_5_add = Add()([block_4_add, block_5_drop])

    block_6 = yolo_fastext_conv_block_withpad(64, 16)(block_5_add)
    block_7 = yolo_fastext_conv_block(96, 16)(block_6)
    block_7_drop = Dropout(0.2)(block_7)
    block_7_add = Add()([block_6, block_7_drop])
    block_8 = yolo_fastext_conv_block(96, 16)(block_7_add)
    block_8_drop = Dropout(0.2)(block_8)
    block_8_add = Add()([block_7_add, block_8_drop])

    block_9 = yolo_fastext_conv_block(96, 32)(block_8_add)
    block_10 = yolo_fastext_conv_block(192, 32)(block_9)
    block_10_drop = Dropout(0.2)(block_10)
    block_10_add = Add()([block_9, block_10_drop])
    block_11 = yolo_fastext_conv_block(192, 32)(block_10_add)
    block_11_drop = Dropout(0.2)(block_11)
    block_11_add = Add()([block_10_add, block_11_drop])
    block_12 = yolo_fastext_conv_block(192, 32)(block_11_add)
    block_12_drop = Dropout(0.2)(block_12)
    block_12_add = Add()([block_11_add, block_12_drop])
    block_13 = yolo_fastext_conv_block(192, 32)(block_12_add)
    block_13_drop = Dropout(0.2)(block_13)
    block_13_add = Add()([block_12_add, block_13_drop])

    block_14 = yolo_fastext_conv_block_withpad(192, 48)(block_13_add)
    block_15 = yolo_fastext_conv_block(272, 48)(block_14)
    block_15_drop = Dropout(0.2)(block_15)
    block_15_add = Add()([block_14, block_15_drop])
    block_16 = yolo_fastext_conv_block(272, 48)(block_15_add)
    block_16_drop = Dropout(0.2)(block_16)
    block_16_add = Add()([block_15_add, block_16_drop])
    block_17 = yolo_fastext_conv_block(272, 48)(block_16_add)
    block_17_drop = Dropout(0.2)(block_17)
    block_17_add = Add()([block_16_add, block_17_drop])
    block_18 = yolo_fastext_conv_block(272, 48)(block_17_add)
    block_18_drop = Dropout(0.2)(block_18)
    block_18_add = Add()([block_17_add, block_18_drop])
    # concat

    block_19 = yolo_fastext_conv_block_withpad(272, 96)(block_18_add)
    block_20 = yolo_fastext_conv_block(448, 96)(block_19)
    block_20_drop = Dropout(0.2)(block_20)
    block_20_add = Add()([block_19, block_20_drop])
    block_21 = yolo_fastext_conv_block(448, 96)(block_20_add)
    block_21_drop = Dropout(0.2)(block_21)
    block_21_add = Add()([block_20_add, block_21_drop])
    block_22 = yolo_fastext_conv_block(448, 96)(block_21_add)
    block_22_drop = Dropout(0.2)(block_22)
    block_22_add = Add()([block_21_add, block_22_drop])
    block_23 = yolo_fastext_conv_block(448, 96)(block_22_add)
    block_23_drop = Dropout(0.2)(block_23)
    block_23_add = Add()([block_22_add, block_23_drop])
    block_24 = yolo_fastext_conv_block(448, 96)(block_23_add)
    block_24_drop = Dropout(0.2)(block_24)
    block_24_add = Add()([block_23_add, block_24_drop])

    # SPP maxpool
    spp1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                        padding='same')(block_24_add)
    spp2 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1),
                        padding='same')(block_24_add)
    spp3 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1),
                        padding='same')(block_24_add)

    spp_concate = Concatenate()([spp3, spp2, spp1, block_24_add])
    spp_final = DarknetConv2D_BN_Leaky(
        96, (1, 1), padding='valid', strides=(1, 1))(spp_concate)

    branch_1 = compose(
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(96, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(96, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1),
                      padding='same', strides=(1, 1), name='predict_conv_1')
    )(spp_final)

    if cfg.YOLO.fixshape:
        # method = tf.image.ResizeMethod.BILINEAR
        # spp_upsample = tf.image.resize(spp_final, [20,20], method)
        spp_upsample = Resizing(int(cfg.YOLO.input_shape[0]/32 *2), int(cfg.YOLO.input_shape[0]/32 *2), interpolation="bilinear", crop_to_aspect_ratio=False)(spp_final)
    else:
        spp_upsample = UpSampling2D((2, 2), interpolation="bilinear")(spp_final)
        
    #spp_upsample = tf.compat.v1.image.resize(spp_final, [20,20], method)
    spp_branch2_concate = Concatenate()([spp_upsample, block_18_add])

    branch_2 = compose(
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(144, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(144, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1),
                      padding='same', strides=(1, 1), name='predict_conv_2')
    )(spp_branch2_concate)

    return Model(inputs, [branch_1, branch_2])


def yolo_fastest_body_body(inputs, num_anchors, num_classes):
    '''Create YOLO Fastest model CNN body in keras.'''

    head = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        DarknetConv2D_BN_Leaky(8, (3, 3), padding='valid', strides=(2, 2)),
    )(inputs)

    block_1 = yolo_fastext_conv_block(8, 4)(head)
    block_2 = yolo_fastext_conv_block(8, 4)(block_1)
    block_2_drop = Dropout(0.15)(block_2)
    block_2_add = Add()([block_1, block_2_drop])

    block_3 = yolo_fastext_conv_block_withpad(24, 8)(block_2_add)
    block_4 = yolo_fastext_conv_block(32, 8)(block_3)
    block_4_drop = Dropout(0.15)(block_4)
    block_4_add = Add()([block_3, block_4_drop])
    block_5 = yolo_fastext_conv_block(32, 8)(block_4_add)
    block_5_drop = Dropout(0.15)(block_5)
    block_5_add = Add()([block_4_add, block_5_drop])

    block_6 = yolo_fastext_conv_block_withpad(32, 8)(block_5_add)
    block_7 = yolo_fastext_conv_block(48, 8)(block_6)
    block_7_drop = Dropout(0.15)(block_7)
    block_7_add = Add()([block_6, block_7_drop])
    block_8 = yolo_fastext_conv_block(48, 8)(block_7_add)
    block_8_drop = Dropout(0.15)(block_8)
    block_8_add = Add()([block_7_add, block_8_drop])

    block_9 = yolo_fastext_conv_block(48, 16)(block_8_add)
    block_10 = yolo_fastext_conv_block(96, 16)(block_9)
    block_10_drop = Dropout(0.15)(block_10)
    block_10_add = Add()([block_9, block_10_drop])
    block_11 = yolo_fastext_conv_block(96, 16)(block_10_add)
    block_11_drop = Dropout(0.15)(block_11)
    block_11_add = Add()([block_10_add, block_11_drop])
    block_12 = yolo_fastext_conv_block(96, 16)(block_11_add)
    block_12_drop = Dropout(0.15)(block_12)
    block_12_add = Add()([block_11_add, block_12_drop])
    block_13 = yolo_fastext_conv_block(96, 16)(block_12_add)
    block_13_drop = Dropout(0.15)(block_13)
    block_13_add = Add()([block_12_add, block_13_drop])

    block_14 = yolo_fastext_conv_block_withpad(96, 24)(block_13_add)
    block_15 = yolo_fastext_conv_block(136, 24)(block_14)
    block_15_drop = Dropout(0.15)(block_15)
    block_15_add = Add()([block_14, block_15_drop])
    block_16 = yolo_fastext_conv_block(136, 24)(block_15_add)
    block_16_drop = Dropout(0.15)(block_16)
    block_16_add = Add()([block_15_add, block_16_drop])
    block_17 = yolo_fastext_conv_block(136, 24)(block_16_add)
    block_17_drop = Dropout(0.15)(block_17)
    block_17_add = Add()([block_16_add, block_17_drop])
    block_18 = yolo_fastext_conv_block(136, 24)(block_17_add)
    block_18_drop = Dropout(0.15)(block_18)
    block_18_add = Add()([block_17_add, block_18_drop])
    # concat

    block_19 = yolo_fastext_conv_block_withpad(136, 48)(block_18_add)
    block_20 = yolo_fastext_conv_block(224, 48)(block_19)
    block_20_drop = Dropout(0.15)(block_20)
    block_20_add = Add()([block_19, block_20_drop])
    block_21 = yolo_fastext_conv_block(224, 48)(block_20_add)
    block_21_drop = Dropout(0.15)(block_21)
    block_21_add = Add()([block_20_add, block_21_drop])
    block_22 = yolo_fastext_conv_block(224, 48)(block_21_add)
    block_22_drop = Dropout(0.15)(block_22)
    block_22_add = Add()([block_21_add, block_22_drop])
    block_23 = yolo_fastext_conv_block(224, 48)(block_22_add)
    block_23_drop = Dropout(0.15)(block_23)
    block_23_add = Add()([block_22_add, block_23_drop])
    block_24 = yolo_fastext_conv_block(224, 48)(block_23_add)
    block_24_drop = Dropout(0.15)(block_24)
    block_24_add = Add()([block_23_add, block_24_drop])

    # SPP maxpool
    spp1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                        padding='same')(block_24_add)
    spp2 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1),
                        padding='same')(block_24_add)
    spp3 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1),
                        padding='same')(block_24_add)

    spp_concate = Concatenate()([spp3, spp2, spp1, block_24_add])
    spp_final = DarknetConv2D_BN_Leaky(
        96, (1, 1), padding='valid', strides=(1, 1))(spp_concate)

    branch_1 = compose(
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(96, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(96, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1),
                      padding='same', strides=(1, 1), name='predict_conv_1')
    )(spp_final)


    if cfg.YOLO.fixshape:
        # method = tf.image.ResizeMethod.BILINEAR
        # spp_upsample = tf.image.resize(spp_final, [20,20], method)
        spp_upsample = Resizing(int(cfg.YOLO.input_shape[0]/32 *2), int(cfg.YOLO.input_shape[0]/32 *2), interpolation="bilinear", crop_to_aspect_ratio=False)(spp_final)
    else:
        spp_upsample = UpSampling2D((2, 2), interpolation="bilinear")(spp_final)
        
    spp_branch2_concate = Concatenate()([spp_upsample, block_18_add])

    branch_2 = compose(
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(120, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(120, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1),
                      padding='same', strides=(1, 1), name='predict_conv_2')
    )(spp_branch2_concate)

    return Model(inputs, [branch_1, branch_2])


def yolo_fastest_body(inputs, num_anchors, num_classes):
    '''Create YOLO Fastest model CNN body in keras.'''

    head = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        DarknetConv2D_BN_Leaky(8, (3, 3), padding='valid', strides=(2, 2)),
    )(inputs)

    block_1 = yolo_fastext_conv_block(8, 4)(head)
    block_2 = yolo_fastext_conv_block(8, 4)(block_1)
    block_2_drop = Dropout(0.15)(block_2)
    block_2_add = Add()([block_1, block_2_drop])

    block_3 = yolo_fastext_conv_block_withpad(24, 8)(block_2_add)
    block_4 = yolo_fastext_conv_block(32, 8)(block_3)
    block_4_drop = Dropout(0.15)(block_4)
    block_4_add = Add()([block_3, block_4_drop])
    block_5 = yolo_fastext_conv_block(32, 8)(block_4_add)
    block_5_drop = Dropout(0.15)(block_5)
    block_5_add = Add()([block_4_add, block_5_drop])

    block_6 = yolo_fastext_conv_block_withpad(32, 8)(block_5_add)
    block_7 = yolo_fastext_conv_block(48, 8)(block_6)
    block_7_drop = Dropout(0.15)(block_7)
    block_7_add = Add()([block_6, block_7_drop])
    block_8 = yolo_fastext_conv_block(48, 8)(block_7_add)
    block_8_drop = Dropout(0.15)(block_8)
    block_8_add = Add()([block_7_add, block_8_drop])

    block_9 = yolo_fastext_conv_block(48, 16)(block_8_add)
    block_10 = yolo_fastext_conv_block(96, 16)(block_9)
    block_10_drop = Dropout(0.15)(block_10)
    block_10_add = Add()([block_9, block_10_drop])
    block_11 = yolo_fastext_conv_block(96, 16)(block_10_add)
    block_11_drop = Dropout(0.15)(block_11)
    block_11_add = Add()([block_10_add, block_11_drop])
    block_12 = yolo_fastext_conv_block(96, 16)(block_11_add)
    block_12_drop = Dropout(0.15)(block_12)
    block_12_add = Add()([block_11_add, block_12_drop])
    block_13 = yolo_fastext_conv_block(96, 16)(block_12_add)
    block_13_drop = Dropout(0.15)(block_13)
    block_13_add = Add()([block_12_add, block_13_drop])

    block_14 = yolo_fastext_conv_block_withpad(96, 24)(block_13_add)
    block_15 = yolo_fastext_conv_block(136, 24)(block_14)
    block_15_drop = Dropout(0.15)(block_15)
    block_15_add = Add()([block_14, block_15_drop])
    block_16 = yolo_fastext_conv_block(136, 24)(block_15_add)
    block_16_drop = Dropout(0.15)(block_16)
    block_16_add = Add()([block_15_add, block_16_drop])
    block_17 = yolo_fastext_conv_block(136, 24)(block_16_add)
    block_17_drop = Dropout(0.15)(block_17)
    block_17_add = Add()([block_16_add, block_17_drop])
    block_18 = yolo_fastext_conv_block(136, 24)(block_17_add)
    block_18_drop = Dropout(0.15)(block_18)
    block_18_add = Add()([block_17_add, block_18_drop])

    block_19 = DarknetConv2D_BN_Leaky(136, (1, 1), padding='valid', strides=(1, 1))(block_18_add)
    # concat
    block_19_pad = ZeroPadding2D(((1, 0), (1, 0)))(block_19)
    block_19_dconv = Darknet_Depthwise_Conv2D_BN_Leaky((3, 3), padding='valid', strides=(2, 2))(block_19_pad)
    block_19_conv = DarknetConv2D(48, (1, 1), padding='valid', strides=(1, 1), use_bias=False)(block_19_dconv)
    block_19_bn = CustomBatchNormalization()(block_19_conv)

    block_20 = yolo_fastext_conv_block(224, 48)(block_19_bn)
    block_20_drop = Dropout(0.15)(block_20)
    block_20_add = Add()([block_19_bn, block_20_drop])
    block_21 = yolo_fastext_conv_block(224, 48)(block_20_add)
    block_21_drop = Dropout(0.15)(block_21)
    block_21_add = Add()([block_20_add, block_21_drop])
    block_22 = yolo_fastext_conv_block(224, 48)(block_21_add)
    block_22_drop = Dropout(0.15)(block_22)
    block_22_add = Add()([block_21_add, block_22_drop])
    block_23 = yolo_fastext_conv_block(224, 48)(block_22_add)
    block_23_drop = Dropout(0.15)(block_23)
    block_23_add = Add()([block_22_add, block_23_drop])
    block_24 = yolo_fastext_conv_block(224, 48)(block_23_add)
    block_24_drop = Dropout(0.15)(block_24)
    block_24_add = Add()([block_23_add, block_24_drop])
    block_24_final = DarknetConv2D_BN_Leaky(96, (1, 1), padding='valid', strides=(1, 1))(block_24_add)

    branch_1 = compose(
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(128, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(128, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1),
                      padding='same', strides=(1, 1), name='predict_conv_1')
    )(block_24_final)

    
    if cfg.YOLO.fixshape:
        # method = tf.image.ResizeMethod.BILINEAR
        # spp_upsample = tf.image.resize(spp_final, [20,20], method)
        block_24_final_upsample = Resizing(int(cfg.YOLO.input_shape[0]/32 *2), int(cfg.YOLO.input_shape[0]/32 *2), interpolation="bilinear", crop_to_aspect_ratio=False)(block_24_final)
    else:
        block_24_final_upsample = UpSampling2D((2, 2), interpolation="bilinear")(block_24_final)
        
    branch2_concate = Concatenate()([block_24_final_upsample, block_19])

    branch_2 = compose(
        DarknetConv2D_BN_Leaky(96, (1, 1), padding='same', strides=(1, 1)),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(96, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        Darknet_Depthwise_Conv2D_BN_Leaky(
            (5, 5), padding='same', strides=(1, 1)),
        DarknetConv2D(96, (1, 1), padding='same',
                      strides=(1, 1), use_bias=False),
        CustomBatchNormalization(),
        DarknetConv2D(num_anchors*(num_classes+5), (1, 1),
                      padding='same', strides=(1, 1), name='predict_conv_2')
    )(branch2_concate)

    return Model(inputs, [branch_1, branch_2])