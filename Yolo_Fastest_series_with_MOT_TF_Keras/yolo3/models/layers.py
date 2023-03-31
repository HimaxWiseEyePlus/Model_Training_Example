#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common layer definition for YOLOv3 models building
"""
from functools import wraps, reduce

import tensorflow.keras.backend as K
from tensorflow.keras.layers import LeakyReLU, ReLU

from common.backbones.layers import YoloConv2D, YoloDepthwiseConv2D, CustomBatchNormalization
from core.config import cfg

ACT_TYPE = cfg.YOLO.ACT_TYPE

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(YoloConv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get(
        'strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloConv2D(*args, **darknet_conv_kwargs)


@wraps(YoloDepthwiseConv2D)
def DarknetDepthwiseConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for YoloDepthwiseConv2D."""
    #darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    #darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs = {'padding': 'valid' if kwargs.get(
        'strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return YoloDepthwiseConv2D(*args, **darknet_conv_kwargs)


def Darknet_Depthwise_Conv2D_BN_Leaky(kernel_size=(3, 3), block_id_str=None, **kwargs):
    """Depthwise  Convolution2D."""
    if not block_id_str:
        block_id_str = str(K.get_uid())
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetDepthwiseConv2D(
            kernel_size, name='conv_dw_' + block_id_str, **no_bias_kwargs),
        CustomBatchNormalization(name='conv_dw_%s_bn' % block_id_str),
        LeakyReLU(alpha=0.1, name='conv_dw_%s_leaky_relu' % block_id_str) if ACT_TYPE == 'leaky' else ReLU(name='conv_dw_%s_relu' % block_id_str) if ACT_TYPE == 'relu' else ReLU(max_value = 6., name='conv_dw_%s_relu6' % block_id_str))


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by CustomBatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        CustomBatchNormalization(),
        LeakyReLU(alpha=0.1) if ACT_TYPE == 'leaky' else ReLU() if ACT_TYPE == 'relu' else ReLU(max_value = 6.)
    )
