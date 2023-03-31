#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""Model utility functions."""
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow_addons as tfa
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay, PiecewiseConstantDecay
from tensorflow.keras.experimental import CosineDecay

import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
import tensorflow_model_optimization as tfmot

from typing import Callable, Union
from core.config import cfg


def add_metrics(model, metric_dict):
    '''
    add metric scalar tensor into model, which could be tracked in training
    log and tensorboard callback
    '''
    for (name, metric) in metric_dict.items():
        # seems add_metric() is newly added in tf.keras. So if you
        # want to customize metrics on raw keras model, just use
        # "metrics_names" and "metrics_tensors" as follow:
        #
        # model.metrics_names.append(name)
        # mod   el.metrics_tensors.append(loss)
        model.add_metric(metric, name=name, aggregation='mean')

# from tensorflow_model_optimization.quantization.keras import QuantizeConfig

# class NoOpQuantizeConfig(QuantizeConfig):
#     """QuantizeConfig which does not quantize any part of the layer."""

#     def get_weights_and_quantizers(self, layer):
#         return []

#     def get_activations_and_quantizers(self, layer):
#         return []

#     def set_quantize_weights(self, layer, quantize_weights):
#         pass

#     def set_quantize_activations(self, layer, quantize_activations):
#         pass

#     def get_output_quantizers(self, layer):
#         return []

#     def get_config(self):
#         return {}

def get_qat_model(model):

    quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(model)

    with tfmot.quantization.keras.quantize_scope():
        if cfg.MOT.qat_mode == 'pqat':
            qat_model = tfmot.quantization.keras.quantize_apply(
                quant_aware_annotate_model,
                tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme()
            )
        elif cfg.MOT.qat_mode == 'pcqat':
            qat_model = tfmot.quantization.keras.quantize_apply(
                quant_aware_annotate_model,
                tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True)
            )
        elif cfg.MOT.qat_mode == 'cqat':
            qat_model = tfmot.quantization.keras.quantize_apply(
              quant_aware_annotate_model,
              tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme())
        else:
            qat_model = tfmot.quantization.keras.quantize_apply(
                quant_aware_annotate_model,
                tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme()
            )

    return qat_model


def get_clustering_model(model):
    cluster_weights = tfmot.clustering.keras.cluster_weights
    
    if cfg.MOT.cluster_centroids_init == 'KMEANS_PLUS_PLUS':
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS
    elif cfg.MOT.cluster_centroids_init == 'DENSITY_BASED':
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization.DENSITY_BASED
    else:
        CentroidInitialization = tfmot.clustering.keras.CentroidInitialization.LINEAR

    clustering_params = {
        'number_of_clusters': cfg.MOT.number_of_clusters,
        'cluster_centroids_init': CentroidInitialization,
        'preserve_sparsity': cfg.MOT.preserve_sparsity
    }

    from tensorflow_model_optimization.python.core.clustering.keras import clustering_registry
    ClusteringRegistry = clustering_registry.ClusteringRegistry

    def apply_clustering(layer):
        if ClusteringRegistry.supports(layer):
            return cluster_weights(layer, **clustering_params)
        return layer

    clustered_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_clustering,
    )

    return clustered_model


def get_pruning_model(model, begin_step, end_step):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                     final_sparsity=cfg.MOT.final_sparsity,
                                                     begin_step=begin_step,
                                                     end_step=end_step,
                                                     frequency=cfg.MOT.frequency)
    }

    from tensorflow_model_optimization.python.core.sparsity.keras import prune_registry
    PruneRegistry = prune_registry.PruneRegistry

    def apply_pruning(layer):
        if PruneRegistry.supports(layer):
            return sparsity.prune_low_magnitude(layer, **pruning_params)
        return layer

    pruning_model = tf.keras.models.clone_model(
        model,
        clone_function=apply_pruning,
    )

    return pruning_model


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        initial_steps: int,
        power: float = 2.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name
        self.initial_steps = initial_steps

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step+self.initial_steps, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * \
                tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step+self.initial_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "initial_steps": self.initial_steps,
            "power": self.power,
            "name": self.name,
        }


class WD(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        weight_decay: float,
        decay_schedule_fn: Union[float, Callable],
        learning_rate: float,
        name: str = None,
    ):
        super().__init__()
        self.weight_decay = weight_decay
        self.decay_schedule_fn = decay_schedule_fn
        self.learning_rate = learning_rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WD") as name:
            # /self.learning_rate
            if isinstance(self.decay_schedule_fn, Callable):
                return self.weight_decay * self.decay_schedule_fn(step)
            else:
                return self.weight_decay * self.decay_schedule_fn

    def get_config(self):
        return {
            "weight_decay": self.weight_decay,
            "decay_schedule_fn": self.decay_schedule_fn,
            "name": self.name,
        }


def get_lr_scheduler(learning_rate, decay_type, decay_steps, total_epoch, initial_steps):
    if decay_type:
        decay_type = decay_type.lower()
    print(decay_type)
    if decay_type == None or decay_type == 'none':
        lr_scheduler = learning_rate
    elif decay_type == 'cosine':
        lr_scheduler = CosineDecay(
            initial_learning_rate=learning_rate, decay_steps=decay_steps)
    elif decay_type == 'exponential':
        lr_scheduler = ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=0.9)
    elif decay_type == 'polynomial':
        lr_scheduler = PolynomialDecay(initial_learning_rate=learning_rate,
                                       decay_steps=decay_steps*0.8, end_learning_rate=learning_rate/100)
    elif decay_type == 'piecewise_constant':
        # apply a piecewise constant lr scheduler, including warmup stage
        boundaries = [int(decay_steps*0.8), int(decay_steps*0.9)]
        values = [learning_rate, learning_rate/10., learning_rate/100.]
        piecewise_lr_scheduler = PiecewiseConstantDecay(
            boundaries=boundaries, values=values)
        lr_scheduler = WarmUp(
            learning_rate, piecewise_lr_scheduler, int(decay_steps*cfg.YOLO.warmup_epochs/total_epoch), initial_steps)

    else:
        raise ValueError('Unsupported lr decay type')

    return lr_scheduler


def get_optimizer(optim_type, learning_rate, total_epoch, decay_type='cosine', decay_steps=100000, initial_steps = 0):
    optim_type = optim_type.lower()

    print(optim_type)
    lr_scheduler = get_lr_scheduler(
        learning_rate, decay_type, decay_steps, total_epoch, initial_steps)

    if optim_type == 'adam':
        optimizer = Adam(learning_rate=lr_scheduler)
    elif optim_type == 'rmsprop':
        optimizer = RMSprop(learning_rate=lr_scheduler,
                            rho=0.9, momentum=0.0, centered=False)
    elif optim_type == 'sgd':
        optimizer = SGD(learning_rate=lr_scheduler,
                        momentum=0.949, nesterov=False)
    elif optim_type == 'sgd0':
        optimizer = SGD(learning_rate=lr_scheduler,
                        momentum=0.0, nesterov=False)
    elif optim_type == 'sgdw':
        wd = WD(0.0005, lr_scheduler, learning_rate)
        optimizer = tfa.optimizers.SGDW(
            learning_rate=lr_scheduler, weight_decay=wd, momentum=0.949, nesterov=False)

    else:
        raise ValueError('Unsupported optimizer type')

    return optimizer
