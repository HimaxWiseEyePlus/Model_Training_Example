#!/usr/bin/python3
# -*- coding=utf-8 -*-
"""custom model callbacks."""
import os
import sys

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow.keras.callbacks import Callback

sys.path.append(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..'))


class MOTSaveCallBack(Callback):
    def __init__(self, save_epoch_interval, model_pruning, model_clustering , log_dir):

        self.save_epoch_interval = save_epoch_interval
        self.model_pruning = model_pruning
        self.model_clustering = model_clustering
        self.log_dir = log_dir

    def update_eval_model(self, train_model):
        try:
            if self.model_pruning:
                eval_model = sparsity.strip_pruning(train_model)
            if self.model_clustering:
                eval_model = tfmot.clustering.keras.strip_clustering(train_model)
        except ValueError:
            print("ValueError strip_pruning fail")
            return train_model
        return eval_model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.save_epoch_interval == 0:
            # Do eval every eval_epoch_interval epochs
            eval_model = self.update_eval_model(self.model)
            eval_model.save_weights(os.path.join(self.log_dir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-prun.h5'.format(
                epoch=(epoch+1), loss=logs.get('loss'), val_loss=logs.get('val_loss'))))
