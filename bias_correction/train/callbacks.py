import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint

try:
    import horovod.tensorflow as hvd

    _horovod = True
except ModuleNotFoundError:
    _horovod = False

import os
from typing import List

from bias_correction.train.eval import Interpretability


class FeatureImportanceCallback(tf.keras.callbacks.Callback):

    def __init__(self, data_loader, cm, exp, mode):
        self.it = Interpretability(data_loader, cm, exp)
        self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        df_rmse, df_ae, _, _ = self.it.compute_feature_importance(self.mode)
        name_figure = str(int(epoch))
        save_path = os.path.join(self.it.exp.path_to_feature_importance, name_figure, "figure_tmp.tmp")
        subfolder_in_filename = self.it.check_if_subfolder_in_filename(save_path)
        if subfolder_in_filename:
            new_path, _ = self.it.create_subfolder_if_necessary(name_figure, save_path)
        df_rmse.to_csv(os.path.join(save_path, "df_rmse.csv"))
        df_ae.to_csv(os.path.join(save_path, "df_ae.csv"))
        print("Feature importance computed and saved")


callbacks_dict = {"TensorBoard": TensorBoard,
                  "ReduceLROnPlateau": ReduceLROnPlateau,
                  "EarlyStopping": EarlyStopping,
                  "CSVLogger": CSVLogger,
                  "ModelCheckpoint": ModelCheckpoint,
                  "FeatureImportanceCallback": FeatureImportanceCallback,
                  # Horovod: broadcast initial variable states from rank 0 to all other processes.
                  # This is necessary to ensure consistent initialization of all workers when
                  # training is started with random weights or restored from a checkpoint.
                  "BroadcastGlobalVariablesCallback": hvd.keras.callbacks.BroadcastGlobalVariablesCallback,
                  # Horovod: average metrics among workers at the end of every epoch.
                  # Note: This callback must be in the list before the ReduceLROnPlateau,
                  # TensorBoard or other metrics-based callbacks.
                  "MetricAverageCallback": hvd.keras.callbacks.MetricAverageCallback,
                  # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
                  # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
                  # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
                  "LearningRateWarmupCallback": hvd.keras.callbacks.LearningRateWarmupCallback
                  }


def load_callbacks(callbacks_str: List[str], config: dict) -> list:
    callbacks = []
    for callback_str in callbacks_str:
        args = config["args_callbacks"][callback_str]
        kwargs = config["kwargs_callbacks"][callback_str]
        callback = callbacks_dict[callback_str](*args, **kwargs)
        callbacks.append(callback)
    return callbacks


def get_callbacks(callbacks_str: str, distribution_strategy: str, config: dict):
    if distribution_strategy == "Horovod" and _horovod:
        callbacks = load_callbacks(["BroadcastGlobalVariablesCallback", "MetricAverageCallback"], config)
        if hvd.rank() == 0:
            callbacks += load_callbacks(callbacks_str, config)
    else:
        callbacks = load_callbacks(callbacks_str, config)

    return callbacks


def load_callback_with_custom_model(cm, data_loader=None, mode_callback=None):

    args_callbacks = {"CSVLogger": [cm.path_to_logs + "tf_logs.csv"],
                      "BroadcastGlobalVariablesCallback": [0],
                      "FeatureImportanceCallback": [data_loader, cm, cm.exp, mode_callback],
                      "MetricAverageCallback": [],
                      "LearningRateWarmupCallback": []
                      }

    kwargs_callbacks = {"TensorBoard": {"log_dir": cm.path_to_tensorboard_logs},
                        "ModelCheckpoint": {"filepath": cm.path_to_best_model}
                        }
    # Create args dict
    cm.config["args_callback"] = args_callbacks

    # Update old kwargs dict with new kwargs
    for callback in cm.config["kwargs_callbacks"]:
        try:
            cm.config["kwargs_callbacks"][callback] = cm.config["kwargs_callbacks"][callback] | kwargs_callbacks[
                callback]
        except KeyError:
            pass

    return get_callbacks(cm.config["callbacks"], cm.config["distribution_strategy"], cm.config)
