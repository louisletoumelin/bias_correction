import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, \
    ReduceLROnPlateau,\
    EarlyStopping,\
    CSVLogger,\
    ModelCheckpoint,\
    LearningRateScheduler

try:
    import horovod.tensorflow as hvd
    _horovod = True
except ModuleNotFoundError:
    _horovod = False

import os
from typing import List
from copy import deepcopy

from bias_correction.train.eval import Interpretability
from bias_correction.train.utils import no_raise_on_key_error

initial_learning_rate = 0.01
epochs = 100
decay = initial_learning_rate / epochs


def learning_rate_time_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


class FeatureImportanceCallback(tf.keras.callbacks.Callback):

    def __init__(self, data_loader, cm, exp, mode):
        super().__init__()
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
                  "CSVLogger_dir": CSVLogger,
                  "ModelCheckpoint": ModelCheckpoint,
                  "FeatureImportanceCallback": FeatureImportanceCallback,
                  "learning_rate_decay": LearningRateScheduler(learning_rate_time_decay, verbose=1)
                  }

if _horovod:
    try:
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        callbacks_dict["BroadcastGlobalVariablesCallback"] = hvd.keras.callbacks.BroadcastGlobalVariablesCallback
        # Horovod: average metrics among workers at the end of every epoch.
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        callbacks_dict["MetricAverageCallback"] = hvd.keras.callbacks.MetricAverageCallback
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        callbacks_dict["LearningRateWarmupCallback"] = hvd.keras.callbacks.LearningRateWarmupCallback
    except AttributeError as e:
        print(e)


def load_callbacks(callbacks_str: List[str], args_callbacks: dict, kwargs_callbacks: dict) -> list:
    callbacks = []

    for callback_str in callbacks_str:
        args = args_callbacks[callback_str]
        kwargs = kwargs_callbacks[callback_str]
        callback = callbacks_dict[callback_str](*args, **kwargs)
        callbacks.append(callback)

    return callbacks


def get_callbacks(callbacks_str: List[str], distribution_strategy: str, args_callbacks: dict, kwargs_callbacks: dict):
    normal_callbacks = load_callbacks(callbacks_str, args_callbacks, kwargs_callbacks)

    if distribution_strategy == "Horovod" and _horovod:
        callbacks_str = ["BroadcastGlobalVariablesCallback", "MetricAverageCallback"]
        hvdcallbacks = load_callbacks(callbacks_str, args_callbacks, kwargs_callbacks)
        if hvd.rank() == 0:
            return hvdcallbacks + normal_callbacks
    else:
        return normal_callbacks


def load_callback_with_custom_model(cm, data_loader=None, mode_callback=None):
    _tmp_args_callbacks = {"CSVLogger": [cm.exp.path_to_logs + "tf_logs.csv"],
                           "CSVLogger_dir": [cm.exp.path_to_logs_dir + "tf_logs.csv"],
                           "BroadcastGlobalVariablesCallback": [0],
                           "FeatureImportanceCallback": [data_loader, cm, cm.exp, mode_callback],
                           "MetricAverageCallback": [],
                           "LearningRateWarmupCallback": [],
                           "learning_rate_decay": []
                           }

    _tmp_kwargs_callbacks = {"TensorBoard": {"log_dir": cm.exp.path_to_tensorboard_logs},
                             "ModelCheckpoint": {"filepath": cm.exp.path_to_best_model}
                             }

    args = deepcopy(cm.config["args_callbacks"])
    kwargs = deepcopy(cm.config["kwargs_callbacks"])

    # Update old kwargs dict with new kwargs
    for callback in cm.config["kwargs_callbacks"]:
        with no_raise_on_key_error():
            d = cm.config["kwargs_callbacks"][callback].copy()
            d.update(_tmp_kwargs_callbacks[callback])
            kwargs[callback] = d

        with no_raise_on_key_error():
            args[callback].extend(_tmp_args_callbacks[callback])

    return get_callbacks(cm.config["callbacks"], cm.config["distribution_strategy"], args, kwargs)
