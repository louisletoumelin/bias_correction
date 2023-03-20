import numpy as np
import pandas as pd
import tensorflow as tf


def bias(y_true, y_pred):
    """Bias"""
    return y_pred - y_true


def n_bias(y_true, y_pred, epsilon=0.01):
    """Normalized bias"""
    return (y_pred - y_true)/(y_true+epsilon)


def ae(y_true, y_pred):
    """Absolute error"""
    return np.abs(y_pred - y_true)


def n_ae(y_true, y_pred, epsilon=0.01):
    """Normalized absolute error"""
    return np.abs(n_bias(y_true, y_pred, epsilon=epsilon))


def mbe(y_true, y_pred):
    """Mean bias"""
    return np.nanmean(bias(y_true, y_pred))


def m_n_be(y_true, y_pred, epsilon=0.01):
    """Normalized mean bias"""
    return np.nanmean(n_bias(y_true, y_pred, epsilon=epsilon))


def m_n_ae(y_true, y_pred, epsilon=0.01):
    """Mean normalized absolute error"""
    return np.nanmean(n_ae(y_true, y_pred, epsilon=epsilon))


def corr(y_true, y_pred):
    """Pearson correlation coefficient"""
    df = pd.DataFrame(np.transpose([y_true, y_pred]), columns=["y_true", "y_pred"])
    return df.corr().iloc[0, 1]


def rmse(y_true, y_pred):
    """Root mean squared error"""
    return np.sqrt(np.nanmean(bias(y_true, y_pred)**2))


def mae(y_true, y_pred):
    """Mean absolute error"""
    return np.nanmean(ae(y_true, y_pred))


def tf_mbe(y_true, y_pred):
    """Mean biad error written in Tensorflow"""
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops

    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return tf.keras.backend.mean(y_pred - y_true, axis=-1)


def bias_direction(y_true, y_pred):
    """Bias for wind direction"""
    # Wind direction = 0 means wind direction is not specified
    pred = np.where(y_pred != 0, y_pred, np.nan)
    true = np.where(y_true != 0, y_true, np.nan)

    diff1 = np.mod((pred - true), 360)
    diff2 = np.mod((true - pred), 360)

    return np.where(diff1 <= diff2, diff1, -diff2)


def abs_bias_direction(y_true, y_pred):
    """Absolute bias for wind direction"""
    return np.abs(bias_direction(y_true, y_pred))


def mean_abs_bias_direction(y_true, y_pred):
    """Absolute bias for wind direction"""
    return np.nanmean(np.abs(bias_direction(y_true, y_pred)))


dict_metrics = {"bias": bias,
                "n_bias": n_bias,
                "ae": ae,
                "n_ae": n_ae,
                "mbe": mbe,
                "m_n_be": m_n_be,
                "m_n_ae": m_n_ae,
                "corr": corr,
                "rmse": rmse,
                "mae": mae,
                "tf_rmse": tf.keras.metrics.RootMeanSquaredError(),
                "tf_mae": tf.keras.metrics.MeanAbsoluteError(),
                "tf_mbe": tf_mbe,
                "bias_direction": bias_direction,
                "abs_bias_direction": abs_bias_direction,
                "mean_abs_bias_direction": mean_abs_bias_direction
                }


def get_metric(metric_name):
    return dict_metrics[metric_name]
