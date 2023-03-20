import numpy as np
import pandas as pd

def bias(y_true, y_pred):
    """Bias"""
    return y_pred - y_true


def n_bias(y_true, y_pred):
    """Normalized bias"""
    return (y_pred - y_true)/y_true


def ae(y_true, y_pred):
    """Absolute error"""
    return np.abs(y_pred - y_true)


def n_ae(y_true, y_pred):
    """Normalized absolute error"""
    return np.abs(n_bias(y_true, y_pred))


def mbe(y_true, y_pred):
    """Mean bias"""
    return np.nanmean(bias(y_true, y_pred))


def m_n_be(y_true, y_pred):
    """Normalized mean bias"""
    return np.nanmean(n_bias(y_true, y_pred))


def m_n_ae(y_true, y_pred):
    """Mean normalized absolute error"""
    return np.nanmean(n_ae(y_true, y_pred))


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


dict_metrics = {"bias": bias,
                "n_bias": n_bias,
                "ae": ae,
                "n_ae": n_ae,
                "mbe": mbe,
                "m_n_be": m_n_be,
                "m_n_ae": m_n_ae,
                "corr": corr,
                "rmse": rmse,
                "mae": mae}


def get_metric(metric_name):
    return dict_metrics[metric_name]
