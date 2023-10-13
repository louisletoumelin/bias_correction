import numpy as np
import pandas as pd

from typing import List, Tuple, Union


from bias_correction.train.metrics import get_metric
from bias_correction.train.dataloader import CustomDataHandler


def classify_topo_carac(stations: pd.DataFrame,
                        df: pd.DataFrame,
                        topo_carac: list = ['mu', 'curvature', 'tpi_500', 'tpi_2000', 'laplacian', 'alti'],
                        config: Union[dict, None] = None
                        ) -> pd.DataFrame:
    if config is not None:
        stations = stations[~stations["country"].isin(config["country_to_reject_during_training"]) & ~stations["name"].isin(
            config["stations_to_reject"])]
        df = df[df["name"].isin(stations["name"].unique())]

    print("\ndebug")
    print(df["name"].unique())

    for carac in topo_carac:

        if not hasattr(df, f"class_{carac}"):
            df[f"class_{carac}"] = np.nan

        carac_nn = carac + '_NN_0' if carac not in ["alti", "country"] else carac

        q25 = np.quantile(stations[carac_nn].values, 0.25)
        q50 = np.quantile(stations[carac_nn].values, 0.5)
        q75 = np.quantile(stations[carac_nn].values, 0.75)

        filter_1 = (df[carac_nn] <= q25)
        filter_2 = (df[carac_nn] > q25) & (df[carac_nn] <= q50)
        filter_3 = (df[carac_nn] > q50) & (df[carac_nn] <= q75)
        filter_4 = (df[carac_nn] > q75)

        df.loc[filter_1, f"class_{carac}"] = "$x \leq q_{25}$"
        df.loc[filter_2, f"class_{carac}"] = "$q_{25}<x \leq q_{50}$"
        df.loc[filter_3, f"class_{carac}"] = "$q_{50}<x \leq q_{75}$"
        df.loc[filter_4, f"class_{carac}"] = "$q_{75}<x$"

        print(f"Quantiles {carac}: ", q25, q50, q75)
    return df


def classify_alti(df: pd.DataFrame
                  ) -> pd.DataFrame:

    df.loc[:, "class_alti0"] = np.nan

    filter_1 = (df["alti"] <= 500)
    filter_2 = (500 < df["alti"]) & (df["alti"] <= 1000)
    filter_3 = (1000 < df["alti"]) & (df["alti"] <= 2000)
    filter_4 = (2000 < df["alti"])

    df.loc[filter_1, "class_alti0"] = "$Elevation [m] \leq 500$"
    df.loc[filter_2, "class_alti0"] = "$500<Elevation [m] \leq 1000$"
    df.loc[filter_3, "class_alti0"] = "$1000<Elevation [m] \leq 2000$"
    df.loc[filter_4, "class_alti0"] = "$2000<Elevation [m]$"

    for filter_alti in [filter_1, filter_2, filter_3, filter_4]:
        print(len(df.loc[filter_alti, "name"].unique()))

    return df


def classify_forecast_term(df: pd.DataFrame
                           ) -> pd.DataFrame:

    def compute_lead_time(hour):
        return (hour - 6) % 24 + 6

    df.loc[:, "lead_time"] = compute_lead_time(df.index.hour.values)

    return df


def add_metric_to_df(df: pd.DataFrame,
                     keys: List[str],
                     key_obs: str,
                     metrics: Tuple[str, ...] = ("bias", "n_bias", "ae", "n_ae")
                     ) -> pd.DataFrame:
    for metric in metrics:
        metric_func = get_metric(metric)
        for key in keys:
            result = metric_func(df[key_obs].values, df[key].values)
            key = '_' + key.split('_')[-1]
            df[f"{metric}{key}"] = result
    return df


def add_topo_carac_from_stations_to_df(df, stations,
                                       topo_carac=('mu',
                                                   'curvature',
                                                   'tpi_500',
                                                   'tpi_2000',
                                                   'laplacian',
                                                   'alti',
                                                   'country')):
    for station in df["name"].unique():
        filter_df = df["name"] == station
        filter_s = stations["name"] == station
        for carac in topo_carac:
            carac = carac + '_NN_0' if carac not in ["alti", "country"] else carac
            if hasattr(df, carac):
                df.loc[filter_df, carac] = stations.loc[filter_s, carac].values[0]
            else:
                df[carac] = np.nan
                df.loc[filter_df, carac] = stations.loc[filter_s, carac].values[0]
    return df


def add_other_models(df: pd.DataFrame,
                     models: List[str],
                     current_variable: str,
                     data_loader: CustomDataHandler):

    for model_str in models:

        assert hasattr(data_loader, f"predicted{model_str}")
        results = []

        df.loc[:, current_variable + model_str] = np.nan

        for station in df["name"].unique():
            filter_df_results = df["name"] == station
            df_station = df.loc[filter_df_results, :]

            model = getattr(data_loader, f"predicted{model_str}")
            filter_df_model = model["name"] == station
            model_station = model.loc[filter_df_model, :]

            filter_time = df_station.index.intersection(model_station.index)
            df_station.loc[filter_time, current_variable + model_str] = model_station.loc[
                filter_time, current_variable + model_str]
            results.append(df_station)

        df = pd.concat(results)

    return df

