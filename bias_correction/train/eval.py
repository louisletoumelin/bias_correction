import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from typing import Union, List

try:
    import seaborn as sns

    sns_ = True
except (ImportError, ModuleNotFoundError):
    sns_ = False

from bias_correction.train.visu import VizualizationResults
from bias_correction.train.metrics import get_metric
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.train.experience_manager import ExperienceManager
import bias_correction.train.dataframe_computer as computer


class CustomEvaluation(VizualizationResults):

    def __init__(self,
                 exp: ExperienceManager,
                 data: CustomDataHandler,
                 mode: str = "test",
                 keys: List[str] = ["_AROME", "_nn"],
                 stations_to_remove: Union[List[str], List] = [],
                 other_models: Union[List[str], List] = [],
                 quick: bool = False):

        super().__init__(exp)

        self.exp = exp
        self.data = data
        self.current_variable = self.exp.get_config()["current_variable"]
        self.mode = mode

        if self.exp.config["get_intermediate_output"]:
            keys.append("_int")
        self._set_key_attributes(keys)
        self._set_key_list(keys)
        self.df_results = self.create_df_results()

        if other_models:
            self.df_results = computer.add_other_models(self.df_results,
                                                        other_models,
                                                        self.current_variable,
                                                        self.data)
            keys = keys + other_models
            self._set_key_attributes(keys)
            self._set_key_list(keys)

        if stations_to_remove:
            self.df_results = self.df_results[~self.df_results["name"].isin(stations_to_remove)]

        if not quick:
            self.df_results = computer.add_metric_to_df(self.df_results,
                                                        self.keys,
                                                        self.key_obs,
                                                        metrics=["bias", "n_bias", "ae", "n_ae"])
            self.df_results = computer.add_topo_carac_from_stations_to_df(self.df_results,
                                                                          self.data.get_stations(),
                                                                          topo_carac=['mu', 'curvature', 'tpi_500',
                                                                                      'tpi_2000', 'laplacian',
                                                                                      'alti',
                                                                                      'country'])
            self.df_results = computer.classify_topo_carac(self.data.get_stations(), self.df_results)
            self.df_results = computer.classify_alti(self.df_results)
            self.df_results = computer.classify_forecast_term(self.df_results)

    def _set_key_attributes(self,
                            keys: List[str]
                            ) -> None:
        self.key_obs = f"{self.current_variable}_obs"
        for key in keys:
            setattr(self, f"key{key}", f"{self.current_variable}{key}")

    def _set_key_list(self,
                      keys: List[str]
                      ) -> None:
        self.keys = [f"{self.current_variable}{key}" for key in keys]

    def create_df_results(self):

        mode = self.mode
        assert hasattr(self.data, f"predicted_{mode}")
        df = self.data.get_predictions(mode)

        if self.current_variable == "UV":
            return self.create_df_speed(df)
        else:
            return self.create_df_temp(df)

    def create_df_temp(self,
                       df: pd.DataFrame
                       ) -> pd.DataFrame:
        """
        Select variables for temperature predictions

        :param df: DataFrame with data
        :type df: pandas.DataFrame
        :return: DataFrame with temperature variables
        """
        labels = self.data.get_labels(self.mode)
        inputs = self.data.get_inputs(self.mode)
        df.loc[:, "T2m_obs"] = labels.values
        df.loc[:, "T2m_AROME"] = inputs["Tair"].values
        columns = ["name"] + self.keys
        return df[columns]

    def create_df_speed(self,
                        df: pd.DataFrame
                        ) -> pd.DataFrame:

        labels = self.data.get_labels(self.mode)
        inputs = self.data.get_inputs(self.mode)

        if "component" in self.data.config["type_of_output"]:

            df.loc[:, "UV_obs"] = np.sqrt(labels["U_obs"] ** 2 + labels["V_obs"] ** 2)
            df.loc[:, "UV_AROME"] = inputs["Wind"].values

        elif "speed" in self.data.config["type_of_output"]:

            df.loc[:, "UV_obs"] = labels.values
            df.loc[:, "UV_AROME"] = inputs["Wind"].values

        columns = ["name", f"{self.current_variable}_obs"] + self.keys
        return df[columns]

    @staticmethod
    def _df2metric(df: pd.DataFrame,
                   metric_name: str,
                   key_obs: str,
                   keys: List[str],
                   print_: bool = False
                   ) -> list:
        metric_func = get_metric(metric_name)
        results = []
        for key in keys:
            metric = metric_func(df[key_obs].values, df[key].values)
            if print_:
                print(f"\n{metric_name}{key}")
                print(metric)
            results.append(metric)
        return results

    def df2metric(self,
                  metric_name: str,
                  print_: bool = False
                  ) -> list:
        return self._df2metric(self.df_results,
                               metric_name,
                               self.key_obs,
                               self.keys,
                               print_=print_)

    def df2mae(self,
               print_: bool = False
               ) -> list:
        return self.df2metric("mae", print_=print_)

    def df2rmse(self,
                print_: bool = False
                ) -> list:
        return self.df2metric("rmse", print_=print_)

    def df2mbe(self,
               print_: bool = False
               ) -> list:
        return self.df2metric("mbe", print_=print_)

    def df2m_n_be(self,
                  min_obs: float,
                  min_model: float,
                  print_: bool = False
                  ) -> None:
        for key in self.keys:
            filter_obs = self.df_results[self.key_obs] >= min_obs
            filter_model = self.df_results[key] >= min_model
            df = self.df_results[filter_obs & filter_model]
            self._df2metric(df, "m_n_be", self.key_obs, self.keys, print_=print_)

    def df2m_n_ae(self,
                  min_obs: float,
                  min_model: float,
                  print_: bool = False
                  ) -> None:
        for key in self.keys:
            key_model = f"{self.current_variable}{key}"
            filter_obs = self.df_results[self.key_obs] >= min_obs
            filter_model = self.df_results[key_model] >= min_model
            df = self.df_results[filter_obs & filter_model]
            self._df2metric(df, "m_n_ae", self.key_obs, self.keys, print_=print_)

    def df2ae(self,
              print_: bool = False
              ) -> list:
        return self._df2metric(self.df_results, "ae", self.key_obs, self.keys, print_=print_)

    def df2bias(self,
                print_: bool = False
                ) -> list:
        return self._df2metric(self.df_results, "bias", self.key_obs, self.keys, print_=print_)

    def df2correlation(self,
                       print_: bool = False
                       ) -> list:
        for station in self.df_results["name"].unique():
            filter_station = self.df_results["name"] == station
            self.df_results.loc[filter_station, :] = self.df_results.loc[filter_station, :].sort_index()

        return self.df2metric("corr", print_=print_)

    def print_stats(self
                    ) -> None:
        self.df2mae(print_=True)
        self.df2rmse(print_=True)
        self.df2mbe(print_=True)
        self.df2correlation(print_=True)


class StaticEval(VizualizationResults):

    def __init__(self, exp=None):
        super().__init__(exp)

    @staticmethod
    def print_train_test_val_stats_above_elevation(stations, time_series):
        assert "mode" in time_series
        assert "mode" in stations

        time_series = time_series[["name", "mode", "alti", "Wind", "vw10m(m/s)"]].dropna()

        for alti in np.linspace(0, 2500, 6):
            print(f"\n Alti >= {np.round(alti)}m")
            for mode in stations["mode"].unique():
                filter_mode = time_series["mode"] == mode
                filter_alti = time_series["alti"] >= alti

                nwp = time_series.loc[filter_mode & filter_alti, "Wind"].values
                obs = time_series.loc[filter_mode & filter_alti, "vw10m(m/s)"].values
                nb_stations = len(time_series.loc[filter_mode & filter_alti, "name"].unique())

                try:
                    results = []
                    for metric in ["mbe", "rmse", "corr", "mae"]:
                        metric_func = get_metric(metric)
                        result = metric_func(obs, nwp)
                        results.append(result)

                    print(f"Mode: {mode},  "
                          f"Nb stations, {nb_stations},  "
                          f"bias: {np.around(results[0], 2): .2f},  "
                          f"rmse: {np.around(results[1], 2): .2f},  "
                          f"corr: {np.around(results[2], 2): .2f},  "
                          f"mae: {np.around(results[3], 2): .2f}")
                except ValueError:
                    print(f"ValueError for {mode} and {alti}")

    @staticmethod
    def print_train_test_val_stats_by_elevation_category(stations, time_series,
                                                         list_min=[0, 1000, 2000, 3000],
                                                         list_max=[1000, 2000, 3000, 5000]):
        assert "mode" in time_series
        assert "mode" in stations

        time_series = time_series[["name", "mode", "alti", "Wind", "vw10m(m/s)"]].dropna()

        print("\n\nGeneral results")
        for metric in ["mbe", "rmse", "corr", "mae"]:
            for mode in stations["mode"].unique():
                filter_mode = time_series["mode"] == mode
                nwp = time_series.loc[filter_mode, "Wind"].values
                obs = time_series.loc[filter_mode, "vw10m(m/s)"].values
                metric_func = get_metric(metric)
                result = metric_func(obs, nwp)
                print(f"{mode}_{metric}: {result}")

        for alti_min, alti_max in zip(list_min, list_max):
            print(f"\n Alti category = {np.int(alti_min), np.int(alti_max)}m")
            for mode in stations["mode"].unique():
                filter_mode = time_series["mode"] == mode
                filter_alti = (time_series["alti"] >= alti_min) & (time_series["alti"] < alti_max)
                nwp = time_series.loc[filter_mode & filter_alti, "Wind"].values
                obs = time_series.loc[filter_mode & filter_alti, "vw10m(m/s)"].values
                nb_stations = len(time_series.loc[filter_mode & filter_alti, "name"].unique())

                try:
                    results = []
                    for metric in ["mbe", "rmse", "corr", "mae"]:
                        metric_func = get_metric(metric)
                        result = metric_func(obs, nwp)
                        results.append(result)

                    print(f"Mode: {mode},  "
                          f"Nb stations: {nb_stations},  "
                          f"bias: {np.round(results[0], 2): .2f},  "
                          f"rmse: {np.round(results[1], 2): .2f},  "
                          f"corr: {np.round(results[2], 2): .2f},  "
                          f"mae: {np.round(results[3], 2): .2f}")
                except ValueError:
                    print(f"ValueError for {mode} and [{alti_min}, {alti_max}]")


class Interpretability(VizualizationResults):

    def __init__(self, data, custom_model, exp=None):
        super().__init__(exp)
        self.data = data
        self.cm = custom_model

    def compute_feature_importance(self, mode, epsilon=0.01):
        list_results_rmse = []
        list_results_ae = []

        inputs = self.data.get_inputs(mode)
        length_batch = getattr(self.data, f"length_{mode}")
        inputs_tf = self.data.get_tf_zipped_inputs(inputs=inputs).batch(length_batch)

        with tf.device('/GPU:0'):
            results = self.cm.predict_with_batch(inputs_tf)

        # Calculate the new RMSE
        self.data.set_predictions(results, mode=mode)
        c_eval = CustomEvaluation(self.exp, self.data, mode=mode, keys=["_nn"], quick=True)
        se = np.array(c_eval.df2bias()[0]) ** 2
        ae = np.array(c_eval.df2ae()[0])

        for idx_pred, predictor in enumerate(inputs):
            print(predictor)

            # Create a copy of X_test
            inputs_copy = inputs.copy()

            # Scramble the values of the given predictor
            inputs_copy[predictor] = inputs[predictor].sample(frac=1).values

            # Prepare dataset
            length_batch = getattr(self.data, f"length_{mode}")
            inputs_tf = self.data.get_tf_zipped_inputs(inputs=inputs_copy).batch(length_batch)

            # Predict
            with tf.device('/GPU:0'):
                results_permutation = self.cm.predict_with_batch(inputs_tf)

            # Calculate the new metrics
            self.data.set_predictions(results_permutation, mode=mode)
            c_eval = CustomEvaluation(self.exp, self.data, mode=mode, keys=["_nn"], quick=True)
            se_permuted = np.array(c_eval.df2bias()[0]) ** 2  # Vector
            ae_permuted = np.array(c_eval.df2ae()[0])  # Vector

            # Metrics
            metric_se = 100 * (se_permuted - se) / (se + epsilon)
            metric_ae = 100 * (ae_permuted - ae) / (ae + epsilon)

            # Label names
            str_rmse = r'$\frac{RMSE_{permuted} - RMSE_{not \quad permuted}}{RMSE_{not \quad permuted}}$ [%]'
            str_ae = r'$\frac{AE_{permuted} - AE_{not \quad permuted}}{AE_{not \quad permuted}}$ [%]'

            list_results_rmse.append({'Predictor': predictor,
                                      str_rmse: np.sqrt(np.nanmean(metric_se)),
                                      "std": np.sqrt(np.nanstd(metric_se))})

            list_results_ae.append({'Predictor': predictor,
                                    str_ae: np.nanmean(metric_ae),
                                    "std": np.nanstd(ae)})

        # Put the results into a pandas dataframe and rank the predictors by score
        df_rmse = pd.DataFrame(list_results_rmse).sort_values(by=str_rmse, ascending=False)
        df_ae = pd.DataFrame(list_results_ae).sort_values(by=str_ae, ascending=False)

        return df_rmse, df_ae, str_rmse, str_ae

    def _plot_bar_with_error(self, df, x, y, err, name, width=0.01, figsize=(15, 12)):
        x = df[x].values
        y0 = df[y].values
        yerr = df[err].values
        plt.figure(figsize=figsize)
        plt.bar(x, y0, width=width, yerr=yerr)
        plt.ylabel(y)
        self.save_figure(name)

    def plot_feature_importance(self, mode, width=0.8, epsilon=0.01, figsize=(15, 12), name="Feature_importance"):

        df_rmse, df_ae, str_rmse, str_ae = self.compute_feature_importance(mode, epsilon=epsilon)

        # RMSE
        self._plot_bar_with_error(df_rmse,
                                  "Predictor",
                                  str_rmse,
                                  "std",
                                  f"Feature_importance/{name}_rmse",
                                  width=width,
                                  figsize=figsize)

        # AE
        self._plot_bar_with_error(df_ae,
                                  "Predictor",
                                  str_ae,
                                  "std",
                                  f"Feature_importance/{name}_ae",
                                  width=width,
                                  figsize=figsize)

    def plot_partial_dependence(self, mode, name="Partial_dependence_plot"):
        inputs = self.data.get_inputs(mode)
        c = cm.viridis(np.linspace(0, 1, len(inputs.keys())))
        for idx_pred, predictor in enumerate(inputs):
            print(predictor)
            mean_pred = np.nanmean(inputs[predictor])
            min_pred = np.nanmin(inputs[predictor])
            max_pred = np.nanmax(inputs[predictor])

            # min_pred = min_pred - 0.1 * np.abs(mean_pred)
            # max_pred = max_pred + 0.1 * np.abs(max_pred)
            # print(min_pred, max_pred)

            list_results = []
            inputs = self.data.get_inputs(mode)
            for fixed_value in np.linspace(min_pred, max_pred, 5, endpoint=True):
                print(fixed_value)
                # Create a copy of X_test
                inputs_copy = inputs.copy()

                # Replace predictor by fixed value
                inputs_copy[predictor] = fixed_value

                # Prepare dataset
                length_batch = getattr(self.data, f"length_{mode}")
                inputs_tf = self.data.get_tf_zipped_inputs(inputs=inputs_copy).batch(length_batch)

                # Predict
                with tf.device('/GPU:0'):
                    results_fixed_value = self.cm.predict_with_batch(inputs_tf)

                mean = np.nanmean(results_fixed_value)
                std = np.nanstd(results_fixed_value)
                list_results.append({'Fixed value': fixed_value, "mean": mean, "std": std})

            df = pd.DataFrame(list_results)
            plt.figure()
            plt.plot(df["Fixed value"], df["mean"], label='mean_1', color=c[idx_pred])
            plt.fill_between(df["Fixed value"], df["mean"] - df["std"], df["mean"] + df["std"],
                             color=c[idx_pred],
                             alpha=0.1)
            plt.ylim(0, 20)
            plt.title(predictor)
            self.save_figure(f"Partial_dependence_plot/{name}_{predictor}")
