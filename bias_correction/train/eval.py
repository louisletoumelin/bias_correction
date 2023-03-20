import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

try:
    import seaborn as sns
    sns_ = True
except (ImportError, ModuleNotFoundError):
    sns_ = False

from bias_correction.train.visu import VizualizationResults
from bias_correction.train.metrics import get_metric


class CustomEvaluation(VizualizationResults):

    def __init__(self, exp, data, mode="test", keys=["_AROME", "_nn"], quick=False):
        super().__init__(exp)

        self.exp = exp
        self.data = data
        self.current_variable = self.exp.config["current_variable"]
        self.mode = mode

        if self.exp.config["get_intermediate_output"]:
            keys.append("_int")
        self._set_key_attributes(keys)
        self._set_key_list(keys)
        self.df_results = self.create_df_results()
        if not quick:
            self._add_metric_to_df_results()
            self._add_topo_carac_to_df_results()
            self.classify_topo_carac()
            self.classify_alti()
            self.classify_forecast_term()

    def _set_key_attributes(self, keys):
        self.key_obs = f"{self.current_variable}_obs"
        for key in keys:
            setattr(self, f"key{key}", f"{self.current_variable}{key}")

    def _set_key_list(self, keys):
        self.keys = [f"{self.current_variable}{key}" for key in keys]

    def create_df_results(self):

        mode = self.mode
        assert hasattr(self.data, f"predicted_{mode}")
        df = self.data.get_predictions(mode)

        if self.current_variable == "UV":
            return self.create_df_speed(df)
        else:
            return self.create_df_temp(df)

    def create_df_temp(self, df):
        """
        Select variables for temperature predictions

        :param df: DataFrame with data
        :type df: pandas.DataFrame
        :return: DataFrame with temperature variables
        """
        labels = self.data.get_labels(self.mode)
        inputs = self.data.get_inputs(self.mode)
        df["T2m_obs"] = labels.values
        df["T2m_AROME"] = inputs["Tair"].values
        columns = ["name"] + self.keys
        return df[columns]

    def create_df_speed(self, df):

        labels = self.data.get_labels(self.mode)
        inputs = self.data.get_inputs(self.mode)

        if "component" in self.data.config["type_of_output"]:

            df["UV_obs"] = np.sqrt(labels["U_obs"] ** 2 + labels["V_obs"] ** 2)
            df["UV_AROME"] = inputs["Wind"].values

        elif "speed" in self.data.config["type_of_output"]:

            df["UV_obs"] = labels.values
            df["UV_AROME"] = inputs["Wind"].values

        columns = ["name", f"{self.current_variable}_obs"] + self.keys
        return df[columns]

    def _add_metric_to_df_results(self, metrics=["bias", "n_bias", "ae", "n_ae"]):
        for metric in metrics:
            metric_func = get_metric(metric)
            for key in self.keys:
                result = metric_func(self.df_results[self.key_obs].values, self.df_results[key].values)
                key = '_' + key.split('_')[1]
                self.df_results[f"{metric}{key}"] = result

    def _add_topo_carac_to_df_results(self, topo_carac=['mu', 'curvature', 'tpi_500', 'tpi_2000', 'laplacian', 'alti',
                                                        'country']):
        stations = self.data.get_stations()
        for station in self.df_results["name"].unique():
            filter_df = self.df_results["name"] == station
            filter_s = stations["name"] == station
            for carac in topo_carac:
                carac = carac + '_NN_0' if carac not in ["alti", "country"] else carac
                if hasattr(self.df_results, carac):
                    self.df_results.loc[filter_df, carac] = stations.loc[filter_s, carac].values[0]
                else:
                    self.df_results[carac] = np.nan
                    self.df_results.loc[filter_df, carac] = stations.loc[filter_s, carac].values[0]

    def classify_topo_carac(self, topo_carac=['mu', 'curvature', 'tpi_500', 'tpi_2000', 'laplacian', 'alti']):
        stations = self.data.get_stations()
        for carac in topo_carac:

            if not hasattr(self.df_results, f"class_{carac}"):
                self.df_results[f"class_{carac}"] = np.nan

            carac_nn = carac + '_NN_0' if carac not in ["alti", "country"] else carac

            q25 = np.quantile(stations[carac_nn].values, 0.25)
            q50 = np.quantile(stations[carac_nn].values, 0.5)
            q75 = np.quantile(stations[carac_nn].values, 0.75)

            filter_1 = (self.df_results[carac_nn] <= q25)
            filter_2 = (self.df_results[carac_nn] > q25) & (self.df_results[carac_nn] <= q50)
            filter_3 = (self.df_results[carac_nn] > q50) & (self.df_results[carac_nn] <= q75)
            filter_4 = (self.df_results[carac_nn] > q75)

            self.df_results.loc[filter_1, f"class_{carac}"] = "$x \leq q_{25}$"
            self.df_results.loc[filter_2, f"class_{carac}"] = "$q_{25}<x \leq q_{50}$"
            self.df_results.loc[filter_3, f"class_{carac}"] = "$q_{50}<x \leq q_{75}$"
            self.df_results.loc[filter_4, f"class_{carac}"] = "$q_{75}<x$"

            print(f"Quantiles {carac}: ", q25, q50, q75)

    def classify_alti(self):
        self.df_results["class_alti0"] = np.nan

        filter_1 = (self.df_results["alti"] <= 500)
        filter_2 = (500 < self.df_results["alti"]) & (self.df_results["alti"] <= 1000)
        filter_3 = (1000 < self.df_results["alti"]) & (self.df_results["alti"] <= 2000)
        filter_4 = (2000 < self.df_results["alti"])

        self.df_results.loc[filter_1, "class_alti0"] = "$Elevation [m] \leq 500$"
        self.df_results.loc[filter_2, "class_alti0"] = "$500<Elevation [m] \leq 1000$"
        self.df_results.loc[filter_3, "class_alti0"] = "$1000<Elevation [m] \leq 2000$"
        self.df_results.loc[filter_4, "class_alti0"] = "$2000<Elevation [m]$"

        print(len(self.df_results.loc[filter_1, "name"].unique()))
        print(len(self.df_results.loc[filter_2, "name"].unique()))
        print(len(self.df_results.loc[filter_3, "name"].unique()))
        print(len(self.df_results.loc[filter_4, "name"].unique()))

    def classify_forecast_term(self):

        def compute_lead_time(hour):
            return (hour - 6) % 24 + 6

        self.df_results["lead_time"] = compute_lead_time(self.df_results.index.hour.values)

    @staticmethod
    def _df2metric(df, metric_name, current_variable, key_obs, keys):
        metric_func = get_metric(metric_name)
        results = []
        for key in keys:
            metric = metric_func(df[key_obs].values, df[key].values)
            print(f"\n{metric_name}{key}")
            print(metric)
            results.append(metric)
        return results

    def df2metric(self, metric_name):
        return self._df2metric(self.df_results, metric_name, self.current_variable, self.key_obs, self.keys)

    def df2mae(self):
        return self.df2metric("mae")

    def df2rmse(self):
        return self.df2metric("rmse")

    def df2mbe(self):
        return self.df2metric("mbe")

    def df2m_n_be(self, min_obs, min_model):
        for key in self.keys:
            filter_obs = self.df_results[self.key_obs] >= min_obs
            filter_model = self.df_results[key] >= min_model
            df = self.df_results[filter_obs & filter_model]
            self._df2metric(df, "m_n_be", self.current_variable, self.key_obs, self.keys)

    def df2m_n_ae(self, min_obs, min_model):
        for key in self.keys:
            key_model = f"{self.current_variable}{key}"
            filter_obs = self.df_results[self.key_obs] >= min_obs
            filter_model = self.df_results[key_model] >= min_model
            df = self.df_results[filter_obs & filter_model]
            self._df2metric(df, "m_n_ae", self.current_variable, self.key_obs, self.keys)

    def df2ae(self):
        return self._df2metric(self.df_results, "ae", self.current_variable, self.key_obs, self.keys)

    def df2bias(self):
        return self._df2metric(self.df_results, "bias", self.current_variable, self.key_obs, self.keys)

    def df2correlation(self):
        for station in self.df_results["name"].unique():
            filter_station = self.df_results["name"] == station
            self.df_results.loc[filter_station, :] = self.df_results.loc[filter_station, :].sort_index()

        return self.df2metric("corr")


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

    def plot_feature_importance(self, mode, width=0.8, epsilon=0.01, figsize=(15, 12)):

        df_rmse, df_ae, str_rmse, str_ae = self.compute_feature_importance(mode, epsilon=epsilon)

        # RMSE
        self._plot_bar_with_error(df_rmse,
                                  "Predictor",
                                  str_rmse,
                                  "std",
                                  "Feature_importance/Feature_importance_rmse",
                                  width=width,
                                  figsize=figsize)

        # AE
        self._plot_bar_with_error(df_ae,
                                  "Predictor",
                                  str_ae,
                                  "std",
                                  "Feature_importance/Feature_importance_ae",
                                  width=width,
                                  figsize=figsize)

    def plot_partial_dependence(self, mode):
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
            self.save_figure(f"Partial_dependence_plot/Partial_dependence_plot_{predictor}")
