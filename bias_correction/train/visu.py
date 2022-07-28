import os

import numpy as np
import matplotlib.pyplot as plt

import uuid
from bias_correction.utils_bc.decorators import pass_if_doesnt_has_module

try:
    import seaborn as sns

    _sns = True
except ModuleNotFoundError:
    _sns = False


class StaticPlots:

    def __init__(self, exp):
        self.exp = exp

    @staticmethod
    def check_if_subfolder_in_filename(name_figure):
        if "/" in name_figure:
            return True
        else:
            return False

    @staticmethod
    def get_path_subfolder(name_figure):
        list_path = name_figure.split("/")[:-1]
        filename = name_figure.split("/")[-1]
        path_subfolder = '/'.join(list_path)
        return path_subfolder, filename

    @staticmethod
    def _save_figure(name_figure, save_path, format_="png", svg=False):
        ax = plt.gca()
        fig = ax.get_figure()
        uuid_str = str(uuid.uuid4())[:4]
        fig.savefig(save_path + f"{name_figure}_{uuid_str}.{format_}")
        if svg:
            fig.savefig(save_path + f"{name_figure}_{uuid_str}.svg")

    def create_subfolder_if_necessary(self, name_figure, save_path):
        path_subfolder, filename = self.get_path_subfolder(name_figure)
        save_path = os.path.join(save_path, path_subfolder)
        save_path = save_path + '/'
        self.exp.create_folder_if_doesnt_exist(save_path, _raise=False)
        return save_path, filename

    def save_figure(self, name_figure, save_path=None, format_="png", svg=False):

        if (save_path is None) and (hasattr(self, "exp")):
            save_path = self.exp.path_to_figures
        elif (save_path is None) and not hasattr(self, "exp"):
            save_path = os.getcwd()
        else:
            pass

        if hasattr(self, "exp"):
            self.exp.create_folder_if_doesnt_exist(save_path, _raise=False)

        subfolder_in_filename = self.check_if_subfolder_in_filename(name_figure)
        if subfolder_in_filename:
            save_path, name_figure = self.create_subfolder_if_necessary(name_figure, save_path)
        self._save_figure(name_figure, save_path, format_=format_, svg=svg)

    @pass_if_doesnt_has_module()
    def plot_pair_plot_parameters(self, stations, figsize=(10, 10), s=15):
        """Pair plot parameters"""
        if not _sns:
            raise ModuleNotFoundError("Seaborn is required for this function")
        # Figure 1:
        stations = stations.rename(columns={"alti": "Elevation [m]",
                                            "tpi_500_NN_0": "TPI [m]",
                                            "mu_NN_0": "Slope []",
                                            "Y": "Y coord. [m]"})
        sns.set_style("ticks", {'axes.grid': True})
        plt.figure(figsize=figsize)
        sns.pairplot(
            data=stations[["Elevation [m]", "TPI [m]", "Slope []", "Y coord. [m]", "mode"]],
            hue="mode",
            plot_kws={"s": s})
        self.save_figure("Pair_plot_param")

    @pass_if_doesnt_has_module()
    def plot_pair_plot_metrics(self, stations, figsize=(10, 10), s=15):
        """Pair plot metrics"""
        metric_computed = "rmse" in stations or "mbe" in stations or "corr" in stations or "mae" in stations
        assert metric_computed, "metrics (rmse, mbe, corr, mae) must be computed befor plotting this function"

        stations = stations.rename(columns={"rmse": "Root mean squared Error [$m\:s^{-1}$]",
                                            "mbe": "Mean bias [$m\:s^{-1}$]",
                                            "corr": "Correlation []",
                                            "mae": "Mean absolute error [$m\:s^{-1}$]"})
        sns.set_style("ticks", {'axes.grid': True})
        plt.figure(figsize=figsize)
        sns.pairplot(
            data=stations[["Root mean squared Error [$m\:s^{-1}$]",
                           "Mean bias [$m\:s^{-1}$]",
                           "Correlation []",
                           "Mean absolute error [$m\:s^{-1}$]",
                           "mode"]],
            hue="mode",
            plot_kws={"s": s})
        self.save_figure("Pair_plot_metric")

    def plot_pairplot_all(self, stations, figsize=(10, 10), s=15):
        """Pair plot metrics and parameters"""
        stations = stations.rename(columns={"alti": "Elevation [m]",
                                            "tpi_500_NN_0": "TPI [m]",
                                            "mu_NN_0": "Slope []",
                                            "Y": "Y coord. [m]",
                                            "rmse": "Root mean squared Error [$m\:s^{-1}$]",
                                            "mbe": "Mean bias [$m\:s^{-1}$]",
                                            "corr": "Correlation []",
                                            "mae": "Mean absolute error [$m\:s^{-1}$]"})

        sns.set_style("ticks", {'axes.grid': True})
        plt.figure(figsize=figsize)
        sns.pairplot(
            data=stations[["Elevation [m]",
                           "TPI [m]",
                           "Slope []",
                           "Y coord. [m]",
                           "Root mean squared Error [$m\:s^{-1}$]",
                           "Mean bias [$m\:s^{-1}$]",
                           "Correlation []",
                           "Mean absolute error [$m\:s^{-1}$]",
                           "mode"]],
            hue="mode",
            plot_kws={"s": s})
        self.save_figure("Pair_plot_all")


class ModelVersusObsPlots(StaticPlots):

    def __init__(self, exp=None):
        super().__init__(exp)

    def _plot_1_1_model_vs_arome(self, df, keys=["UV_nn", "UV_AROME"], figsize=(20, 10), s=1):
        nb_columns = len(keys)
        for idx, key in enumerate(keys):
            self.plot_1_1_subplot(df, key, nb_columns, idx+1, s=s)

    def plot_1_1_subplot(self, df, key_model, nb_columns, id_plot, s=1):

        # Get values
        obs = df[f"{self.exp.config['current_variable']}_obs"].values
        model = df[key_model].values

        # Get limits
        if self.exp.config['current_variable'] == "UV":
            min_value = -1  # np.min(df["UV_obs"].values) - 5
            max_value = 25  # np.max(df["UV_obs"].values) + 5
            text_x = 0
            text_y = 21
        elif self.exp.config['current_variable'] == "T2m":
            min_value = -40  # np.min(df["UV_obs"].values) - 5
            max_value = 40  # np.max(df["UV_obs"].values) + 5
            text_x = -38
            text_y = 38

        # Figure
        plt.subplot(1, nb_columns, id_plot)
        plt.scatter(obs, model, s=s)
        plt.plot(obs, obs, color='red')

        # Text
        try:
            plt.text(text_x, text_y, f"Mean bias {key_model}: {round(np.mean(model - obs), 2):.2f}")
            plt.text(text_x, text_y - 2, f"RMSE {key_model}: {round(np.sqrt(np.mean((model - obs) ** 2)), 2):.2f}")
            corr_coeff = df[[f"{self.exp.config['current_variable']}_obs", key_model]].corr().iloc[0, 1]
            plt.text(text_x, text_y - 4, f"Corr. {key_model}: {round(corr_coeff, 2):.2f}")
        except:
            print("Error in text figure")

        """
        # Additionnal text
        try:  # ${incomeTax:.2f}
            if (stations is not None) and (station is not None):
                laplacian = stations["laplacian_NN_0"][stations["name"] == station].values[0]
                tpi_500 = stations["tpi_500_NN_0"][stations["name"] == station].values[0]
                tpi_1000 = stations["tpi_2000_NN_0"][stations["name"] == station].values[0]
                mu = stations["mu_NN_0"][stations["name"] == station].values[0]
                curvature = stations["curvature_NN_0"][stations["name"] == station].values[0]
                plt.text(text_x, text_y - 7, f"lap: {round(laplacian, 2):.2f}")
                plt.text(text_x, text_y - 9, f"tpi_500: {round(tpi_500, 2):.2f}")
                plt.text(text_x, text_y - 11, f"tpi_2000: {round(tpi_1000, 2):.2f}")
                plt.text(text_x, text_y - 13, f"mu: {round(mu, 2):.2f}")
                plt.text(text_x, text_y - 15, f"cur: {round(curvature, 2):.2f}")
        except Exception as e:
            print("Error in text figure")
            print(e)
        """

        # xlim and ylim
        plt.xlim(min_value, max_value)
        plt.ylim(min_value, max_value)

    def plot_1_1_all(self, df, key_model, figsize=(20, 10), s=1, name="1_1_all"):
        self._plot_1_1_model_vs_arome(df, key_model, figsize=figsize, s=s)
        self.save_figure(f"Model_vs_obs/{name}")

    def plot_1_1_by_station(self, df, key_model, figsize=(20, 10), s=1, name=""):
        for station in df["name"].unique():
            plt.figure(figsize=figsize)
            self._plot_1_1_model_vs_arome(df[df["name"] == station], key_model, figsize=figsize, s=s)
            plt.title(station)
            var_i = self.exp.config['current_variable']
            self.save_figure(f"Model_vs_obs_by_station/1_1_{station}_{var_i}_models_vs_{var_i}_obs_{name}")


class SeasonalEvolution(ModelVersusObsPlots):

    def __init__(self, exp=None):
        super().__init__(exp)

    @staticmethod
    def _plot_seasonal_evolution(df, metric, fontsize=15, figsize=(20, 15), keys=["UV_nn", "UV_AROME"], groupby="month"):
        keys = ['_'+key.split('_')[1] for key in keys]
        list_metrics_to_plot = [f"{metric}{key}" for key in keys]
        if hasattr(df.index, groupby):
            index_groupby = getattr(df.index, groupby)
        else:
            index_groupby = groupby
        df.groupby(index_groupby).mean()[list_metrics_to_plot].plot(figsize=figsize)
        plt.xlabel(groupby.capitalize(), fontsize=fontsize)
        plt.ylabel(metric.capitalize(), fontsize=fontsize)

    def plot_seasonal_evolution(self,
                                df,
                                metrics=["bias", "ae", "n_bias", "n_ae"],
                                keys=["UV_nn", "UV_AROME"],
                                groupby="month",
                                fontsize=15,
                                figsize=(20, 15),
                                name="Seasonal_evolution"):
        for metric in metrics:
            self._plot_seasonal_evolution(df,
                                          metric,
                                          keys=keys,
                                          fontsize=fontsize,
                                          figsize=figsize,
                                          groupby=groupby)
            self.save_figure(f"Seasonal_evolution/{name}")

    def plot_seasonal_evolution_by_station(self,
                                           df,
                                           metrics=["bias", "ae", "n_bias", "n_ae"],
                                           keys=["UV_nn", "UV_AROME"],
                                           groupby="month",
                                           fontsize=15,
                                           figsize=(20, 15),
                                           name=""):
        for station in df["name"].unique():
            for metric in metrics:
                self._plot_seasonal_evolution(df[df["name"] == station],
                                              metric,
                                              keys=keys,
                                              fontsize=fontsize,
                                              figsize=figsize,
                                              groupby=groupby)
                plt.title(station)
                self.save_figure(f"Seasonal_evolution_by_station/Seasonal_evolution_{station}_{name}")


class Leadtime(SeasonalEvolution):

    def __init__(self, exp=None):
        super().__init__(exp)

    def plot_lead_time(self,
                       df,
                       metrics=["bias", "ae", "n_bias", "n_ae"],
                       keys=["UV_nn", "UV_AROME"],
                       groupby="lead_time",
                       fontsize=15,
                       figsize=(20, 15),
                       name="Lead_time"):
        for metric in metrics:
            self._plot_seasonal_evolution(df,
                                          metric,
                                          keys=keys,
                                          fontsize=fontsize,
                                          figsize=figsize,
                                          groupby=groupby)
            self.save_figure(f"Lead_time/{name}")

    def plot_lead_time_by_station(self,
                                  df,
                                  metrics=["bias", "ae", "n_bias", "n_ae"],
                                  keys=["UV_nn", "UV_AROME"],
                                  groupby="month",
                                  fontsize=15,
                                  figsize=(20, 15)):
        keys = ['_' + key.split('_')[1] for key in keys]
        for station in df["name"].unique():
            for metric in metrics:
                self._plot_seasonal_evolution(df[df["name"] == station],
                                              metric,
                                              keys=keys,
                                              fontsize=fontsize,
                                              figsize=figsize,
                                              groupby=groupby)

                self.save_figure(f"Lead_time_{station}")


class VizualizationResults(Leadtime):

    def __init__(self, exp=None):
        super().__init__(exp)

    @staticmethod
    def _plot_boxplot_topo_carac(df,
                                 carac,
                                 metric,
                                 dict_keys={"_nn": "Neural Network", "_AROME": "AROME"},
                                 showfliers=False,
                                 figsize=(15, 10)
                                 ):
        keys = dict_keys.keys()
        values = dict_keys.values()
        new_columns = {f"{metric}{old_name}": new_name for (old_name, new_name) in zip(keys, values)}
        df = df.rename(columns=new_columns)
        df_melted = df.melt(id_vars=["name", f"class_{carac}"],
                            value_vars=list(dict_keys.values()),
                            var_name='Model',
                            value_name=metric.capitalize())

        plt.figure(figsize=figsize)

        sns.boxplot(data=df_melted,
                    y=metric.capitalize(),
                    x=f"class_{carac}",
                    hue='Model',
                    orient="v",
                    showfliers=showfliers)
        sns.despine(trim=True, left=True)

    @pass_if_doesnt_has_module()
    def plot_boxplot_topo_carac(self,
                                df,
                                metrics=["bias", "ae", "n_bias", "n_ae"],
                                topo_carac=['mu', 'curvature', 'tpi_500', 'tpi_2000', 'laplacian', 'alti'],
                                dict_keys={"_nn": "Neural Network", "_AROME": "AROME"},
                                showfliers=False,
                                figsize=(15, 10),
                                name="Boxplot_topo_carac"):
        for metric in metrics:
            for carac in topo_carac:
                self._plot_boxplot_topo_carac(df,
                                              carac,
                                              metric,
                                              dict_keys=dict_keys,
                                              figsize=figsize,
                                              showfliers=showfliers)
                self.save_figure(f"Boxplots/{name}")
