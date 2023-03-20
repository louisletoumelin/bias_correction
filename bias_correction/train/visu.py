import numpy as np
import matplotlib.pyplot as plt

import os
import uuid
from typing import Union, Tuple

from bias_correction.utils_bc.decorators import pass_if_doesnt_has_module
from bias_correction.train.utils import create_folder_if_doesnt_exist
from bias_correction.train.experience_manager import ExperienceManager

try:
    import seaborn as sns

    _sns = True
except ModuleNotFoundError:
    _sns = False


def check_if_subfolder_in_filename(name_figure: str) -> bool:
    if "/" in name_figure:
        return True
    else:
        return False


def get_path_subfolder(name_figure: str) -> Tuple[str, str]:
    list_path = name_figure.split("/")[:-1]
    filename = name_figure.split("/")[-1]
    path_subfolder = '/'.join(list_path)
    return path_subfolder, filename


def _save_figure(name_figure: str,
                 save_path: str,
                 format_: str = "png",
                 svg: bool = False
                 ) -> None:
    ax = plt.gca()
    fig = ax.get_figure()
    uuid_str = str(uuid.uuid4())[:4]
    fig.savefig(save_path + f"{name_figure}_{uuid_str}.{format_}")
    if svg:
        fig.savefig(save_path + f"{name_figure}_{uuid_str}.svg")


def create_subfolder_if_necessary(name_figure: str,
                                  save_path: str) -> Tuple[str, str]:
    path_subfolder, filename = get_path_subfolder(name_figure)
    save_path = os.path.join(save_path, path_subfolder)
    save_path = save_path + '/'
    create_folder_if_doesnt_exist(save_path, _raise=False)
    return save_path, filename


def save_figure(name_figure: str,
                exp: Union[ExperienceManager, None] = None,
                save_path: str = None,
                format_: str = "png",
                svg: bool = False
                ) -> None:

    exp_is_provided = exp is not None
    save_path_is_provided = save_path is not None

    if not save_path_is_provided and exp_is_provided:
        save_path = exp.path_to_figures
    elif not save_path_is_provided and not exp_is_provided:
        save_path = os.getcwd()
    else:
        pass

    if exp_is_provided:
        exp.create_folder_if_doesnt_exist(save_path, _raise=False)

    subfolder_in_filename = check_if_subfolder_in_filename(name_figure)
    if subfolder_in_filename:
        save_path, name_figure = create_subfolder_if_necessary(name_figure, save_path)

    _save_figure(name_figure, save_path, format_=format_, svg=svg)


class StaticPlots:

    def __init__(self, exp=None):
        self.exp = exp

    @pass_if_doesnt_has_module()
    def plot_pair_plot_parameters(self, stations, figsize=(10, 10), s=15, hue_order=['Training', 'Test', 'Validation']):
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
            hue_order=hue_order,
            plot_kws={"s": s})
        save_figure("Pair_plot_param")

    @pass_if_doesnt_has_module()
    def plot_pair_plot_metrics(self, stations, figsize=(10, 10), s=15, hue_order=['Training', 'Test', 'Validation']):
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
            hue_order=hue_order,
            plot_kws={"s": s})
        save_figure("Pair_plot_metric")

    def plot_pairplot_all(self, stations, figsize=(10, 10), s=15, hue_order=['Training', 'Test', 'Validation']):
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
            hue_order=hue_order,
            plot_kws={"s": s})
        save_figure("Pair_plot_all")


#
# current_variable = self.exp.config['current_variable']
def plot_single_subplot(df,
                        key_model="UV_AROME",
                        key_obs="UV_obs",
                        scaling_variable="UV",
                        nb_columns=2,
                        id_plot=1,
                        s=1,
                        min_value=None,
                        max_value=None,
                        text_x=None,
                        text_y=None):
    # Get values
    obs = df[key_obs].values
    model = df[key_model].values

    # Get limits
    if scaling_variable == "UV":
        min_value = -1  # np.min(df["UV_obs"].values) - 5
        max_value = 25  # np.max(df["UV_obs"].values) + 5
        text_x = 0
        text_y = 21

    elif scaling_variable == "T2m":
        min_value = -40  # np.min(df["UV_obs"].values) - 5
        max_value = 40  # np.max(df["UV_obs"].values) + 5
        text_x = -38
        text_y = 38

    else:
        min_value = min_value
        max_value = max_value
        text_x = text_x
        text_y = text_y

    # Figure
    plt.subplot(1, nb_columns, id_plot)
    plt.scatter(obs, model, s=s)
    plt.plot(obs, obs, color='red')

    # Text
    try:
        plt.text(text_x, text_y, f"Mean bias {key_model}: {round(np.mean(model - obs), 2):.2f}")
        plt.text(text_x, text_y - 2, f"RMSE {key_model}: {round(np.sqrt(np.mean((model - obs) ** 2)), 2):.2f}")
        corr_coeff = df[[key_obs, key_model]].corr().iloc[0, 1]
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


def plot_1_1_models_vs_arome(df,
                             keys_models=["UV_nn", "UV_AROME"],
                             key_obs="UV_obs",
                             scaling_variable="UV",
                             figsize=(20, 10),
                             s=1):
    plt.figure(figsize=figsize)
    nb_columns = len(keys_models)
    for idx, key in enumerate(keys_models):
        plot_single_subplot(df, key, key_obs, scaling_variable, nb_columns, idx + 1, s=s)


class ModelVersusObsPlots:

    def __init__(self, exp=None):
        self.exp = exp

    def plot_1_1_all(self, df, keys_models=["UV_nn", "UV_AROME"], figsize=(20, 10), s=1, name="1_1_all"):
        current_variable = self.exp.config['current_variable']
        key_obs = f"{current_variable}_obs"
        plot_1_1_models_vs_arome(df,
                                 keys_models=keys_models,
                                 key_obs=key_obs,
                                 scaling_variable=current_variable,
                                 figsize=figsize,
                                 s=s)
        save_figure(f"Model_vs_obs/{name}")

    def plot_1_1_by_station(self, df, keys_models=["UV_nn", "UV_AROME"], figsize=(20, 10), s=1, name=""):
        current_variable = self.exp.config['current_variable']
        key_obs = f"{current_variable}_obs"
        for station in df["name"].unique():
            plot_1_1_models_vs_arome(df[df["name"] == station],
                                     keys_models=keys_models,
                                     key_obs=key_obs,
                                     scaling_variable=current_variable,
                                     figsize=figsize,
                                     s=s)
            plt.title(station)
            var_i = self.exp.config['current_variable']
            save_figure(f"Model_vs_obs_by_station/1_1_{station}_{var_i}_models_vs_{var_i}_obs_{name}")


def plot_evolution(df,
                   hue_names_to_plot=["bias_AROME", "bias_DEVINE"],
                   y_label_name="Bias",
                   fontsize=15,
                   figsize=(20, 15),
                   groupby="month"):

    if hasattr(df.index, groupby):
        index_groupby = getattr(df.index, groupby)
    else:
        index_groupby = groupby

    df.groupby(index_groupby).mean()[hue_names_to_plot].plot(figsize=figsize)
    plt.xlabel(groupby.capitalize(), fontsize=fontsize)
    plt.ylabel(y_label_name.capitalize(), fontsize=fontsize)


class SeasonalEvolution:

    def __init__(self, exp=None):
        self.exp = exp

    def plot_seasonal_evolution(self,
                                df,
                                metrics=["bias", "ae", "n_bias", "n_ae"],
                                fontsize=15,
                                figsize=(20, 15),
                                keys=["UV_nn", "UV_AROME"],
                                groupby="month",
                                name="Seasonal_evolution"):
        keys = ['_' + key.split('_')[1] for key in keys]
        for metric in metrics:
            list_metrics_to_plot = [f"{metric}{key}" for key in keys]
            plot_evolution(df,
                           hue_names_to_plot=list_metrics_to_plot,
                           y_label_name=metric,
                           fontsize=fontsize,
                           figsize=figsize,
                           groupby=groupby)
            save_figure(f"Seasonal_evolution/{name}", exp=self.exp)

    @staticmethod
    def plot_seasonal_evolution_by_station(df,
                                           metrics=["bias", "ae", "n_bias", "n_ae"],
                                           keys=["UV_nn", "UV_AROME"],
                                           groupby="month",
                                           fontsize=15,
                                           figsize=(20, 15),
                                           name=""):
        keys = ['_' + key.split('_')[1] for key in keys]
        for station in df["name"].unique():
            for metric in metrics:
                list_metrics_to_plot = [f"{metric}{key}" for key in keys]
                plot_evolution(df[df["name"] == station],
                               hue_names_to_plot=list_metrics_to_plot,
                               y_label_name=metric,
                               fontsize=fontsize,
                               figsize=figsize,
                               groupby=groupby)
                plt.title(station)
                save_figure(f"Seasonal_evolution_by_station/Seasonal_evolution_{station}_{name}")


class Leadtime:

    def __init__(self, exp=None):
        self.exp = exp

    @staticmethod
    def plot_lead_time(df,
                       metrics=["bias", "ae", "n_bias", "n_ae"],
                       keys=["UV_nn", "UV_AROME"],
                       groupby="lead_time",
                       fontsize=15,
                       figsize=(20, 15),
                       name="Lead_time"):
        keys = ['_' + key.split('_')[1] for key in keys]
        for metric in metrics:
            list_metrics_to_plot = [f"{metric}{key}" for key in keys]
            plot_evolution(df,
                           hue_names_to_plot=list_metrics_to_plot,
                           y_label_name=metric,
                           fontsize=fontsize,
                           figsize=figsize,
                           groupby=groupby)
            save_figure(f"Lead_time/{name}")

    @staticmethod
    def plot_lead_time_by_station(df,
                                  metrics=["bias", "ae", "n_bias", "n_ae"],
                                  keys=["UV_nn", "UV_AROME"],
                                  groupby="lead_time",
                                  fontsize=15,
                                  figsize=(20, 15)):

        keys = ['_' + key.split('_')[1] for key in keys]
        for station in df["name"].unique():
            for metric in metrics:
                list_metrics_to_plot = [f"{metric}{key}" for key in keys]
                plot_evolution(df[df["name"] == station],
                               hue_names_to_plot=list_metrics_to_plot,
                               y_label_name=metric,
                               fontsize=fontsize,
                               figsize=figsize,
                               groupby=groupby)

                save_figure(f"Lead_time/Lead_time_{station}")


def plot_boxplot_models(df,
                        carac="laplacian",
                        metric="bias",
                        models_names=["Neural Network", "AROME"],
                        showfliers=False,
                        orient="v",
                        figsize=(15, 12),
                        ):

    df_melted = df.melt(id_vars=["name", f"class_{carac}"],
                        value_vars=models_names,
                        var_name='Model',
                        value_name=metric.capitalize())

    plt.figure(figsize=figsize)

    sns.boxplot(data=df_melted,
                y=metric.capitalize(),
                x=f"class_{carac}",
                hue='Model',
                orient=orient,
                showfliers=showfliers)

    sns.despine(trim=True, left=True)


class Boxplots:

    def __init__(self, exp):
        self.exp = exp

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

                keys = dict_keys.keys()
                values = dict_keys.values()
                new_columns = {f"{metric}{old_name}": new_name for (old_name, new_name) in zip(keys, values)}
                df = df.rename(columns=new_columns)

                plot_boxplot_models(df,
                                    carac=carac,
                                    metric=metric,
                                    models_names=list(dict_keys.values()),
                                    showfliers=showfliers,
                                    orient="v",
                                    figsize=figsize,
                                    )

                save_figure(f"Boxplots/{name}")


class VizualizationResults(Boxplots, Leadtime, SeasonalEvolution, ModelVersusObsPlots, StaticPlots):

    def __init__(self, exp=None):
        super().__init__(exp)


