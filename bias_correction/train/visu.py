import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as cm_plt

import os
import uuid
from typing import Union, Tuple, Dict, MutableSequence
from functools import partial

from bias_correction.utils_bc.decorators import pass_if_doesnt_has_module, pass_if_doesnt_have_seaborn_version
from bias_correction.train.utils import create_folder_if_doesnt_exist
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.windrose import plot_windrose
from bias_correction.train.ale import ale_plot

try:
    import seaborn as sns

    _sns = True
except ModuleNotFoundError:
    _sns = False

KEY2NEW_NAMES = {"_AROME": "$AROME_{forecast}$",
                 "_D": "DEVINE",
                 "_nn": "Neural Network + DEVINE",
                 "_int": "Neural Network",
                 "_A": "$AROME_{analysis}$",
                 "_DA": "$AROME_{analysis}$ + DEVINE",
                 }

METRICS2NAMES = {"bias": "Wind speed bias [$m\:s^{-1}$]",
                 "n_bias": "Wind speed normalized bias [$m\:s^{-1}$]",
                 "ae": "Wind speed \nabsolute error [$m\:s^{-1}$]",
                 "n_ae": "Wind speed \n normalized absolute error [$m\:s^{-1}$]",
                 "bias_direction": "Wind direction bias [°]",
                 "abs_bias_direction": "Wind direction \nabsolute error [°]"}

CARAC2NAME = {"lead_time": "Forecast lead time [hour]",
              "month": "Month",
              'class_mu': "Slope category of the observation stations",
              'class_curvature': "Curvature category of the observation stations",
              'class_tpi_500': "$TPI_{500m}$ category of the observation stations",
              'class_tpi_2000': "$TPI_{2000m}$ category of the observation stations",
              'class_laplacian': "Laplacian category of the observation stations",
              'class_alti': "Elevation category of the observation stations"
              }

CLASS2CARAC = {'class_mu': "Slope",
               'class_curvature': "Curvature",
               'class_tpi_500': "TPI_{500m}",
               'class_tpi_2000': "TPI_{2000m}",
               'class_laplacian': "Laplacian",
               'class_alti': "Elevation"
               }


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


def create_subfolder_if_necessary(name_figure: str,
                                  save_path: str) -> Tuple[str, str]:
    path_subfolder, filename = get_path_subfolder(name_figure)
    save_path = os.path.join(save_path, path_subfolder)
    save_path = save_path + '/'
    create_folder_if_doesnt_exist(save_path, _raise=False)
    return save_path, filename


def _save_figure(name_figure: str,
                 save_path: str,
                 format_: str = "png",
                 svg: bool = False,
                 fig: Union[None, matplotlib.figure.Figure] = None
                 ) -> None:
    if fig is None:
        ax = plt.gca()
        fig = ax.get_figure()
    uuid_str = str(uuid.uuid4())[:4]
    fig.savefig(save_path + f"{name_figure}_{uuid_str}.{format_}")
    print("Saving figure: " + save_path + f"{name_figure}_{uuid_str}.{format_}")
    if svg:
        fig.savefig(save_path + f"{name_figure}_{uuid_str}.svg")


def save_figure(name_figure: str,
                exp: Union[ExperienceManager, None] = None,
                save_path: str = None,
                format_: str = "png",
                svg: bool = False,
                fig: Union[None, matplotlib.figure.Figure] = None
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
        create_folder_if_doesnt_exist(save_path, _raise=False)

    subfolder_in_filename = check_if_subfolder_in_filename(name_figure)
    if subfolder_in_filename:
        save_path, name_figure = create_subfolder_if_necessary(name_figure, save_path)

    _save_figure(name_figure, save_path, format_=format_, svg=svg, fig=fig)


class StaticPlots:

    def __init__(self, exp: Union[ExperienceManager, None] = None
                 ) -> None:
        self.exp = exp

    @pass_if_doesnt_has_module()
    def plot_pair_plot_parameters(self,
                                  stations: pd.DataFrame,
                                  figsize: Tuple[int, int] = (10, 10),
                                  s: int = 15,
                                  hue_order: Tuple[str] = ('Training', 'Test', 'Validation')
                                  ) -> None:
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
            hue_order=list(hue_order),
            plot_kws={"s": s})
        save_figure("Pair_plot_param", exp=self.exp)

    @pass_if_doesnt_has_module()
    def plot_pair_plot_metrics(self,
                               stations: pd.DataFrame,
                               figsize: Tuple[int, int] = (10, 10),
                               s: int = 15,
                               hue_order: Tuple[str] = ('Training', 'Test', 'Validation')
                               ) -> None:
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
            hue_order=list(hue_order),
            plot_kws={"s": s})
        save_figure("Pair_plot_metric", exp=self.exp)

    def plot_pairplot_all(self,
                          stations: pd.DataFrame,
                          figsize: Tuple[int, int] = (10, 10),
                          s: int = 15,
                          hue_order: Tuple[str] = ('Training', 'Test', 'Validation')
                          ) -> None:
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
            hue_order=list(hue_order),
            plot_kws={"s": s})
        save_figure("Pair_plot_all", exp=self.exp)


#
# current_variable = self.exp.config['current_variable']
def plot_single_subplot(df: pd.DataFrame,
                        key_model: str = "UV_AROME",
                        key_obs: str = "UV_obs",
                        scaling_variable: str = "UV",
                        nb_columns: int = 2,
                        id_plot: int = 1,
                        s: int = 1,
                        min_value: Union[float, None] = None,
                        max_value: Union[float, None] = None,
                        text_x: str = None,
                        text_y: str = None,
                        color: str = "C0",
                        print_: bool = False
                        ) -> None:
    # Get values
    obs = df[key_obs].values
    model = df[key_model].values

    if print_:
        print(f"key_model {key_model}, key_obs {key_obs}, nb obs {len(obs)}")

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
    plt.scatter(obs, model, c=color, s=s)
    plt.plot(obs, obs, color='black')

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


def plot_single_1_1(df: pd.DataFrame,
                    key_model: str = "UV_AROME",
                    key_obs: str = "UV_obs",
                    scaling_variable: str = "UV",
                    nb_columns: int = 2,
                    s: int = 1,
                    min_value: Union[float, None] = None,
                    max_value: Union[float, None] = None,
                    text_x: str = None,
                    text_y: str = None,
                    color: str = "C0",
                    figsize: Tuple[int, int] = (15, 15),
                    density: bool = False,
                    xlabel: Union[str, None] = None,
                    ylabel: Union[str, None] = None,
                    fontsize: float = 20,
                    plot_text: bool = True,
                    print_=False
                    ) -> Union[None, matplotlib.figure.Figure]:
    # Get values
    obs = df[key_obs].values
    model = df[key_model].values

    if print_:
        print(f"key_model: {key_model}, key_obs: {key_obs}, nb of obs: {len(obs)}")

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

    if scaling_variable == "UV_DIR":
        min_value = -1  # np.min(df["UV_obs"].values) - 5
        max_value = 365  # np.max(df["UV_obs"].values) + 5
        text_x = 0
        text_y = 21

    else:
        min_value = min_value
        max_value = max_value
        text_x = text_x
        text_y = text_y

    # Figure
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.scatter(obs, model, c=color, s=s)
    ax.plot(obs, obs, color='black')
    # xlim and ylim
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize)

    if density:
        sns.kdeplot(df, x=key_obs, y=key_model, color="black", ax=ax).set(xlim=(0), ylim=(0))

    # Text
    if plot_text:
        try:
            ax.text(text_x, text_y, f"Mean bias {key_model}: {round(np.mean(model - obs), 2):.2f}")
            ax.text(text_x, text_y - 2, f"RMSE {key_model}: {round(np.sqrt(np.mean((model - obs) ** 2)), 2):.2f}")
            corr_coeff = df[[key_obs, key_model]].corr().iloc[0, 1]
            ax.text(text_x, text_y - 4, f"Corr. {key_model}: {round(corr_coeff, 2):.2f}")
        except:
            print("Error in text figure")

    return fig


def plot_1_1_multiple_subplots(df: pd.DataFrame,
                               keys_models: Tuple[str] = ("UV_nn", "UV_AROME"),
                               key_obs: str = "UV_obs",
                               scaling_variable: str = "UV",
                               figsize: Tuple[int, int] = (20, 10),
                               s: int = 1,
                               color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4"),
                               print_: bool = False
                               ) -> None:
    plt.figure(figsize=figsize)
    nb_columns = len(keys_models)
    for idx, key in enumerate(keys_models):
        plot_single_subplot(df,
                            key,
                            key_obs,
                            scaling_variable,
                            nb_columns,
                            idx + 1,
                            color=color[idx],
                            s=s,
                            print_=print_)


class ModelVersusObsPlots:

    def __init__(self, exp=None):
        self.exp = exp

    def plot_1_1_all(self,
                     df: pd.DataFrame,
                     keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                     figsize: Tuple[int, int] = (15, 15),
                     s: int = 1,
                     name: str = "1_1_all",
                     color: Tuple[str, ...] = ("C1", "C0", "C2", "C3", "C4"),
                     density: bool = False,
                     plot_text: bool = True,
                     fontsize: float = 25,
                     xlabel: Union[str, None] = None,
                     ylabel: Union[str, None] = None,
                     print_: bool = False
                     ) -> None:

        current_variable = self.exp.config['current_variable']
        key_obs = f"{current_variable}_obs"
        for idx, key in enumerate(keys):
            if _sns:
                sns.set_style("ticks", {'axes.grid': True})
            print("debug plot_1_1_all")
            print(list(df.columns))
            fig = plot_single_1_1(df, key, key_obs, current_variable,
                                  s=s,
                                  figsize=figsize,
                                  color=color[idx],
                                  density=density,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  plot_text=plot_text,
                                  fontsize=fontsize,
                                  print_=print_)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            save_figure(f"Model_vs_obs/{name}_{key}", exp=self.exp, svg=True, fig=fig)

    def plot_1_1_by_station(self,
                            df: pd.DataFrame,
                            keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                            s: int = 1,
                            name: str = "",
                            color: Tuple[str, ...] = ("C1", "C0", "C2", "C3", "C4"),
                            figsize: Tuple[int, int] = (40, 10),
                            print_: bool = False
                            ) -> None:
        current_variable = self.exp.config['current_variable']
        key_obs = f"{current_variable}_obs"
        for station in df["name"].unique():
            plot_1_1_multiple_subplots(df[df["name"] == station],
                                       keys_models=keys,
                                       key_obs=key_obs,
                                       scaling_variable=current_variable,
                                       figsize=figsize,
                                       s=s,
                                       color=color,
                                       print_=print_)
            plt.title(station)
            var_i = self.exp.config['current_variable']
            save_figure(f"Model_vs_obs_by_station/1_1_{station}_{var_i}_models_vs_{var_i}_obs_{name}", exp=self.exp)


def plot_evolution(df: pd.DataFrame,
                   hue_names_to_plot: Tuple[str] = ("bias_AROME", "bias_DEVINE"),
                   y_label_name: str = "Bias",
                   fontsize: int = 15,
                   figsize: Tuple[int, int] = (20, 15),
                   groupby: str = "month",
                   color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4"),
                   print_: bool = False,
                   yerr: Union[bool, None] = None,
                   errorbar: Union[str, None] = None,
                   alpha: float = 0.15
                   ) -> None:
    if isinstance(groupby, list):
        groupby = groupby[0]

    if hasattr(df.index, groupby):
        index_groupby = getattr(df.index, groupby)
    else:
        index_groupby = groupby

    plt.figure(figsize=figsize)
    ax = plt.gca()
    if print_:
        print(f"hue_names_to_plot {hue_names_to_plot}, "
              f"y_label_name {y_label_name}, "
              f"groupby {groupby}, "
              f"nb obs {len(df)}")

    dict_color = {key: value for key, value in zip(list(hue_names_to_plot), list(color))}

    if yerr:
        if groupby not in df:
            df[groupby] = getattr(df.index, groupby)
        sns.lineplot(data=df, x=groupby, y="year", hue=hue_names_to_plot, errorbar=errorbar)
    else:
        alpha = None
        df.groupby(index_groupby).mean()[list(hue_names_to_plot)].plot(color=dict_color,
                                                                       ax=ax,
                                                                       alpha=alpha)
    if _sns:
        sns.set_style("ticks", {'axes.grid': True})

    plt.xlabel(groupby.capitalize(), fontsize=fontsize)
    plt.ylabel(y_label_name.capitalize(), fontsize=fontsize)


def _old_names2new_names(df, keys, key2old_name):
    old2new_names = {key2old_name[key]: KEY2NEW_NAMES[key] for key in keys}
    new_names: MutableSequence[str] = list(old2new_names.values())
    df = df.rename(columns=old2new_names)
    return df, new_names


class SeasonalEvolution:

    def __init__(self,
                 exp: Union[ExperienceManager, None] = None
                 ) -> None:
        self.exp = exp

    def plot_seasonal_evolution(self,
                                df: pd.DataFrame,
                                metrics: Tuple[str, ...] = ("bias", "ae", "n_bias", "n_ae"),
                                fontsize: int = 20,
                                figsize: Tuple[int, int] = (20, 15),
                                keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                                groupby: str = "month",
                                name: str = "Seasonal_evolution",
                                color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4"),
                                errorbar: Union[str, None] = None,
                                yerr: Union[bool, None] = False,
                                print_: bool = False
                                ) -> None:

        keys = ['_' + key.split('_')[-1] for key in keys]
        for metric in metrics:
            key2old_name = {"_AROME": f"{metric}_AROME",
                            "_D": f"{metric}_D",
                            "_nn": f"{metric}_nn",
                            "_int": f"{metric}_int",
                            "_A": f"{metric}_A",
                            }

            df, new_names = _old_names2new_names(df, keys, key2old_name)

            plot_evolution(df,
                           hue_names_to_plot=tuple(new_names),
                           y_label_name=metric,
                           fontsize=fontsize,
                           figsize=figsize,
                           groupby=groupby,
                           color=color,
                           errorbar=errorbar,
                           yerr=yerr,
                           print_=print_)
            if _sns:
                sns.set_style("ticks", {'axes.grid': True})

            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.xlabel("Month", fontsize=fontsize)
            plt.ylabel(METRICS2NAMES[metric], fontsize=fontsize)
            save_figure(f"Seasonal_evolution/{name}", exp=self.exp, svg=True)
            df = df.drop(columns=list(set(new_names)))

    def plot_seasonal_evolution_by_station(self,
                                           df: pd.DataFrame,
                                           metrics: Tuple[str, str, str, str] = ("bias", "ae", "n_bias", "n_ae"),
                                           keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                                           groupby: str = "month",
                                           fontsize: int = 15,
                                           figsize: Tuple[int, int] = (20, 15),
                                           name: str = "",
                                           color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4"),
                                           print_: bool = False
                                           ) -> None:
        keys = ['_' + key.split('_')[-1] for key in keys]
        for station in df["name"].unique():
            df_copy = df.copy(deep=True)
            for idx_metric, metric in enumerate(metrics):
                key2old_name = {"_AROME": f"{metric}_AROME",
                                "_D": f"{metric}_D",
                                "_nn": f"{metric}_nn",
                                "_int": f"{metric}_int",
                                "_A": f"{metric}_A",
                                }

                df_copy, new_names = _old_names2new_names(df_copy, keys, key2old_name)

                plot_evolution(df_copy[df_copy["name"] == station],
                               hue_names_to_plot=tuple(set(new_names)),
                               y_label_name=metric,
                               fontsize=fontsize,
                               figsize=figsize,
                               groupby=groupby,
                               color=color,
                               print_=print_)

                plt.title(station)
                save_figure(f"Seasonal_evolution_by_station/Seasonal_evolution_{station}_{name}", exp=self.exp)
                df_copy = df_copy.drop(columns=list(set(new_names)))


class Leadtime:

    def __init__(self,
                 exp: Union[ExperienceManager, None] = None
                 ) -> None:
        self.exp = exp

    def plot_lead_time(self,
                       df: pd.DataFrame,
                       metrics: Tuple[str, ...] = ("bias", "ae"),  # ("bias", "ae", "n_bias", "n_ae")
                       keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                       groupby: str = "lead_time",
                       fontsize: int = 15,
                       figsize: Tuple[int, int] = (20, 15),
                       name: str = "Lead_time",
                       color: Tuple[str, ...] = ("C1", "C0", "C2", "C3", "C4"),
                       print_: bool = False,
                       yerr: Union[bool, None] = False
                       ) -> None:
        keys = ['_' + key.split('_')[-1] for key in keys]
        for metric in metrics:
            key2old_name = {"_AROME": f"{metric}_AROME",
                            "_D": f"{metric}_D",
                            "_nn": f"{metric}_nn",
                            "_int": f"{metric}_int",
                            "_A": f"{metric}_A",
                            "_DA": f"{metric}_DA",
                            }

            df, new_names = _old_names2new_names(df, keys, key2old_name)

            plot_evolution(df,
                           hue_names_to_plot=tuple(set(new_names)),
                           y_label_name=metric,
                           fontsize=fontsize,
                           figsize=figsize,
                           groupby=groupby,
                           color=color,
                           yerr=yerr,
                           print_=print_)

            save_figure(f"Lead_time/{name}", exp=self.exp)
            df = df.drop(columns=list(set(new_names)))

    def plot_lead_time_by_station(self,
                                  df: pd.DataFrame,
                                  metrics: Tuple[str, str, str, str] = ("bias", "ae", "n_bias", "n_ae"),
                                  keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                                  groupby: str = "lead_time",
                                  fontsize: int = 15,
                                  figsize: Tuple[int, int] = (20, 15),
                                  color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4")
                                  ) -> None:
        keys = ['_' + key.split('_')[-1] for key in keys]
        for station in df["name"].unique():
            df_copy = df.copy(deep=True)
            for metric in metrics:
                key2old_name = {"_AROME": f"{metric}_AROME",
                                "_D": f"{metric}_D",
                                "_nn": f"{metric}_nn",
                                "_int": f"{metric}_int",
                                "_A": f"{metric}_A",
                                }

                df_copy, new_names = _old_names2new_names(df_copy, keys, key2old_name)

                plot_evolution(df_copy[df_copy["name"] == station],
                               hue_names_to_plot=tuple(set(new_names)),
                               y_label_name=metric,
                               fontsize=fontsize,
                               figsize=figsize,
                               groupby=groupby,
                               color=color)

                save_figure(f"Lead_time/Lead_time_{station}", exp=self.exp)
                df_copy = df_copy.drop(columns=list(set(new_names)))

    @pass_if_doesnt_have_seaborn_version()
    def plot_lead_time_shadow(self,
                              df: pd.DataFrame,
                              metrics: Tuple[str, ...] = ("bias", "ae", "n_bias", "n_ae"),
                              list_x: Tuple[str, ...] = ("lead_time",),
                              dict_keys: Dict[str, str] = {"_nn": "Neural Network", "_AROME": "AROME"},
                              showfliers: bool = False,
                              figsize: Tuple[int, int] = (15, 10),
                              name: str = "Boxplot_topo_carac",
                              order: Union[Tuple[str, ...], None] = ("$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$",
                                                                     "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"),
                              hue_order: Union[Tuple[str, ...], None] = ("$AROME_{forecast}$", "DEVINE",
                                                                         "Neural Network + DEVINE",
                                                                         "$AROME_{analysis}$"),
                              palette: Union[Tuple[str, ...], None] = ("C1", "C0", "C2", "C4"),
                              print_: bool = False,
                              errorbar: Union[str, Tuple[str, float], None] = None,
                              fontsize: float = 15
                              ) -> None:
        print("debug plot_lead_time_shadow")
        for idx, metric in enumerate(metrics):
            print(metric)
            old_names = list(dict_keys.keys())
            new_names = list(dict_keys.values())
            if idx > 0:
                df.drop(columns=new_names, inplace=True)
            new_columns = {f"{metric}{old_name}": new_name for (old_name, new_name) in zip(old_names, new_names)}
            df = df.rename(columns=new_columns)

            for x in list_x:
                print(x)
                if x not in df:
                    df[x] = getattr(df.index, x)
                df_to_plot = df[["name"] + [x] + new_names]

                if print_:
                    print(f"carac {x}, metric {metric}, models_names {tuple(new_names)}, nb_obs {len(df)}")
                df_melted = df_to_plot.melt(id_vars=["name", x],
                                            value_vars=tuple(new_names),
                                            var_name='Model',
                                            value_name=metric.capitalize())
                plt.figure(figsize=figsize)
                ax = plt.gca()
                sns.set_style("ticks", {'axes.grid': True})
                print("debug")
                print(errorbar)
                try:
                    sns.lineplot(data=df_melted, x=x, y=metric.capitalize(), hue="Model", errorbar=errorbar)
                    print("debug executed")
                    print("sns.lineplot(data=df_melted, x=x, y=metric.capitalize(), hue='Model', errorbar=errorbar)")
                except:
                    sns.lineplot(data=df_melted, x=x, y=metric.capitalize(), hue="Model")
                    print("debug executed")
                    print("sns.lineplot(data=df_melted, x=x, y=metric.capitalize(), hue='Model')")
                plt.xlabel(CARAC2NAME[x], fontsize=fontsize)
                plt.ylabel(METRICS2NAMES[metric], fontsize=fontsize)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)
                save_figure(f"LeadTimes/{metric}_{name}", exp=self.exp, svg=True)


def plot_boxplot_models(df: pd.DataFrame,
                        carac: str = "class_laplacian",
                        metric: str = "bias",
                        models_names: Tuple[str] = ("Neural Network", "AROME"),
                        showfliers: bool = False,
                        orient: str = "v",
                        figsize: Tuple[int, int] = (15, 12),
                        order: Union[Tuple[str], None] = ("$x \leq q_{25}$",
                                                          "$q_{25}<x \leq q_{50}$",
                                                          "$q_{50}<x \leq q_{75}$",
                                                          "$q_{75}<x$"),
                        hue_order: Union[Tuple[str], None] = ("$AROME_{forecast}$", "DEVINE",
                                                              "Neural Network + DEVINE", "$AROME_{analysis}$"),
                        palette: Union[Tuple[str], None] = ("C1", "C0", "C2", "C3"),
                        print_: bool = False,
                        fontsize: float = 20
                        ) -> None:
    if print_:
        print(f"carac {carac}, metric {metric}, models_names {models_names}, nb_obs {len(df)}")

    df_melted = df.melt(id_vars=["name", carac],
                        value_vars=models_names,
                        var_name='Model',
                        value_name=metric.capitalize())

    # e.g. replace "$x \leq q_{25}$" by "$Elevation \leq q_{25}$"
    order_with_name = []
    for value in order:
        new_name = value.replace("x", CLASS2CARAC[carac])
        df_melted.loc[df_melted[carac] == value, carac] = new_name
        order_with_name.append(new_name)

    plt.figure(figsize=figsize)
    ax = plt.gca()
    sns.boxplot(data=df_melted,
                y=metric.capitalize(),
                x=carac,
                hue='Model',
                orient=orient,
                showfliers=showfliers,
                order=order_with_name,
                hue_order=list(hue_order),
                palette=list(palette))
    plt.xlabel(CARAC2NAME[carac], fontsize=fontsize)
    plt.ylabel(METRICS2NAMES[metric], fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    sns.despine(trim=True, left=True)


class Boxplots:

    def __init__(self, exp: Union[ExperienceManager, None] = None
                 ) -> None:
        self.exp = exp

    @pass_if_doesnt_has_module()
    def plot_boxplot_topo_carac(self,
                                df: pd.DataFrame,
                                metrics: Tuple[str, ...] = ("bias", "ae", "n_bias", "n_ae"),
                                topo_carac: Tuple[str, ...] = (
                                        'mu', 'curvature', 'tpi_500', 'tpi_2000', 'laplacian', 'alti'),
                                dict_keys: Dict[str, str] = {"_nn": "Neural Network", "_AROME": "AROME"},
                                showfliers: bool = False,
                                figsize: Tuple[int, int] = (15, 10),
                                name: str = "Boxplot_topo_carac",
                                order: Union[Tuple[str], None] = ("$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$",
                                                                  "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"),
                                hue_order: Union[Tuple[str], None] = ("$AROME_{forecast}$", "DEVINE",
                                                                      "Neural Network + DEVINE", "$AROME_{analysis}$"),
                                palette: Union[Tuple[str], None] = ("C1", "C0", "C2", "C4"),
                                fontsize: float = 20,
                                print_: bool = False
                                ) -> None:

        for carac in topo_carac:
            assert f"class_{carac}" in df, f"class_{carac} should be in input DataFrame. " \
                                           f"Dataframe columns are {df.columns}"

        for idx, metric in enumerate(metrics):
            old_names = list(dict_keys.keys())
            new_names = list(dict_keys.values())
            if idx > 0:
                df.drop(columns=new_names, inplace=True)
            new_columns = {f"{metric}{old_name}": new_name for (old_name, new_name) in zip(old_names, new_names)}
            df = df.rename(columns=new_columns)
            for carac in topo_carac:
                df_to_plot = df[["name"] + [f"class_{carac}"] + new_names]
                plot_boxplot_models(df_to_plot,
                                    carac=f"class_{carac}",
                                    metric=metric,
                                    models_names=tuple(new_names),
                                    showfliers=showfliers,
                                    orient="v",
                                    figsize=figsize,
                                    order=order,
                                    hue_order=hue_order,
                                    palette=palette,
                                    print_=print_,
                                    fontsize=fontsize
                                    )

                save_figure(f"Boxplots/{name}", exp=self.exp, svg=True)
                try:
                    print("debug")
                    print(df_to_plot.groupby([f"class_{carac}"])["name"].nunique())
                except Exception as e1:
                    print("e1")
                    print(e1)
                    try:
                        print(df_to_plot.groupby([f"class_{carac}"])["name"].unique())
                    except Exception as e2:
                        print("e2")
                        print(e2)
                    pass


class WindRoses:
    def __init__(self, exp: Union[ExperienceManager, None] = None
                 ) -> None:
        self.exp = exp

    def plot_wind_direction_all(self,
                                df: pd.DataFrame,
                                keys: Tuple[str, ...] = ('UV_DIR_AROME',
                                                         'UV_DIR_D',
                                                         'UV_DIR_nn',
                                                         'UV_DIR_int',
                                                         'UV_DIR_A',
                                                         "UV_DIR_DA"),
                                metrics: Tuple[str, ...] = ("abs_bias_direction",),
                                name: str = "wind_direction_all",
                                cmap: matplotlib.colors.ListedColormap = cm_plt.viridis,
                                kind="bar",
                                print_: bool = True,
                                rmax: int = 13
                                ):

        keys = ['_' + key.split('_')[-1] for key in keys]
        for metric in metrics:
            if metric == "bias_direction":
                bins = np.arange(-100, 120, 20)
            else:
                bins = np.arange(0, 150, 30)

            for key in keys:
                if print_:
                    print(f"metric: {metric}, key: {key}, nb of obs {len(df)}")
                df[f'UV_DIR{key}'] = np.mod(df[f'UV_DIR{key}'], 360)
                plot_windrose(df,
                              var_name=f"{metric}{key}",
                              direction_name=f"UV_DIR{key}",
                              normed=True,
                              rmax=rmax,
                              kind=kind,
                              bins=bins,
                              cmap=cmap)
                plt.title(key)
                save_figure(f"{name}/wind_direction_all_{metric}_{key}", exp=self.exp, svg=True)

    def plot_wind_rose_for_observation(self, df_results, name="Wind_direction", cmap=cm_plt.get_cmap("plasma"), plot_by_station=False):
        # Plot wind rose for observations
        speed_df = pd.read_pickle(os.path.join(self.exp.path_to_current_experience, f"df_results_UV.pkl"))

        # Select speed
        df_results_speed = []
        for station in df_results["name"].unique():
            index_intersection = df_results[df_results["name"] == station].index.intersection(
                speed_df[speed_df["name"] == station].index)
            df_results_speed.append(
                speed_df[(speed_df["name"] == station) & (speed_df.index.isin(index_intersection))])
        df_results_speed = pd.concat(df_results_speed)

        # Plot windrose
        plot_windrose(df_results["UV_DIR_obs"].values,
                      var=df_results_speed["UV_obs"].values,
                      bins=np.array([1, 4, 6, 9, 12]),
                      normed=True,
                      rmax=13,
                      kind="bar",
                      cmap=cmap)
        if plot_by_station:
            save_figure(f"{name}/UV_obs_{station}", exp=self.exp, svg=True)
        else:
            save_figure(f"{name}/UV_obs", exp=self.exp, svg=True)

    def plot_wind_direction_by_station(self,
                                df: pd.DataFrame,
                                keys: Tuple[str, ...] = ('UV_DIR_AROME',
                                                         'UV_DIR_D',
                                                         'UV_DIR_nn',
                                                         'UV_DIR_int',
                                                         'UV_DIR_A'),
                                metrics: Tuple[str, ...] = ("abs_bias_direction",),
                                name: str = "wind_direction_all",
                                cmap: matplotlib.colors.ListedColormap = cm_plt.viridis,
                                kind="bar",
                                print_: bool = True,
                                ):

        for station in df["name"].unique():
            keys = ['_' + key.split('_')[-1] for key in keys]

            for metric in metrics:
                if metric == "bias_direction":
                    bins = np.arange(-100, 120, 20)
                else:
                    bins = np.arange(0, 150, 30)

                for key in keys:
                    if print_:
                        print(f"metric: {metric}, key: {key}, nb of obs {len(df)}")

                    plot_windrose(df[df["name"] == station],
                                  var_name=f"{metric}{key}",
                                  direction_name=f"UV_DIR{key}",
                                  normed=True,
                                  kind=kind,
                                  bins=bins,
                                  cmap=cmap)
                    plt.title(key)
                    save_figure(f"{name}/{station}_{metric}_{key}", exp=self.exp, svg=True)


class ALEPlot:
    def __init__(self, exp: Union[ExperienceManager, None] = None
                 ) -> None:
        self.exp = exp

    def plot_ale(self,
                 cm,
                 data_loader,
                 features,
                 bins,
                 monte_carlo=False,
                 rugplot_lim=None,
                 cmap="viridis",
                 marker='x',
                 markersize=1,
                 linewidth=1,
                 folder_name="ALE_speed",
                 ylim=(-2, 2),
                 exp=None,
                 use_std=True,
                 type_of_output="speed",
                 only_local_effects=False,
                 colors=None,
                 fontsize=25):

        df_inputs = data_loader.get_inputs(mode="test")
        cmap = plt.get_cmap(cmap, 3)

        if colors is None:
            colors = cmap(np.linspace(0, 1, 3))

        predictor = partial(cm.predict_multiple_batches,
                            model_version="last",
                            batch_size=128,
                            index_max=data_loader.get_length("test"),
                            output_shape=(data_loader.get_length("test"),),
                            force_build=False)

        for feature in features:
            print(f"Feature: {feature}")

            if feature in ["alti", "ZS", "tpi_500", "curvature", "mu", "laplacian"]:
                color = cmap(0)
            elif feature in ["Tair", "LWnet", "SWnet", "CC_cumul", "BLH"]:
                color = cmap(1)
            else:
                color = cmap(2)

            ale_plot(cm.model,
                     df_inputs,
                     [feature],
                     bins=bins,
                     monte_carlo=monte_carlo,
                     rugplot_lim=rugplot_lim,
                     data_loader=data_loader,
                     color=color,
                     marker=marker,
                     markersize=markersize,
                     predictor=predictor,
                     exp=exp,
                     use_std=use_std,
                     folder_name=folder_name,
                     type_of_output=type_of_output,
                     only_local_effects=only_local_effects,
                     linewidth=linewidth)
            if ylim:
                plt.ylim(ylim)
            ax = plt.gca()
            # ax.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.tight_layout()
            save_figure(f"{folder_name}/ale_{feature}", exp=self.exp, svg=True)

    def plot_ale_two_variables(self,
                               cm,
                               data_loader,
                               features,
                               bins,
                               monte_carlo=False,
                               rugplot_lim=None,
                               cmap="viridis",
                               marker='x',
                               markersize=1,
                               linewidth=1,
                               folder_name="ALE_speed",
                               ylim=(-2, 2),
                               exp=None,
                               use_std=None,
                               type_of_output="speed",
                               fontsize=25):
        df_inputs = data_loader.get_inputs(mode="test")
        cmap = plt.get_cmap(cmap, 4)
        colors = cmap(np.linspace(0, 1, len(features)))
        predictor = partial(cm.predict_multiple_batches,
                            model_version="last",
                            batch_size=128,
                            index_max=data_loader.get_length("test"),
                            output_shape=(data_loader.get_length("test"),),
                            force_build=False)

        for list_feature, color in zip(features, colors):
            print(f"Features: {list_feature}")
            ale_plot(cm.model,
                     df_inputs,
                     list_feature,
                     bins=bins,
                     monte_carlo=monte_carlo,
                     rugplot_lim=rugplot_lim,
                     data_loader=data_loader,
                     color=color,
                     marker=marker,
                     markersize=markersize,
                     predictor=predictor,
                     exp=exp,
                     use_std=use_std,
                     folder_name=folder_name,
                     type_of_output=type_of_output,
                     linewidth=linewidth)
            if ylim:
                plt.ylim(ylim)
            ax = plt.gca()
            # ax.tick_params(axis='both', which='major', labelsize=fontsize)
            plt.tight_layout()
            save_figure(f"{folder_name}/ale_two_variables_{list_feature[0]}_{list_feature[1]}", exp=self.exp, svg=True)


def qq_plot(obs, model, nb_point=10_000, marker="x", linestyle="-", markersize=5, color="C0", color_1_1="red",
            linewidth=2, ax=None):
    # quantiles
    percs = np.round(np.linspace(0, 100, nb_point), 2)
    qn_obs = np.percentile(obs, percs)
    qn_model = np.percentile(model, percs)

    # QQ-plot
    ax.plot(qn_obs, qn_model, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize,
            color=color)

    # 1-1 line
    x = np.linspace(np.min((qn_obs.min(),
                            qn_obs.min())),
                    np.max((qn_obs.max(),
                            qn_obs.max())))

    plt.plot(x, x, ls="--", color=color_1_1, label='_nolegend_')
    plt.xlim(-1, 45)
    plt.ylim(-1, 45)
    plt.axis("square")


class QQplot:

    def __init__(self, exp: Union[ExperienceManager, None] = None
                 ) -> None:
        self.exp = exp

    def qq_single(self,
                  df: pd.DataFrame,
                  figsize: Tuple[int, int] = (15, 10),
                  key_obs="UV_obs",
                  key_model="UV_nn",
                  color="C0",
                  marker="x",
                  linestyle="-",
                  markersize=5,
                  color_1_1="red",
                  linewidth=2,
                  fontsize=20,
                  ax=None
                  ) -> None:

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        obs = df[key_obs].values
        model = df[key_model].values
        qq_plot(obs=obs,
                model=model,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=markersize,
                color_1_1=color_1_1,
                linewidth=linewidth,
                ax=ax)

        plt.xlabel("Observed wind speed [$m\:s^{-1}$]", fontsize=fontsize)
        plt.ylabel("Modeled wind speed [$m\:s^{-1}$]", fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.tight_layout()

        save_figure(f"QQ_plot/qq_plot_single", exp=self.exp)

    def qq_double(self,
                  df0: pd.DataFrame,
                  df1: pd.DataFrame,
                  figsize: Tuple[int, int] = (15, 10),
                  key_obs="UV_obs",
                  key_model="UV_nn",
                  color0="C2",
                  color1="C9",
                  marker0="x",
                  marker1="d",
                  marker_AROME="d",
                  linestyle0="-",
                  linestyle1="-",
                  markersize=2,
                  color_1_1="red",
                  fontsize=25,
                  linewidth=2,
                  legend=("$L_{speed}$", "mse", "AROME"),
                  ax=None
                  ) -> None:

        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        obs = df0[key_obs].values
        model = df0[key_model].values
        qq_plot(obs=obs,
                model=model,
                color=color0,
                marker=marker0,
                linestyle=linestyle0,
                markersize=markersize,
                linewidth=linewidth,
                color_1_1=color_1_1,
                ax=ax)

        obs = df1[key_obs].values
        model = df1[key_model].values
        qq_plot(obs=obs,
                model=model,
                color=color1,
                marker=marker_AROME,
                linestyle=linestyle1,
                markersize=markersize,
                linewidth=linewidth,
                color_1_1=color_1_1,
                ax=ax)

        obs = df1[key_obs].values
        model = df1["UV_AROME"].values
        qq_plot(obs=obs,
                model=model,
                color="C1",
                marker=marker1,
                linestyle=linestyle1,
                markersize=markersize,
                linewidth=linewidth,
                color_1_1=color_1_1,
                ax=ax)

        plt.grid(visible=True)
        plt.xlabel("Observed wind speed [$m\:s^{-1}$]", fontsize=fontsize)
        plt.ylabel("Modeled wind speed [$m\:s^{-1}$]", fontsize=fontsize)
        plt.legend(legend, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.tight_layout()
        save_figure(f"QQ_plot/qq_plot_double", exp=self.exp)


class VizualizationResults(Boxplots,
                           Leadtime,
                           SeasonalEvolution,
                           ModelVersusObsPlots,
                           StaticPlots,
                           WindRoses,
                           ALEPlot,
                           QQplot):

    def __init__(self, exp: Union[ExperienceManager, None] = None
                 ) -> None:
        super().__init__(exp)
