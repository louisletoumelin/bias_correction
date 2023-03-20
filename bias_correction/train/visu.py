import numpy as np
import pandas as pd
import matplotlib
from matplotlib import cm
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import os
import uuid
from typing import Union, Tuple, Dict, MutableSequence

from bias_correction.utils_bc.decorators import pass_if_doesnt_has_module
from bias_correction.train.utils import create_folder_if_doesnt_exist
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.windrose import plot_windrose

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
    print("debug")
    print(fig)
    print(save_path)
    print(f"{name_figure}_{uuid_str}.{format_}")
    print(fig.savefig(save_path + f"{name_figure}_{uuid_str}.{format_}"))
    fig.savefig(save_path + f"{name_figure}_{uuid_str}.{format_}")
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
    print("debug")
    print(obs[:5])
    print(model[:5])
    # xlim and ylim
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)

    # Text
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
                     print_: bool = False
                     ) -> None:
        current_variable = self.exp.config['current_variable']
        key_obs = f"{current_variable}_obs"
        print("debug")
        print(df.columns)
        for idx, key in enumerate(keys):
            print("debug key")
            print(key)
            print(key_obs)
            print(current_variable)
            fig = plot_single_1_1(df, key, key_obs, current_variable,
                                  s=s, figsize=figsize, color=color[idx], print_=print_)
            save_figure(f"Model_vs_obs/{name}_{key}", exp=self.exp, fig=fig)

    def plot_1_1_by_station(self,
                            df: pd.DataFrame,
                            keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                            s: int = 1,
                            name: str = "",
                            color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4"),
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
                   print_: bool = False
                   ) -> None:
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
    df.groupby(index_groupby).mean()[list(hue_names_to_plot)].plot(color=dict_color, ax=ax)
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
                                fontsize: int = 15,
                                figsize: Tuple[int, int] = (20, 15),
                                keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                                groupby: str = "month",
                                name: str = "Seasonal_evolution",
                                color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4"),
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
                           print_=print_)

            save_figure(f"Seasonal_evolution/{name}", exp=self.exp)
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
                       metrics: Tuple[str, ...] = ("bias", "ae", "n_bias", "n_ae"),
                       keys: Tuple[str, ...] = ("UV_nn", "UV_AROME"),
                       groupby: str = "lead_time",
                       fontsize: int = 15,
                       figsize: Tuple[int, int] = (20, 15),
                       name: str = "Lead_time",
                       color: Tuple[str] = ("C1", "C0", "C2", "C3", "C4"),
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
                           hue_names_to_plot=tuple(set(new_names)),
                           y_label_name=metric,
                           fontsize=fontsize,
                           figsize=figsize,
                           groupby=groupby,
                           color=color,
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


def plot_boxplot_models(df: pd.DataFrame,
                        carac: str = "class_laplacian",
                        metric: str = "bias",
                        models_names: Tuple[str] = ("Neural Network", "AROME"),
                        showfliers: bool = False,
                        orient: str = "v",
                        figsize: Tuple[int, int] = (15, 12),
                        order: Union[Tuple[str], None] = ("$x \leq q_{25}$", "$q_{25}<x \leq q_{50}$",
                                                          "$q_{50}<x \leq q_{75}$", "$q_{75}<x$"),
                        hue_order: Union[Tuple[str], None] = ("$AROME_{forecast}$", "DEVINE",
                                                              "Neural Network + DEVINE", "$AROME_{analysis}$"),
                        palette: Union[Tuple[str], None] = ("C1", "C0", "C2", "C3"),
                        print_: bool = False
                        ) -> None:
    if print_:
        print(f"carac {carac}, metric {metric}, models_names {models_names}, nb_obs {len(df)}")
    df_melted = df.melt(id_vars=["name", carac],
                        value_vars=models_names,
                        var_name='Model',
                        value_name=metric.capitalize())

    plt.figure(figsize=figsize)

    sns.boxplot(data=df_melted,
                y=metric.capitalize(),
                x=carac,
                hue='Model',
                orient=orient,
                showfliers=showfliers,
                order=list(order),
                hue_order=list(hue_order),
                palette=list(palette))

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
                                palette: Union[Tuple[str], None] = ("C1", "C0", "C2", "C3"),
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
                                    print_=print_
                                    )

                save_figure(f"Boxplots/{name}", exp=self.exp)


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
                                                         'UV_DIR_A'),
                                metrics: Tuple[str, ...] = ("abs_bias_direction",),
                                name: str = "wind_direction_all",
                                cmap: matplotlib.colors.ListedColormap = cm.viridis,
                                kind="bar",
                                print_: bool = True
                                ):

        keys = ['_' + key.split('_')[-1] for key in keys]
        for metric in metrics:
            if metric == "bias_direction":
                bins = np.arange(-100, 120, 20)
            else:
                bins = np.arange(0, 150, 20)

            for key in keys:
                if print_:
                    print(f"metric: {metric}, key: {key}, nb of obs {len(df)}")
                print("debug")
                print(df.columns)
                plot_windrose(df,
                              var_name=f"{metric}{key}",
                              direction_name=f"UV_DIR{key}",
                              kind=kind,
                              bins=bins,
                              cmap=cmap)
                save_figure(f"Wind_direction/{name}_{metric}_{key}", exp=self.exp)


class VizualizationResults(Boxplots, Leadtime, SeasonalEvolution, ModelVersusObsPlots, StaticPlots, WindRoses):

    def __init__(self, exp: Union[ExperienceManager, None] = None
                 ) -> None:
        super().__init__(exp)
