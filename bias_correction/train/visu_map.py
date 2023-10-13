import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LightSource
from tensorflow.keras.models import Model

from bias_correction.train.wind_utils import wind2comp, comp2speed, comp2dir
from bias_correction.train.visu import save_figure
from bias_correction.utils_bc.plot_utils import MidPointNorm

try:
    import seaborn as sns

    _sns = True
except ModuleNotFoundError:
    _sns = False


def plot_hillshade(x, y, z, fraction=0.01, vert_exag=10_000, dx=30, dy=30, vmin=0.48, vmax=0.51):
    ls = LightSource(azdeg=270, altdeg=45)
    plt.contourf(x,
                 y,
                 ls.hillshade(z,
                              fraction=fraction,
                              vert_exag=vert_exag,
                              dx=dx,
                              dy=dy),
                 cmap=plt.cm.gray,
                 vmin=vmin,
                 vmax=vmax)


def plot_contour(x, y, z, color="dimgrey", levels=15, alpha=0.75, plot_levels=True, fontsize=10):
    CS = plt.contour(x,
                     y,
                     z,
                     colors=color,
                     levels=levels,
                     alpha=alpha)
    ax = plt.gca()
    if plot_levels:
        ax.clabel(CS, CS.levels, fmt="%d", inline=True, fontsize=fontsize)


def plot_quiver(x, y, u, v, uv, axis_equal=True, **kwargs):
    ax = plt.gca()
    fig = ax.quiver(x, y, u, v, uv, **kwargs)
    if axis_equal:
        plt.axis("equal")
    return fig


def get_two_slope_norm(vmin=0, vcenter=5, vmax=10):
    return colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


class VisuMap:

    def __init__(self, exp, d, station, data_loader, model, config):
        self.exp = exp
        self.d = d
        self.station = station
        self.data_loader = data_loader
        self.model = model
        self.config = config

    def get_intermediate_outputs(self, cm):
        assert self.config["type_of_output"] == "map"

        print("\nModel for map prediction:")
        print("Intermediate outputs")
        inputs = self.data_loader.get_inputs("test")[self.config["input_variables"]]
        names = self.data_loader.get_names("test")

        self.config["standardize"] = True
        self.data_loader.config["standardize"] = True
        cm.config["standardize"] = True
        self.config["global_architecture"] = "double_ann"
        self.data_loader.config["global_architecture"] = "double_ann"
        cm.config["global_architecture"] = "double_ann"

        # Filter time
        filter_time = inputs.index == self.d
        inputs = inputs[filter_time]
        names = names[filter_time]

        # Filter station
        filter_station = names == self.station
        inputs = inputs[filter_station]
        names = names[filter_station]

        # Get id of current time_step
        id_t = np.squeeze(np.where(inputs.index == self.d))
        print("inputs")
        print(inputs)
        print("time")
        print(inputs.index.values[id_t])

        # Prepare data for predictions
        inputs_test = self.data_loader.get_tf_zipped_inputs(inputs=inputs, names=names).batch(len(inputs))

        # Predict speed
        cm.build_model_with_strategy(print_=False)
        cm.load_weights()
        new_model = Model(inputs=cm.model.input, outputs=(cm.model.get_layer("Add_dense_output_speed_ann").output,))
        new_model.compile(loss=cm.get_loss(), optimizer=cm.get_optimizer(), metrics=cm.get_training_metrics())
        cm.model = new_model
        uv = cm.predict_single_bath(inputs_test, model_version=None, force_build=False, print_=False)

        # Predict dir
        cm.build_model_with_strategy(print_=False)
        cm.load_weights()
        new_model = Model(inputs=cm.model.input, outputs=(cm.model.get_layer("Add_dense_output_dir_ann").output,))
        new_model.compile(loss=cm.get_loss(), optimizer=cm.get_optimizer(), metrics=cm.get_training_metrics())
        cm.model = new_model
        dir = cm.predict_single_bath(inputs_test, model_version=None, force_build=False, print_=False)

        u, v = wind2comp(uv, dir, unit_direction="degree")

        return uv, dir, u, v

    def get_map_prediction(self, cm, path_to_last_weights=None):
        assert self.config["type_of_output"] == "map"

        print("\nModel for map prediction:")
        print(self.model)
        inputs = self.data_loader.get_inputs("test")[self.config["input_variables"]]
        names = self.data_loader.get_names("test")

        # Filter time
        filter_time = inputs.index == self.d
        inputs = inputs[filter_time]
        names = names[filter_time]

        # Filter station
        filter_station = names == self.station
        inputs = inputs[filter_station]
        names = names[filter_station]

        # Get id of current time_step
        id_t = np.squeeze(np.where(inputs.index == self.d))
        print("inputs")
        print(inputs)
        print("time")
        print(inputs.index.values[id_t])

        # Select variables
        if self.model == "devine_only":
            inputs = inputs[["Wind", "Wind_DIR"]]
        else:
            inputs = inputs[self.config["input_variables"]]

        # Prepare data for predictions
        inputs_test = self.data_loader.get_tf_zipped_inputs(inputs=inputs, names=names).batch(len(inputs))

        # Build model
        cm.build_model_with_strategy(print_=False)

        if path_to_last_weights:
            print("\ndebug")
            print("tmp_load_modified_weights")
            print(path_to_last_weights)
            cm.load_weights(path_to_last_weights)

        # Load weights
        if self.model == "double_ann":
            cm.load_weights()

        # Predict
        results_test = cm.predict_single_bath(inputs_test, model_version=None, force_build=False, print_=False)

        # Prepare maps
        uv = results_test[0][:, :, :, 0]
        dir = results_test[1][:, :, :, 0]
        u, v = wind2comp(uv, dir, unit_direction="degree")
        uv = uv[id_t, :, :]
        dir = dir[id_t, :, :]
        u = u[id_t, :, :]
        v = v[id_t, :, :]

        return uv, dir, u, v

    def get_topos(self, initial_length=140, y_offset=39, x_offset=34, preprocess=True):
        dict_topos = self.data_loader.loader.load_dict("topos", get_x_y=True)
        x = dict_topos[self.station]['x']
        y = dict_topos[self.station]['y']
        topos = dict_topos[self.station]['data'][:, :, 0]

        if preprocess:
            y_offset_left = initial_length // 2 - y_offset
            y_offset_right = initial_length // 2 + y_offset + 1
            x_offset_left = initial_length // 2 - x_offset
            x_offset_right = initial_length // 2 + x_offset + 1
            x = x[x_offset_left:x_offset_right]
            y = y[y_offset_left:y_offset_right]
            topos = topos[y_offset_left:y_offset_right, x_offset_left:x_offset_right]

        return x, y, topos

    def get_arome_wind(self):
        inputs = self.data_loader.get_inputs("test")
        names = self.data_loader.get_names("test")

        # Filter time
        filter_time = inputs.index == self.d
        inputs = inputs[filter_time]
        names = names[filter_time]

        # Filter station
        filter_station = names == self.station
        inputs = inputs[filter_station]

        # Select variables
        inputs = inputs[self.config["input_variables"]]

        print("time AROME")
        print(inputs.index.values)

        # Wind fields
        uv = inputs["Wind"].values[0]
        dir = inputs["Wind_DIR"].values[0]
        u, v = wind2comp(uv, dir, unit_direction="degree")

        return uv, dir, u, v

    def get_arome_analysis(self):
        inputs = self.data_loader.predicted_A

        # Filter time
        filter_time = inputs.index == self.d
        inputs = inputs[filter_time]

        # Filter station
        filter_station = inputs["name"] == self.station
        inputs = inputs[filter_station]

        print("time AROME analysis")
        print(inputs.index.values)

        # Wind fields
        uv = inputs["UV_A"].values[0]
        dir = inputs["UV_DIR_A"].values[0]
        u, v = wind2comp(uv, dir, unit_direction="degree")

        return uv, dir, u, v

    def get_arome_x_y(self):

        # Load stations
        stations = self.data_loader.get_stations()

        # Filter station
        filter_station = stations["name"] == self.station
        stations = stations[filter_station]

        # Get x and y coordinates
        x = stations['X_AROME_NN_0'].values[0]
        y = stations['Y_AROME_NN_0'].values[0]

        return x, y

    def get_observation_x_y(self):

        # Load stations
        stations = self.data_loader.get_stations()

        # Filter station
        filter_station = stations["name"] == self.station
        stations = stations[filter_station]

        # x and y coordinates
        x = stations['X'].values[0]
        y = stations['Y'].values[0]

        return x, y

    def get_observation_wind(self):

        # Select variables
        inputs = self.data_loader.loader.load_time_series_pkl()[['vw10m(m/s)', 'winddir(deg)'] + ["name"]]

        # Filter time
        filter_time = inputs.index == self.d
        inputs = inputs[filter_time]

        # Filter station
        filter_station = inputs["name"] == self.station
        inputs = inputs[filter_station]

        print("time observations")
        print(inputs.index.values)

        # Wind fields
        uv = inputs["vw10m(m/s)"].values[0]
        dir = inputs["winddir(deg)"].values[0]
        u, v = wind2comp(uv, dir, unit_direction="degree")

        return uv, dir, u, v

    def plot_quiver(self, cm,
                    fraction=0.01, vert_exag=10_000, dx=30, dy=30, vmin=0.48, vmax=0.51,
                    color="dimgrey", levels=15, alpha=0.75, plot_levels=True, fontsize=20, edgecolor="C0",
                    cmap="coolwarm", norm=None, linewidths=1, scale=1 / 0.004, axis_equal=True, nb_arrows_to_skip=1,
                    edgecolor_arome="C1", figsize=(15, 10), edgecolor_observation="black", width=0.001, name: str="",
                    path_to_last_weights: str = None):
        nb_a = nb_arrows_to_skip
        x, y, topos = self.get_topos()
        plt.figure(figsize=figsize)

        # Topography
        plot_hillshade(x, y, topos, fraction=fraction, vert_exag=vert_exag, dx=dx, dy=dy, vmin=vmin, vmax=vmax)
        plot_contour(x, y, topos, color=color, levels=levels, alpha=alpha, plot_levels=plot_levels, fontsize=fontsize)

        uv, dir, u, v = self.get_arome_wind()
        print("debug uv")
        print(uv)
        if norm is None:
            norm = MidPointNorm(midpoint=uv, vmin=0, vmax=10)

        # DEVINE
        uv, _, u, v = self.get_map_prediction(cm, path_to_last_weights=path_to_last_weights)
        x, y = np.meshgrid(x, y)
        u = np.where(uv < 0, np.nan, u)
        v = np.where(uv < 0, np.nan, v)
        uv = np.where(uv < 0, np.nan, uv)
        normalized_u = u/uv
        normalized_v = v/uv
        fig = plot_quiver(x[::nb_a, ::nb_a],
                          y[::nb_a, ::nb_a],
                          normalized_u[::nb_a, ::nb_a],
                          normalized_v[::nb_a, ::nb_a],
                          uv[::nb_a, ::nb_a],
                          cmap=cmap,
                          norm=norm,
                          linewidths=linewidths/3,
                          scale=scale,
                          axis_equal=axis_equal,
                          edgecolor=edgecolor,
                          width=width)
        cb = plt.colorbar(fig)
        cb.ax.tick_params(axis='both', which='major', labelsize=fontsize)
        print(f"{self.model}")
        print(uv[79//2, 69//2])
        print("uv max")
        print(np.nanmax(uv))
        print("uv min")
        print(np.nanmin(uv))
        dir = comp2dir(u, v)
        print(dir[79//2, 69//2])

        # AROME
        uv, dir, u, v = self.get_arome_wind()
        print("uv AROME")
        print(uv)
        print(dir)
        u = np.where(uv < 0, np.nan, u)
        v = np.where(uv < 0, np.nan, v)
        uv = np.where(uv < 0, np.nan, uv)
        print("uv AROME debug")
        print(uv)
        x, y = self.get_arome_x_y()
        normalized_u = u/uv
        normalized_v = v/uv
        plot_quiver(x,
                    y,
                    normalized_u,
                    normalized_v,
                    uv,
                    cmap=cmap,
                    norm=norm,
                    linewidths=linewidths,
                    scale=scale/2,
                    width=2*width,
                    axis_equal=axis_equal,
                    edgecolor=edgecolor_arome)

        # AROME analysis
        uv, dir, u, v = self.get_arome_analysis()
        print("uv AROME analysis")
        print(uv)
        print(dir)
        u = np.where(uv < 0, np.nan, u)
        v = np.where(uv < 0, np.nan, v)
        uv = np.where(uv < 0, np.nan, uv)
        x, y = self.get_arome_x_y()
        normalized_u = u/uv
        normalized_v = v/uv
        plot_quiver(x,
                    y+200,
                    normalized_u,
                    normalized_v,
                    uv,
                    cmap=cmap,
                    norm=norm,
                    linewidths=linewidths,
                    scale=scale/2,
                    width=2*width,
                    axis_equal=axis_equal,
                    edgecolor="C4")

        # Intermediate outputs
        if self.model == "double_ann":
            uv, dir, u, v = self.get_intermediate_outputs(cm)
            print("uv intermediate outputs")
            print(np.squeeze(uv))
            print(np.squeeze(dir))
            x, y = self.get_arome_x_y()
            normalized_u = np.squeeze(u/uv)
            normalized_v = np.squeeze(v/uv)
            plot_quiver(x,
                        y-250,
                        normalized_u,
                        normalized_v,
                        uv,
                        cmap=cmap,
                        norm=norm,
                        linewidths=linewidths,
                        scale=scale/2,
                        width=2*width,
                        axis_equal=axis_equal,
                        edgecolor="C3")

        # Observation
        x, y = self.get_observation_x_y()
        uv, dir, u, v = self.get_observation_wind()
        print("uv obs")
        print(uv)
        print(dir)
        u = np.where(uv < 0, np.nan, u)
        v = np.where(uv < 0, np.nan, v)
        uv = np.where(uv < 0, np.nan, uv)
        normalized_u = u/uv
        normalized_v = v/uv
        plot_quiver(x,
                    y,
                    normalized_u,
                    normalized_v,
                    uv,
                    cmap=cmap,
                    norm=norm,
                    linewidths=linewidths,
                    scale=scale/2,
                    width=2*width,
                    axis_equal=axis_equal,
                    edgecolor=edgecolor_observation)
        plt.title(self.model)
        save_figure(f"Map_wind_fields/{self.model}_{name}", exp=self.exp, svg=True)
