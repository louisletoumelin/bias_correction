import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LightSource

from bias_correction.train.wind_utils import wind2comp, comp2speed

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


def plot_quiver(x, y, u, v, uv,
                cmap="coolwarm", norm=None, linewidths=1, scale=1/0.004, axis_equal=True, edgecolor=None):
    ax = plt.gca()
    ax.quiver(x, y, u, v, uv, cmap=cmap, norm=norm, linewidths=linewidths, scale=scale, edgecolor=edgecolor)
    if axis_equal:
        plt.axis("equal")


def get_two_slope_norm(vmin=0, vcenter=5, vmax=10):
    return colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)


class VisuMap:

    def __init__(self, exp, d0, d, d1, station, config):
        super().__init__(exp)
        self.d0 = d0
        self.d = d
        self.d1 = d1
        self.station = station
        self.config = config

    def get_map_prediction(self, time_series, data_loader, cm):
        assert self.config["type_of_output"] == "map"

        print("\nModel for map prediction:")
        print(self.config["global_architecture"])

        inputs = time_series[self.config["input_variables"] + ["name"]]

        # Filter time
        filter_time = (inputs.index >= self.d0) & (inputs.index <= self.d1)
        inputs = inputs[filter_time]

        # Filter station
        filter_station = inputs["name"] == self.station
        inputs = inputs[filter_station]

        # Get names
        names = inputs["name"]

        # Select variables
        inputs = inputs[self.config["input_variables"]]

        # Get id of current time_step
        id_t = np.squeeze(np.where(inputs.index == self.d))

        # Prepare data for predictions
        inputs_test = data_loader.get_tf_zipped_inputs(inputs=inputs, names=names).batch(len(inputs))
        results_test = cm.predict_with_batch(inputs_test)

        # Prepare maps
        uv = results_test[0][:, :, :, 0]
        dir = results_test[1][:, :, :, 0]
        u, v = wind2comp(uv, dir, unit_direction="degree")
        uv = uv[id_t, :, :]
        dir = dir[id_t, :, :]
        u = u[id_t, :, :]
        v = v[id_t, :, :]

        return uv, dir, u, v

    def get_topos(self, data_loader, initial_length=280, y_offset=39, x_offset=34, preprocess=True):
        x = data_loader.dict_topos[self.station]['x']
        y = data_loader.dict_topos[self.station]['y']
        topos = data_loader.dict_topos[self.station]['data'][:, :, 0]

        if preprocess:
            y_offset_left = initial_length // 2 - y_offset
            y_offset_right = initial_length // 2 + y_offset + 1
            x_offset_left = initial_length // 2 - x_offset
            x_offset_right = initial_length // 2 + x_offset + 1
            x = x[x_offset_left:x_offset_right]
            y = y[y_offset_left:y_offset_right]

            initial_length = initial_length // 2
            y_offset_left = initial_length // 2 - y_offset
            y_offset_right = initial_length // 2 + y_offset + 1
            x_offset_left = initial_length // 2 - x_offset
            x_offset_right = initial_length // 2 + x_offset + 1
            topos = topos[y_offset_left:y_offset_right, x_offset_left:x_offset_right]

        return x, y, topos

    def get_arome_wind(self, time_series):
        inputs = time_series[self.config["input_variables"] + ["name"]]

        # Filter time
        filter_time = inputs.index == self.d
        inputs = inputs[filter_time]

        # Filter station
        filter_station = inputs["name"] == self.station
        inputs = inputs[filter_station]

        # Select variables
        inputs = inputs[self.config["input_variables"]]

        # Wind fields
        uv = inputs["Wind"].values[0]
        dir = inputs["Wind_DIR"].values[0]
        u, v = wind2comp(uv, dir, unit_direction="degree")

        return uv, dir, u, v

    def get_arome_x_y(self, data_loader):

        # Load stations
        stations = data_loader.get_stations()

        # Filter station
        filter_station = stations["name"] == self.station
        stations = stations[filter_station]

        # Get x and y coordinates
        x = stations['X_AROME_NN_0'].values[0]
        y = stations['Y_AROME_NN_0'].values[0]

        return x, y

    def get_observation_x_y(self, data_loader):

        # Load stations
        stations = data_loader.get_stations()

        # Filter station
        filter_station = stations["name"] == self.station
        stations = stations[filter_station]

        # x and y coordinates
        x = stations['X'].values[0]
        y = stations['Y'].values[0]

        return x, y

    def get_observation_wind(self, time_series):

        # Select variables
        inputs = time_series[['U_obs', 'V_obs'] + ["name"]]

        # Filter time
        filter_time = inputs.index == self.d
        inputs = inputs[filter_time]

        # Filter station
        filter_station = inputs["name"] == self.station
        inputs = inputs[filter_station]

        # Select variables
        inputs = inputs[['U_obs', 'V_obs']]
        inputs = inputs[filter_station]

        # Wind fields
        u = inputs['U_obs'].values[0]
        v = inputs['V_obs'].values[0]
        uv = comp2speed(u, v)

        return uv, u, v

    def plot_quiver(self, time_series, data_loader, cm,
                    fraction=0.01, vert_exag=10_000, dx=30, dy=30, vmin=0.48, vmax=0.51,
                    color="dimgrey", levels=15, alpha=0.75, plot_levels=True, fontsize=10,
                    cmap="coolwarm", norm=None, linewidths=1, scale=1 / 0.004, axis_equal=True,
                    edgecolor_arome="C0", edgecolor_observation="black"):

        x, y, topos = self.get_topos(data_loader)
        plt.figure()

        # Topography
        plot_hillshade(x, y, topos, fraction=fraction, vert_exag=vert_exag, dx=dx, dy=dy, vmin=vmin, vmax=vmax)
        plot_contour(x, y, topos, color=color, levels=levels, alpha=alpha, plot_levels=plot_levels, fontsize=fontsize)

        # DEVINE
        uv, _, u, v = self.get_map_prediction(time_series, data_loader, cm)
        plot_quiver(x, y, u, v, uv, cmap=cmap, norm=norm, linewidths=linewidths, scale=scale, axis_equal=axis_equal)

        # AROME
        uv, _, u, v = self.get_arome_wind(time_series)
        x, y = self.get_arome_x_y(data_loader)
        plot_quiver(x, y, u, v, uv, cmap=cmap, norm=norm, linewidths=linewidths,
                    scale=scale, axis_equal=axis_equal, edgecolor=edgecolor_arome)

        # Observation
        x, y = self.get_observation_x_y(data_loader)
        uv, _, u, v = self.get_arome_wind(time_series)
        plot_quiver(x, y, u, v, uv, cmap=cmap, norm=norm, linewidths=linewidths,
                    scale=scale, axis_equal=axis_equal, edgecolor=edgecolor_observation)
