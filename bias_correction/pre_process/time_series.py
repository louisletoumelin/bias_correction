import numpy as np
import pandas as pd
import xarray as xr

import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from downscale.operators.wind_utils import Wind_utils
from downscale.operators.interpolation import Interpolation


class TimeSeries(Interpolation):

    """Create file with time series data"""

    def __init__(self, time_series=None, stations=None, interpolated=False, config={}):
        super().__init__()
        self.time_series = time_series
        self.stations = stations
        self.config = config
        self.interpolated = interpolated

    def keep_minimal_variables(self):
        variables_to_return = ['date', 'name', 'T2m(degC)', 'vw10m(m/s)', 'winddir(deg)', 'HTN(cm)']
        if "qc" in self.time_series.columns:
            variables_to_return.append("qc")
        return self.time_series[variables_to_return]

    def _add_AROME_variable_station(self, station, str_x, str_y, variables, filter_time, nwp):

        # Filters
        filter_station_ts = self.time_series["name"] == station
        all_filters_ts = filter_time & filter_station_ts
        filter_station_s = self.stations["name"] == station

        # Coordinates
        xx = np.intp(self.stations[str_x][filter_station_s].values[0])
        yy = np.intp(self.stations[str_y][filter_station_s].values[0])

        # Select data
        nwp_station = nwp[variables].isel(xx=xx, yy=yy).to_dataframe()

        # Index intersection
        nwp_station.index = pd.to_datetime(nwp_station.index)
        index_intersection = nwp_station.index.intersection(self.time_series[variables][all_filters_ts].index)
        nwp_station = nwp_station[nwp_station.index.isin(index_intersection)]

        # Filter intersection
        filter_intersection = self.time_series.index.isin(index_intersection)
        all_filters = filter_time & filter_station_ts & filter_intersection

        self.time_series.loc[all_filters, variables] = nwp_station[variables].values

    def select_date(self):

        date_min = self.time_series.index > '2017-8-1'
        date_max = self.time_series.index < '2020-9-30'   # '2019-3-1'
        self.time_series = self.time_series[date_min & date_max]

    def interpolate_nwp(self, nwp):
        return self.interpolate_wind_grid_xarray(nwp,
                                                 interp=self.config["interp"],
                                                 method=self.config["method"],
                                                 verbose=self.config["verbose"])

    def add_arome_variables(self):

        if self.config["network"] == "local":

            print("We don't add AROME variable to time_series file because of memory issues when AROME is interpolated")

        elif self.config["network"] == "labia":

            assert self.config.get("variables_nwp") is not None
            variables = self.config["variables_nwp"]
            str_interpolated = "_interpolated" if self.interpolated else ""

            # Initialization
            for variable in variables:
                self.time_series[variable] = np.nan

            self.time_series.index = pd.to_datetime(self.time_series["date"])
            str_x = f"X_index_AROME_analysis_NN_0{str_interpolated}_ref_AROME_analysis{str_interpolated}"
            str_y = f"Y_index_AROME_analysis_NN_0{str_interpolated}_ref_AROME_analysis{str_interpolated}"

            for country in ["france", "swiss", "pyr", "corse"]:
                for file in os.listdir(self.config[f"path_nwp_{country}"]):

                    nwp = xr.open_dataset(self.config[f"path_nwp_{country}"] + file)
                    if self.interpolated:
                        nwp = self.interpolate_nwp(nwp)

                    filter_time = self.time_series.index.isin(nwp.time.values)
                    for idx, station in enumerate(self.stations["name"][self.stations["country"] == country]):
                        if idx == 0:
                            logger.info(self.stations.head())
                            logger.info(self.stations.tail())
                            logger.info(self.stations.columns)

                        logger.info(country, file, station)
                        self._add_AROME_variable_station(station, str_x, str_y, variables, filter_time, nwp)

            """
            for country, country_name in zip(["alp", "swiss"], ["france", "switzerland"]):
                for file in os.listdir():
                # Open all files
                try:
                    nwp = xr.open_mfdataset(self.config[f"path_nwp_{country}"] + "AROME*", parallel=True)
                except ModuleNotFoundError:
                    nwp = xr.open_mfdataset(self.config[f"path_nwp_{country}"] + "AROME*", parallel=False)

                filter_time = self.time_series.index.isin(nwp.time.values)

                for station in self.stations["name"][self.stations["country"] == country_name]:

                    self._add_AROME_variable_station(station, str_x, str_y, variables, filter_time, nwp)
            """

    def compute_u_and_v(self):
        U_obs, V_obs = self.horizontal_wind_component(self.time_series["vw10m(m/s)"].values,
                                                      self.time_series["winddir(deg)"].values)
        self.time_series["U_obs"] = U_obs
        self.time_series["V_obs"] = V_obs
        try:
            U_AROME, V_AROME = self.horizontal_wind_component(self.time_series["Wind"].values,
                                                              self.time_series["Wind_DIR"].values)
            self.time_series["U_AROME"] = U_AROME
            self.time_series["V_AROME"] = V_AROME
        except KeyError:
            print("time_series does not contain Wind or Wind_DIR")

    def _remove_variable_not_downcasted(self):
        variables_not_downscasted = ["date", "name", "last_flagged_speed", "last_flagged_direction",
                                     "last_unflagged_direction", "qc_2_speed", "qc_3_speed", "qc_3_direction",
                                     "preferred_direction_during_sequence", "qc_5_speed",
                                     "cardinal", "preferred_direction"]
        for variable in variables_not_downscasted:
            try:
                self.config["variables_time_series"].remove(variable)
            except ValueError:
                pass

    def change_dtype_time_series(self):
        if self.config["network"] == "local":
            print("Not changing dtype since AROME variable are not in time_series")
        elif self.config["network"] == "labia":
            self.time_series = self.time_series[self.config["variables_time_series"]]
            self._remove_variable_not_downcasted()
            variables = self.config["variables_time_series"]
            self.time_series[variables] = self.time_series[variables].astype(np.float32)

    def save_to_csv(self, name=None):
        if name is None:
            name = ""
        self.time_series.to_csv(self.config["path_time_series_pre_processed"]+f"time_series_bc{name}.csv")
        print(f"Saved {self.config['path_time_series_pre_processed']+f'time_series_bc{name}.csv'}")

    def save_to_pickle(self, name=None):
        interp_str = "_interpolated" if self.interpolated else ""
        if name is None:
            name = ""
        self.time_series.to_pickle(self.config["path_time_series_pre_processed"]+f"time_series_bc{interp_str}{name}.pkl")
        print(f"Saved {self.config['path_time_series_pre_processed']+f'time_series_bc{interp_str}{name}.pkl'}")



"""
config = {}
str_x = f"X_index_AROME_NN_0_ref_AROME"
str_y = f"Y_index_AROME_NN_0_ref_AROME"
config["variables_nwp"] = ['Tair', 'T1', 'ts', 'Tmin', 'Tmax', 'Qair', 'Q1', 'RH2m', 'Wind_Gust', 'PSurf', 'ZS',
                           'BLH', 'Rainf', 'Snowf', 'LWdown', 'LWnet', 'DIR_SWdown', 'SCA_SWdown', 'SWnet', 'SWD',
                           'SWU', 'LHF', 'SHF', 'CC_cumul', 'CC_cumul_low', 'CC_cumul_middle', 'CC_cumul_high',
                           'Wind90', 'Wind87', 'Wind84', 'Wind75', 'TKE90', 'TKE87', 'TKE84', 'TKE75', 'TT90', 'TT87',
                           'TT84', 'TT75', 'SWE', 'snow_density', 'snow_albedo', 'vegetation_fraction', 'Wind', 'Wind_DIR']
variables = config["variables_nwp"]

for variable in variables:
    time_series[variable] = np.nan
filter_time = time_series.index.isin(nwp.time.values)

for idx, station in enumerate(stations["name"][stations["country"] == "france"]):

    # Filters
    filter_station_ts = time_series["name"] == station
    all_filters_ts = filter_time & filter_station_ts
    filter_station_s = stations["name"] == station

    # Coordinates
    xx = np.intp(stations[str_x][filter_station_s].values[0])
    yy = np.intp(stations[str_y][filter_station_s].values[0])

    # Select data
    nwp_station = nwp[variables].isel(xx=xx, yy=yy).to_dataframe()

    # Index intersection
    nwp_station.index = pd.to_datetime(nwp_station.index)
    index_intersection = nwp_station.index.intersection(time_series[variables][all_filters_ts].index)
    nwp_station = nwp_station[nwp_station.index.isin(index_intersection)]

    # Filter intersection
    filter_intersection = time_series.index.isin(index_intersection)
    all_filters = filter_time & filter_station_ts & filter_intersection

    time_series.loc[all_filters, variables] = nwp_station[variables].values


date_min = time_series.index > '2018-10-1'
date_max = time_series.index < '2018-10-30'   # '2019-3-1'
time_series = time_series[date_min & date_max]
def horizontal_wind_component(UV, UV_DIR):
    U = -np.sin(UV_DIR) * UV
    V = -np.cos(UV_DIR) * UV
    return U, V

U_obs, V_obs = horizontal_wind_component(time_series["vw10m(m/s)"].values,
                                         time_series["winddir(deg)"].values)
time_series["U_obs"] = U_obs
time_series["V_obs"] = V_obs

U_AROME, V_AROME = horizontal_wind_component(time_series["Wind"].values, time_series["Wind_DIR"].values)
time_series["U_AROME"] = U_AROME
time_series["V_AROME"] = V_AROME
for station in time_series["name"].unique():
    filter_ts = time_series["name"] == station
    filter_station = stations["name"] == station
    time_series.loc[filter_ts, 'lat'] = stations.loc[filter_station, 'lat']
    time_series.loc[filter_ts, 'lon'] = stations.loc[filter_station, 'lon']
    time_series.loc[filter_ts, 'alti'] = stations.loc[filter_station, 'alti']
time_series = time_series.drop(columns="qc")
config["variables_time_series"] = ['date', 'name', 'lon', 'lat', 'alti', 'T2m(degC)', 'vw10m(m/s)', 'winddir(deg)',
                                   'HTN(cm)'] + config["variables_nwp"] + ["U_obs", "V_obs", "U_AROME", "V_AROME"]


time_series = time_series[config["variables_time_series"]]
variables_not_downscasted = ["date", "name", "last_flagged_speed", "last_flagged_direction",
                             "last_unflagged_direction", "qc_2_speed", "qc_3_speed", "qc_3_direction",
                             "preferred_direction_during_sequence", "qc_5_speed",
                             "cardinal", "preferred_direction"]
for variable in variables_not_downscasted:
    try:
        config["variables_time_series"].remove(variable)
    except ValueError:
        pass
variables = config["variables_time_series"]
time_series[variables] = time_series[variables].astype(np.float32)

list_stations_france = []
for station, country in zip(stations["name"].values, stations["country"].values):
    if country == "france":
        list_stations_france.append(station)
"""
