import numpy as np
import pickle
from collections import defaultdict


class DictTopo:
    """Create dictionaries containing topographic information"""
    def __init__(self, stations=None, dem=None, dem_pyr_corse=None, config={}):
        self.stations = stations
        self.dem = dem
        self.dem_pyr_corse = dem_pyr_corse
        self.config = config
        self.name_nwp = config["name_nwp"]
        self.name_dem = config["name_dem"]
        self.nb_pixel_x = config["nb_pixel_topo_x"]
        self.nb_pixel_y = config["nb_pixel_topo_y"]

    def get_dem(self, country):
        if country in ["france", "swiss"]:
            return self.dem
        elif country in ["pyr", "corse"]:
            return self.dem_pyr_corse
        else:
            raise NotImplementedError("No other country than france, swiss, pyr, corse")

    def extract_dem_around_station(self, station, country):

        filter_station = self.stations["name"] == station
        dem = self.get_dem(country)

        # Names
        # before
        # f'X_index_{self.name_nwp}_NN_0_ref_{self.name_dem}'
        # f'Y_index_{self.name_nwp}_NN_0_ref_{self.name_dem}'
        str_x = "X_index_DEM_NN_0_ref_DEM"
        str_y = "Y_index_DEM_NN_0_ref_DEM"

        # Station idx
        idx_x = np.intp(self.stations.loc[filter_station, str_x].values[0])
        idx_y = np.intp(self.stations.loc[filter_station, str_y].values[0])

        # Borders
        y_left = np.intp(idx_y - self.nb_pixel_y)
        y_right = np.intp(idx_y + self.nb_pixel_y)
        x_left = np.intp(idx_x - self.nb_pixel_x)
        x_right = np.intp(idx_x + self.nb_pixel_x)

        # Select data
        if hasattr(dem, "alti"):
            dem_data = dem.alti.values[y_left:y_right, x_left:x_right]
        else:
            dem_data = dem.__xarray_dataarray_variable__.values[0, y_left:y_right, x_left:x_right]
        dem_x = dem.x.values[x_left:x_right]
        dem_y = dem.y.values[y_left:y_right]

        return np.float32(dem_data), np.float32(dem_x), np.float32(dem_y)

    def extract_dem_around_nwp_neighbor(self, station, country, interpolated=False):

        interp_str = "_interpolated" if interpolated else ""
        filter_station = self.stations["name"] == station
        dem = self.get_dem(country)

        str_x = f'X_index_{self.name_nwp}_NN_0{interp_str}_ref_{self.name_dem}'
        str_y = f'Y_index_{self.name_nwp}_NN_0{interp_str}_ref_{self.name_dem}'
        idx_x = np.intp(self.stations.loc[filter_station, str_x].values[0])
        idx_y = np.intp(self.stations.loc[filter_station, str_y].values[0])

        if hasattr(dem, "alti"):
            dem_data = dem.alti.values[idx_y - self.nb_pixel_y:idx_y + self.nb_pixel_y,
                       idx_x - self.nb_pixel_x:idx_x + self.nb_pixel_x]
        else:
            dem_data = dem.__xarray_dataarray_variable__.values[0, idx_y - self.nb_pixel_y:idx_y + self.nb_pixel_y,
                       idx_x - self.nb_pixel_x:idx_x + self.nb_pixel_x]
        dem_x = dem.x.values[idx_x - self.nb_pixel_x:idx_x + self.nb_pixel_x]
        dem_y = dem.y.values[idx_y - self.nb_pixel_y:idx_y + self.nb_pixel_y]

        return np.float32(dem_data), np.float32(dem_x), np.float32(dem_y)

    def store_topo_in_dict(self):
        dict_0 = defaultdict(dict)
        dict_1 = defaultdict(dict)
        dict_2 = defaultdict(dict)

        for country in ["france", "swiss", "pyr", "corse"]:
            filter_country = self.stations["country"] == country
            for station in self.stations.loc[filter_country, "name"]:
                print(f"Extracting topo around {station}")
                dict_0[station]["data"], dict_0[station]["x"], dict_0[station]["y"] = self.extract_dem_around_station(station, country)
                dict_1[station]["data"], dict_1[station]["x"], dict_1[station]["y"] = self.extract_dem_around_nwp_neighbor(station, country=country, interpolated=False)
                dict_2[station]["data"], dict_2[station]["x"], dict_2[station]["y"] = self.extract_dem_around_nwp_neighbor(station, country=country, interpolated=True)

                dict_0[station]["name"] = station
                dict_1[station]["name"] = station
                dict_2[station]["name"] = station

        with open(self.config["path_topos_pre_processed"]+'dict_topo_near_station_2022_10_26.pickle', 'wb') as handle:
            pickle.dump(dict_0, handle)

        with open(self.config["path_topos_pre_processed"]+'dict_topo_near_nwp.pickle_2022_10_26', 'wb') as handle:
            pickle.dump(dict_1, handle)

        with open(self.config["path_topos_pre_processed"]+'dict_topo_near_nwp_inter_2022_10_26.pickle', 'wb') as handle:
            pickle.dump(dict_2, handle)

