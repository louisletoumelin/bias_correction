import numpy as np
from scipy.spatial import cKDTree

from bias_correction.pre_process.topo_characteristics import TopoCaracteristics


class Stations(TopoCaracteristics):
    """Create file with station information"""

    def __init__(self,
                 stations=None,
                 nwp_france=None,
                 nwp_swiss=None,
                 nwp_pyr=None,
                 nwp_corse=None,
                 dem=None,
                 dem_pyr_corse=None,
                 config={}):

        super().__init__(stations=stations, dem=dem, dem_pyr_corse=dem_pyr_corse, config=config)
        self.stations = stations
        self.nwp_france = nwp_france
        self.nwp_swiss = nwp_swiss
        self.nwp_pyr = nwp_pyr
        self.nwp_corse = nwp_corse
        self.dem = dem
        self.dem_pyr_corse = dem_pyr_corse
        self.number_of_neighbors = config["number_of_neighbors"]
        self.name_nwp = config.get("name_nwp")
        self.name_dem = config.get("name_dem")
        self.config = config

    @staticmethod
    def x_y_to_stacked_xy(x_array, y_array):
        """
        x_y_to_stacked_xy(1*np.ones((2,2)), 5*np.ones((2,2)))
        array([[[1., 5.],
                [1., 5.]],
               [[1., 5.],
                [1., 5.]]])

        :param x_array: ndarray (x,y)
        :param y_array:ndarray (x,y)
        :return: ndarray (x,y,2)
        """
        stacked_xy = np.dstack((x_array, y_array))
        return stacked_xy

    @staticmethod
    def grid_to_flat(stacked_xy):
        """
        A = array([[[1., 5.],
                    [1., 5.]],
                   [[1., 5.],
                    [1., 5.]]])

        grid_to_flat(A) = [(1.0, 5.0), (1.0, 5.0), (1.0, 5.0), (1.0, 5.0)]

        :param stacked_xy: ndarray
        :return: ndarray
        """
        x_y_flat = [tuple(i) for line in stacked_xy for i in line]
        return x_y_flat

    @staticmethod
    def get_shape_nwp(nwp):
        if ('xx' in nwp.dims and 'yy' in nwp.dims) or ('xx' in nwp.coords and 'yy' in nwp.coords):
            height = nwp.yy.shape[0]
            length = nwp.xx.shape[0]
        elif ('x' in nwp.dims and 'y' in nwp.dims) or ('x' in nwp.coords and 'y' in nwp.coords):
            height = nwp.y.shape[0]
            length = nwp.x.shape[0]
        else:
            raise KeyError("Did not find the name of x and y coordinates/dimensions: xx and x didn't work")
        return height, length

    @staticmethod
    def assert_nwp_is_correct(nwp):
        assert "X_L93" in nwp, "NWP need to have projected coordinates"
        assert "Y_L93" in nwp, "NWP need to have projected coordinates"

    def update_stations_with_knn_from_nwp(self,
                                          interpolated=False):
        """
        Update stations with nearest neighbors in AROME

        stations.columns
        Index(['Unnamed: 0', 'name', 'X', 'Y', 'lon', 'lat', 'alti', 'country',
           'delta_x_AROME_NN_0', 'AROME_NN_0', 'index_AROME_NN_0_ref_AROME',
           'delta_x_AROME_NN_1', 'AROME_NN_1', 'index_AROME_NN_1_ref_AROME',
           'delta_x_AROME_NN_2', 'AROME_NN_2', 'index_AROME_NN_2_ref_AROME',
           'delta_x_AROME_NN_3', 'AROME_NN_3', 'index_AROME_NN_3_ref_AROME'],
          dtype='object')

        :param interpolated: str
        :return: pandasDataFrame
        """

        interp_str = '' if not interpolated else '_interpolated'

        # Initialization
        for neighbor in range(self.number_of_neighbors):
            self.stations[f'delta_x_{self.name_nwp}_NN_{neighbor}{interp_str}'] = np.nan
            self.stations[f'X_{self.name_nwp}_NN_{neighbor}{interp_str}'] = np.nan
            self.stations[f'Y_{self.name_nwp}_NN_{neighbor}{interp_str}'] = np.nan
            self.stations[f'X_index_{self.name_nwp}_NN_{neighbor}{interp_str}_ref_{self.name_nwp}{interp_str}'] = np.nan
            self.stations[f'Y_index_{self.name_nwp}_NN_{neighbor}{interp_str}_ref_{self.name_nwp}{interp_str}'] = np.nan
            self.stations[f'ZS_{self.name_nwp}_NN_{neighbor}{interp_str}'] = np.nan

        nwps = [self.nwp_france, self.nwp_swiss, self.nwp_pyr, self.nwp_corse]
        countries = ["france", "swiss", "pyr", "corse"]
        for nwp, country in zip(nwps, countries):
            print(country)
            stations_i = self.stations[self.stations["country"] == country]

            # Check that nwp have space coordinates
            self.assert_nwp_is_correct(nwp)
            height, length = self.get_shape_nwp(nwp)

            def k_n_n_point(point):
                distance, idx = tree.query(point, k=self.number_of_neighbors)
                return distance, idx

            # Reference stations
            list_coord_station = zip(stations_i['X'].values, stations_i['Y'].values)

            # Coordinates where to find neighbors
            stacked_xy = self.x_y_to_stacked_xy(nwp["X_L93"], nwp["Y_L93"])
            grid_flat = self.grid_to_flat(stacked_xy)
            tree = cKDTree(grid_flat)

            # Computation of nearest neighbors
            list_nearest = map(k_n_n_point, list_coord_station)

            # Store results as array
            list_nearest = np.array([np.array(station) for station in list_nearest])
            list_index = [(x, y) for x in range(height) for y in range(length)]

            # Update DataFrame
            for neighbor in range(self.number_of_neighbors):

                str_delta_x = f'delta_x_{self.name_nwp}_NN_{neighbor}{interp_str}'
                str_x_l93 = f'X_{self.name_nwp}_NN_{neighbor}{interp_str}'
                str_y_l93 = f'Y_{self.name_nwp}_NN_{neighbor}{interp_str}'
                name_str_x = f'X_index_{self.name_nwp}_NN_{neighbor}{interp_str}_ref_{self.name_nwp}{interp_str}'
                name_str_y = f'Y_index_{self.name_nwp}_NN_{neighbor}{interp_str}_ref_{self.name_nwp}{interp_str}'

                if np.ndim(list_nearest) <= 3:
                    list_nearest = np.expand_dims(list_nearest, axis=-1)
                stations_i.loc[:, str_delta_x] = list_nearest[:, 0, neighbor]
                stations_i.loc[:, str_x_l93] = [grid_flat[int(index)][0] for index in list_nearest[:, 1, neighbor]]
                stations_i.loc[:, str_y_l93] = [grid_flat[int(index)][1] for index in list_nearest[:, 1, neighbor]]
                stations_i.loc[:, name_str_x] = [list_index[int(index)][1] for index in list_nearest[:, 1, neighbor]]
                stations_i.loc[:, name_str_y] = [list_index[int(index)][0] for index in list_nearest[:, 1, neighbor]]


                distances = []
                for i in range(len(stations_i)):
                    x = stations_i[name_str_x].iloc[i]
                    y = stations_i[name_str_y].iloc[i]
                    zs = nwp.ZS.isel(time=0).isel(xx=x, yy=y).values
                    distances.append(zs)

                stations_i.loc[:, [f'ZS_{self.name_nwp}_NN_{neighbor}{interp_str}']] = distances

            self.stations.loc[self.stations["country"] == country] = stations_i

    def update_stations_with_knn_from_mnt_using_ckdtree(self):
        """
        Add columns to stations: 'X_L93_AROME_NN_0', 'Y_L93_AROME_NN_0', 'delta_x_AROME_NN_0'
        :return: pandas DataFrame
        """
        for idx, country in enumerate(["france", "swiss", "pyr", "corse"]):
            filter_country = self.stations["country"] == country
            x_country = self.stations.loc[filter_country, 'X']
            y_country = self.stations.loc[filter_country, 'Y']

            nn_l93, nn_index, nn_delta_x = self.search_neighbors_in_dem_using_ckdtree(x_country,
                                                                                      y_country,
                                                                                      country=country)

            for neighbor in range(self.number_of_neighbors):
                name_str_x = f"X_index_{self.name_dem}_NN_{neighbor}_ref_{self.name_dem}"
                name_str_y = f"Y_index_{self.name_dem}_NN_{neighbor}_ref_{self.name_dem}"
                str_x_l93 = f"X_L93_{self.name_dem}_NN_{neighbor}"
                str_y_l93 = f"Y_L93_{self.name_dem}_NN_{neighbor}"
                str_delta_x = f"delta_x_{self.name_dem}_NN_{neighbor}"

                # Initialization
                if idx == 0:
                    self.stations[name_str_x] = np.nan
                    self.stations[name_str_y] = np.nan
                    self.stations[str_x_l93] = np.nan
                    self.stations[str_y_l93] = np.nan
                    self.stations[str_delta_x] = np.nan

                # Insert neighbors
                self.stations.loc[filter_country, [name_str_x]] = [tuple(index)[0] for index in nn_index[neighbor, :]]
                self.stations.loc[filter_country, [name_str_y]] = [tuple(index)[1] for index in nn_index[neighbor, :]]
                self.stations.loc[filter_country, [str_x_l93]] = nn_l93[neighbor, :, 0]
                self.stations.loc[filter_country, [str_y_l93]] = nn_l93[neighbor, :, 1]
                self.stations.loc[filter_country, [str_delta_x]] = nn_delta_x[neighbor, :]

    def update_stations_with_knn_of_nwp_in_mnt_using_ckdtree(self,
                                                             interpolated=False):

        interp_str = "_interpolated" if interpolated else ""
        for idx, country in enumerate(["france", "swiss", "pyr", "corse"]):
            print("idx")
            print(idx)
            filter_country = self.stations["country"] == country

            for neighbor in range(self.number_of_neighbors):
                x_str = self.stations.loc[filter_country, f"X_{self.name_nwp}_NN_{neighbor}{interp_str}"]
                y_str = self.stations.loc[filter_country, f"Y_{self.name_nwp}_NN_{neighbor}{interp_str}"]
                name_str_x = f'X_index_{self.name_nwp}_NN_{neighbor}{interp_str}_ref_{self.name_dem}'
                name_str_y = f'Y_index_{self.name_nwp}_NN_{neighbor}{interp_str}_ref_{self.name_dem}'

                # Initialization
                if idx == 0:
                    self.stations[name_str_x] = np.nan
                    self.stations[name_str_y] = np.nan
                _, nn_index, _ = self.search_neighbors_in_dem_using_ckdtree(x_str,
                                                                            y_str,
                                                                            country=country)
                self.stations.loc[filter_country, name_str_x] = [tuple(index)[0] for index in nn_index[neighbor, :]]
                self.stations.loc[filter_country, name_str_y] = [tuple(index)[1] for index in nn_index[neighbor, :]]

    def get_dem(self, country):
        if country in ["france", "swiss"]:
            return self.dem
        elif country in ["pyr", "corse"]:
            return self.dem_pyr_corse
        else:
            raise NotImplementedError("No other country than france, swiss, pyr, corse")

    def _create_grid_approximate_nn_in_dem(self, approximate_x, approximate_y, country="france"):
        """
        Given an approximate index of a point in a xarray Dataset,
        create a list of the four nearest neigbhors and indexes

        Input:
        .  .  .  .  .  .  .
        .  .  .  .  .  .  .
        .  .  .  .  .  .  .
        .  .  .  X  .  .  .
        .  .  .  .  .  .  .
        .  .  .  .  .  .  .
        .  .  .  .  .  .  .

        Output:
        .  .  .  .  .  .  .
        .  x  x  x  x  x  .
        .  x  x  x  x  x  .
        .  x  x  X  x  x  .
        .  x  x  x  x  x  .
        .  x  x  x  x  x  .
        .  .  .  .  .  .  .

        :param approximate_x: int
            approximate index x of the location
        :param approximate_y: int
            approximate index y of the location
        :return:
        """
        dem = self.get_dem(country)

        list_nearest_neighbors = []
        list_index_neighbors = []
        for i in range(-5, 6):
            for j in range(-5, 6):
                neighbor_x_l93 = dem.x.data[approximate_x + i]
                neighbor_y_l93 = dem.y.data[approximate_y + j]
                list_nearest_neighbors.append((neighbor_x_l93, neighbor_y_l93))
                list_index_neighbors.append((approximate_x + i, approximate_y + j))
        return list_nearest_neighbors, list_index_neighbors

    def _apply_ckdtree(self,
                       list_nearest_neighbors,
                       list_index_neighbors,
                       x_l93_station,
                       y_l93_station,
                       idx_station,
                       arrays_nearest_neighbors_l93,
                       arrays_nearest_neighbors_index,
                       arrays_nearest_neighbors_delta_x):

        """
        Apply CKDtree to approximates neighbors

        :param list_nearest_neighbors:
            Approximate neighbors (L93)
        :param list_index_neighbors:
            Approximate neighbors (index)
        :param x_l93_station:
            Real coordinate x
        :param y_l93_station:
            Real coordinate y
        :param idx_station:
        :param arrays_nearest_neighbors_l93:
        :param arrays_nearest_neighbors_index:
        :param arrays_nearest_neighbors_delta_x:
        :return: arrays_nearest_neighbors_l93, arrays_nearest_neighbors_index, arrays_nearest_neighbors_delta_x
        """

        tree = cKDTree(list_nearest_neighbors)
        distance, all_idx = tree.query((x_l93_station, y_l93_station), k=self.number_of_neighbors)

        if np.ndim(distance) == 0:
            distance = [distance]

        if np.ndim(all_idx) == 0:
            all_idx = [all_idx]

        for index, idx_neighbor in enumerate(all_idx):
            l93_nearest_neighbor = list_nearest_neighbors[idx_neighbor]
            index_mnt_nearest_neighbor = list_index_neighbors[idx_neighbor]
            arrays_nearest_neighbors_l93[index, idx_station, :] = list(l93_nearest_neighbor)
            arrays_nearest_neighbors_index[index, idx_station, :] = list(index_mnt_nearest_neighbor)
            arrays_nearest_neighbors_delta_x[index, idx_station] = distance[index]

        return arrays_nearest_neighbors_l93, arrays_nearest_neighbors_index, arrays_nearest_neighbors_delta_x

    def search_neighbors_in_dem_using_ckdtree(self, list_x_l93, list_y_l93, country="france"):
        """
        :param list_x_l93: list
        :param list_y_l93: list
        :param country: str
        :return: arrays
        """
        mnt_indexes_x, mnt_indexes_y = self.find_nearest_mnt_index(list_x_l93,
                                                                   list_y_l93,
                                                                   resolution_x=30,
                                                                   resolution_y=30,
                                                                   country=country)

        nb_stations = len(mnt_indexes_x)

        arrays_nearest_neighbors_l93 = np.zeros((self.number_of_neighbors, nb_stations, 2))
        arrays_nearest_neighbors_index = np.zeros((self.number_of_neighbors, nb_stations, 2))
        arrays_nearest_neighbors_delta_x = np.zeros((self.number_of_neighbors, nb_stations))

        for idx_station in range(nb_stations):
            x_l93_station, y_l93_station = list_x_l93.values[idx_station], list_y_l93.values[idx_station]
            approximate_x, approximate_y = np.intp(mnt_indexes_x[idx_station]), np.intp(mnt_indexes_y[idx_station])
            list_nearest_neighbors, list_index_neighbors = self._create_grid_approximate_nn_in_dem(approximate_x,
                                                                                                   approximate_y,
                                                                                                   country=country)
            arrays_nn_l93, arrays_nn_index, arrays_nn_delta_x = self._apply_ckdtree(list_nearest_neighbors,
                                                                                    list_index_neighbors,
                                                                                    x_l93_station,
                                                                                    y_l93_station,
                                                                                    idx_station,
                                                                                    arrays_nearest_neighbors_l93,
                                                                                    arrays_nearest_neighbors_index,
                                                                                    arrays_nearest_neighbors_delta_x)

        return arrays_nn_l93, arrays_nn_index, arrays_nn_delta_x

    def find_nearest_mnt_index(self, x, y, resolution_x=30, resolution_y=30, country="france"):
        """
        Find the index of a set of coordinates (x,y) in a DEM

        :param x: int or list
        :param y: int or list
        :param resolution_x: int
        :param resolution_y:int
        :param country:str
        :return: tuple (x,y)
        """
        dem = self.get_dem(country)
        xmin = np.min(dem['x'].values)
        ymax = np.max(dem['y'].values)

        index_x_mnt = np.intp(np.round((x - xmin) // resolution_x))
        index_y_mnt = np.intp(np.round((ymax - y) // resolution_y))

        return index_x_mnt, index_y_mnt

    @staticmethod
    def project_coordinates(lon=None, lat=None, crs_in=4326, crs_out=2154):
        """
        Reproject a lat/lon to other projection

        :param lon: float
        :param lat: float
        :param crs_in: int
        :param crs_out: int
        :return: tuple (x,y)
        """
        import pyproj
        gps_to_l93_func = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
        projected_points = [point for point in gps_to_l93_func.itransform([(lon, lat)])][0]
        return projected_points

    def convert_lat_lon_to_l93(self):
        """
        Convert lat/lon in stations to L93
        :return stations: DataFrame
        """
        # Where X or Y is not nan (typically in France), we don't reproject
        filter_nan = np.logical_and(np.isnan(self.stations["X"]), np.isnan(self.stations["Y"]))

        x_list = []
        y_list = []
        for lon, lat in self.stations[["lon", "lat"]][filter_nan].values:
            x, y = self.project_coordinates(lon=lon, lat=lat, crs_in=4326, crs_out=2154)
            x_list.append(x)
            y_list.append(y)

        self.stations.loc[filter_nan, ["X"]] = x_list
        self.stations.loc[filter_nan, ["Y"]] = y_list

    def interpolate_nwp(self):
        self.nwp_france = self.interpolate_wind_grid_xarray(self.nwp_france.isel(time=slice(0, 2)),
                                                            interp=self.config["interp"],
                                                            method=self.config["method"],
                                                            verbose=self.config["verbose"])

        self.nwp_swiss = self.interpolate_wind_grid_xarray(self.nwp_swiss.isel(time=slice(0, 2)),
                                                           interp=self.config["interp"],
                                                           method=self.config["method"],
                                                           verbose=self.config["verbose"])
        if self.nwp_pyr is not None:
            self.nwp_pyr = self.interpolate_wind_grid_xarray(self.nwp_pyr.isel(time=slice(0, 2)),
                                                             interp=self.config["interp"],
                                                             method=self.config["method"],
                                                             verbose=self.config["verbose"])
            print("Interpolation not computed on nwp_pyr (nwp_pyr is None)")

        if self.nwp_corse is not None:
            self.nwp_corse = self.interpolate_wind_grid_xarray(self.nwp_corse.isel(time=slice(0, 2)),
                                                               interp=self.config["interp"],
                                                               method=self.config["method"],
                                                               verbose=self.config["verbose"])
            print("Interpolation not computed on nwp_corse (nwp_corse is None)")

    def change_dtype_stations(self, analysis=False):
        """
        Change the dtype for each column of the DataFrame.
        :return: pandas DataFrame
        """
        if analysis:
            self.stations = self.stations.astype({"name": str,
                                                  "X": np.float32,
                                                  "Y": np.float32,
                                                  "lon": np.float32,
                                                  "lat": np.float32,
                                                  "alti": np.float32,
                                                  "country": str,
                                                  "X_index_DEM_NN_0_ref_DEM": np.intp,
                                                  "Y_index_DEM_NN_0_ref_DEM": np.intp,
                                                  "X_L93_DEM_NN_0": np.float32,
                                                  "Y_L93_DEM_NN_0": np.float32,
                                                  "delta_x_DEM_NN_0": np.float32,
                                                  "delta_x_AROME_NN_0": np.float32,
                                                  "X_AROME_NN_0": np.float32,
                                                  "Y_AROME_NN_0": np.float32,
                                                  "X_index_AROME_NN_0_ref_AROME": np.intp,
                                                  "Y_index_AROME_NN_0_ref_AROME": np.intp,
                                                  "ZS_AROME_NN_0": np.float32,
                                                  "X_index_AROME_NN_0_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_0_ref_DEM": np.intp,
                                                  "delta_x_AROME_NN_0_interpolated": np.float32,
                                                  "X_AROME_NN_0_interpolated": np.float32,
                                                  "Y_AROME_NN_0_interpolated": np.float32,
                                                  "X_index_AROME_NN_0_interpolated_ref_AROME_interpolated": np.intp,
                                                  "Y_index_AROME_NN_0_interpolated_ref_AROME_interpolated": np.intp,
                                                  "ZS_AROME_NN_0_interpolated": np.float32,
                                                  "X_index_AROME_NN_0_interpolated_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_0_interpolated_ref_DEM": np.intp
                                                  }, errors="ignore")
        else:
            self.stations = self.stations.astype({"name": str,
                                                  "X": np.float32,
                                                  "Y": np.float32,
                                                  "lon": np.float32,
                                                  "lat": np.float32,
                                                  "alti": np.float32,
                                                  "country": str,
                                                  "X_index_DEM_NN_0_ref_DEM": np.intp,
                                                  "Y_index_DEM_NN_0_ref_DEM": np.intp,
                                                  "X_L93_DEM_NN_0": np.float32,
                                                  "Y_L93_DEM_NN_0": np.float32,
                                                  "delta_x_DEM_NN_0": np.float32,
                                                  "X_index_DEM_NN_1_ref_DEM": np.intp,
                                                  "Y_index_DEM_NN_1_ref_DEM": np.intp,
                                                  "X_L93_DEM_NN_1": np.float32,
                                                  "Y_L93_DEM_NN_1": np.float32,
                                                  "delta_x_DEM_NN_1": np.float32,
                                                  "X_index_DEM_NN_2_ref_DEM": np.intp,
                                                  "Y_index_DEM_NN_2_ref_DEM": np.intp,
                                                  "X_L93_DEM_NN_2": np.float32,
                                                  "Y_L93_DEM_NN_2": np.float32,
                                                  "delta_x_DEM_NN_2": np.float32,
                                                  "X_index_DEM_NN_3_ref_DEM": np.intp,
                                                  "Y_index_DEM_NN_3_ref_DEM": np.intp,
                                                  "X_L93_DEM_NN_3": np.float32,
                                                  "Y_L93_DEM_NN_3": np.float32,
                                                  "delta_x_DEM_NN_3": np.float32,
                                                  "delta_x_AROME_NN_0": np.float32,
                                                  "X_AROME_NN_0": np.float32,
                                                  "Y_AROME_NN_0": np.float32,
                                                  "X_index_AROME_NN_0_ref_AROME": np.intp,
                                                  "Y_index_AROME_NN_0_ref_AROME": np.intp,
                                                  "ZS_AROME_NN_0": np.float32,
                                                  "delta_x_AROME_NN_1": np.float32,
                                                  "X_AROME_NN_1": np.float32,
                                                  "Y_AROME_NN_1": np.float32,
                                                  "X_index_AROME_NN_1_ref_AROME": np.intp,
                                                  "Y_index_AROME_NN_1_ref_AROME": np.intp,
                                                  "ZS_AROME_NN_1": np.float32,
                                                  "delta_x_AROME_NN_2": np.float32,
                                                  "X_AROME_NN_2": np.float32,
                                                  "Y_AROME_NN_2": np.float32,
                                                  "X_index_AROME_NN_2_ref_AROME": np.intp,
                                                  "Y_index_AROME_NN_2_ref_AROME": np.intp,
                                                  "ZS_AROME_NN_2": np.float32,
                                                  "delta_x_AROME_NN_3": np.float32,
                                                  "X_AROME_NN_3": np.float32,
                                                  "Y_AROME_NN_3": np.float32,
                                                  "X_index_AROME_NN_3_ref_AROME": np.intp,
                                                  "Y_index_AROME_NN_3_ref_AROME": np.intp,
                                                  "ZS_AROME_NN_3": np.float32,
                                                  "X_index_AROME_NN_0_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_0_ref_DEM": np.intp,
                                                  "X_index_AROME_NN_1_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_1_ref_DEM": np.intp,
                                                  "X_index_AROME_NN_2_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_2_ref_DEM": np.intp,
                                                  "X_index_AROME_NN_3_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_3_ref_DEM": np.intp,
                                                  "delta_x_AROME_NN_0_interpolated": np.float32,
                                                  "X_AROME_NN_0_interpolated": np.float32,
                                                  "Y_AROME_NN_0_interpolated": np.float32,
                                                  "X_index_AROME_NN_0_interpolated_ref_AROME_interpolated": np.intp,
                                                  "Y_index_AROME_NN_0_interpolated_ref_AROME_interpolated": np.intp,
                                                  "ZS_AROME_NN_0_interpolated": np.float32,
                                                  "delta_x_AROME_NN_1_interpolated": np.float32,
                                                  "X_AROME_NN_1_interpolated": np.float32,
                                                  "Y_AROME_NN_1_interpolated": np.float32,
                                                  "X_index_AROME_NN_1_interpolated_ref_AROME_interpolated": np.intp,
                                                  "Y_index_AROME_NN_1_interpolated_ref_AROME_interpolated": np.intp,
                                                  "ZS_AROME_NN_1_interpolated": np.float32,
                                                  "delta_x_AROME_NN_2_interpolated": np.float32,
                                                  "X_AROME_NN_2_interpolated": np.float32,
                                                  "Y_AROME_NN_2_interpolated": np.float32,
                                                  "X_index_AROME_NN_2_interpolated_ref_AROME_interpolated": np.intp,
                                                  "Y_index_AROME_NN_2_interpolated_ref_AROME_interpolated": np.intp,
                                                  "ZS_AROME_NN_2_interpolated": np.float32,
                                                  "delta_x_AROME_NN_3_interpolated": np.float32,
                                                  "X_AROME_NN_3_interpolated": np.float32,
                                                  "Y_AROME_NN_3_interpolated": np.float32,
                                                  "X_index_AROME_NN_3_interpolated_ref_AROME_interpolated": np.intp,
                                                  "Y_index_AROME_NN_3_interpolated_ref_AROME_interpolated": np.intp,
                                                  "ZS_AROME_NN_3_interpolated": np.float32,
                                                  "X_index_AROME_NN_0_interpolated_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_0_interpolated_ref_DEM": np.intp,
                                                  "X_index_AROME_NN_1_interpolated_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_1_interpolated_ref_DEM": np.intp,
                                                  "X_index_AROME_NN_2_interpolated_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_2_interpolated_ref_DEM": np.intp,
                                                  "X_index_AROME_NN_3_interpolated_ref_DEM": np.intp,
                                                  "Y_index_AROME_NN_3_interpolated_ref_DEM": np.intp,
                                                  "laplacian_NN_0": np.float32,
                                                  "tpi_2000_NN_0": np.float32,
                                                  "tpi_500_NN_0": np.float32,
                                                  "mu_NN_0": np.float32,
                                                  "curvature_NN_0": np.float32,
                                                  "laplacian_NN_1": np.float32,
                                                  "tpi_2000_NN_1": np.float32,
                                                  "tpi_500_NN_1": np.float32,
                                                  "mu_NN_1": np.float32,
                                                  "curvature_NN_1": np.float32,
                                                  "laplacian_NN_2": np.float32,
                                                  "tpi_2000_NN_2": np.float32,
                                                  "tpi_500_NN_2": np.float32,
                                                  "mu_NN_2": np.float32,
                                                  "curvature_NN_2": np.float32,
                                                  "laplacian_NN_3": np.float32,
                                                  "tpi_2000_NN_3": np.float32,
                                                  "tpi_500_NN_3": np.float32,
                                                  "mu_NN_3": np.float32,
                                                  "curvature_NN_3": np.float32
                                                  }, errors="ignore")

    def save_to_pickle(self, name=None):
        if name is None:
            name = ""
        self.stations.to_pickle(self.config["path_stations_pre_processed"] + f"stations_bc{name}.pkl")

    def save_to_csv(self, name=None):
        if name is None:
            name = ""
        self.stations.to_csv(self.config["path_stations_pre_processed"] + f"stations_bc{name}.csv")
