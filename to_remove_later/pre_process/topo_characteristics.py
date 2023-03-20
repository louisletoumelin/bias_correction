import numpy as np

from downscale.operators.rotation import Rotation
from downscale.operators.helbig import DwnscHelbig
from downscale.operators.micro_met import MicroMet
from downscale.operators.interpolation import Interpolation
from downscale.operators.generators import Generators


class TopoCaracteristics(DwnscHelbig, MicroMet, Rotation, Interpolation, Generators):
    """Compute topographic characteristics at station"""

    n_rows, n_col = 79, 69

    def __init__(self, stations=None, dem=None, dem_pyr_corse=None, config={}):
        super().__init__()
        self.stations = stations
        if hasattr(dem, "alti"):
            self.alti = dem.alti.values
        else:
            self.alti = dem.__xarray_dataarray_variable__.values
        if hasattr(dem_pyr_corse, "alti"):
            self.alti_pyr_corse = dem_pyr_corse.alti.values
        else:
            self.alti_pyr_corse = dem_pyr_corse.__xarray_dataarray_variable__.values
        self.config = config
        self.number_of_neighbors = config.get("number_of_neighbors")
        self.name_dem = config.get("name_dem")
        self.resolution_dem = config.get("resolution_dem")

    def update_station_with_topo_characteristics(self):
        for country in ["france", "swiss", "pyr", "corse"]:
            for neighbor in range(self.number_of_neighbors):
                print(neighbor)
                self.neighbor = neighbor

                str_x = f"X_index_{self.name_dem}_NN_{self.neighbor}_ref_{self.name_dem}"
                str_y = f"Y_index_{self.name_dem}_NN_{self.neighbor}_ref_{self.name_dem}"

                self.idx_x = np.intp(self.stations[str_x].values)
                self.idx_y = np.intp(self.stations[str_y].values)
                self.update_stations_with_laplacian(country)
                self.update_stations_with_tpi(country, radius=2000)
                self.update_stations_with_tpi(country, radius=500)
                self.update_stations_with_mu(country)
                self.update_stations_with_curvature(country)

    def get_alti(self, country):
        if country in ["france", "swiss"]:
            return self.alti
        elif country in ["pyr", "corse"]:
            return self.alti_pyr_corse
        else:
            raise NotImplementedError("No other country than france, swiss, pyr, corse")

    def update_stations_with_laplacian(self, country):
        filter_country = self.stations["country"] == country
        str_lapl = f"laplacian_NN_{self.neighbor}"
        alti = self.get_alti(country)
        self.stations.loc[filter_country, str_lapl] = self._laplacian_loop_numpy_1D(alti,
                                                                                    self.idx_x,
                                                                                    self.idx_y,
                                                                                    self.resolution_dem)

    def update_stations_with_sx(self, sx_direction, country):
        filter_country = self.stations["country"] == country
        str_sx = f"sx_300_NN_{self.neighbor}"
        alti = self.get_alti(country)
        self.stations.loc[filter_country, str_sx] = self.sx_idx(alti,
                                                                self.idx_x,
                                                                self.idx_y,
                                                                cellsize=30,
                                                                dmax=300,
                                                                in_wind=sx_direction,
                                                                wind_inc=5,
                                                                wind_width=30)

    def update_stations_with_tpi(self, country, radius=2000):
        filter_country = self.stations["country"] == country
        str_tpi = f"tpi_{str(int(radius))}_NN_{self.neighbor}"
        alti = self.get_alti(country)
        self.stations.loc[filter_country, str_tpi] = self.tpi_idx(alti,
                                                                  self.idx_x,
                                                                  self.idx_y,
                                                                  radius,
                                                                  resolution=self.resolution_dem)

    def update_stations_with_mu(self, country):
        filter_country = self.stations["country"] == country
        str_mu = f"mu_NN_{self.neighbor}"
        alti = self.get_alti(country)
        self.stations.loc[filter_country, str_mu] = self.mu_helbig_idx(alti,
                                                                       self.resolution_dem,
                                                                       self.idx_x,
                                                                       self.idx_y)

    def update_stations_with_curvature(self, country):
        filter_country = self.stations["country"] == country
        str_curva = f"curvature_NN_{self.neighbor}"
        alti = self.get_alti(country)
        self.stations[filter_country, str_curva] = self.curvature_idx(alti,
                                                                      self.idx_x,
                                                                      self.idx_y,
                                                                      method="fast",
                                                                      scale=False)

    def get_stations(self):
        return self.stations
