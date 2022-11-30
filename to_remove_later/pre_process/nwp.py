import numpy as np
import xarray as xr
import os


class Nwp:

    def __init__(self, config):
        self.config = config

    def add_L93_to_all_nwp_files(self):
        """
        Updates nwp files with L93 coordinates. This function should be run on the labia

        :param config: dict
        :return: modified nwp files
        """
        if self.config["network"] == "local":

            print("\nadd_L93_to_all_nwp_files not called")
            print("Should be run on labia")

        elif self.config["network"] == "labia":

            for country in ["alp", "swiss", "pyr", "corse"]:

                print(f"Add X_L93 and Y_L93 to country {country}")
                path_x_y_l93 = self.config[f"path_X_Y_L93_{country}"]
                path_nwp = self.config[f"path_nwp_{country}"]
                X_L93 = np.load(path_x_y_l93 + 'X_L93.npy')
                Y_L93 = np.load(path_x_y_l93 + 'Y_L93.npy')
                path_nwp = path_nwp.split("month/")[0]+"without_L93/"
                for file in os.listdir(path_nwp):

                    print(f"Add X_L93 and Y_L93 to {file}")
                    nwp = xr.open_dataset(path_nwp+file)
                    nwp['X_L93'] = (('yy', 'xx'), X_L93)
                    nwp['Y_L93'] = (('yy', 'xx'), Y_L93)
                    nwp.to_netcdf(path_nwp+"_new_"+file)
                    print(f"File with X_L93 and Y_L93 is called {'_new_'+file}")

    @staticmethod
    def gps_to_l93(data_xr=None, lon='longitude', lat='latitude'):
        """
        Converts a grid of lat/lon to L93

        :param data_xr: xr.Dataset
        :param lon: str
        :param lat: str
        :return: xr.Dataset
        """

        import pyproj

        # Initialization
        length = data_xr.xx.shape[0]
        height = data_xr.yy.shape[0]

        shape = (height, length)
        X_L93 = np.zeros(shape)
        Y_L93 = np.zeros(shape)

        # Load transformer
        gps_to_l93_func = pyproj.Transformer.from_crs(4326, 2154, always_xy=True)

        # Transform coordinates of each points
        for j in range(height):
            for i in range(length):
                if hasattr(data_xr[lon], "time"):
                    projected_points = [point for point in
                                        gps_to_l93_func.itransform(
                                            [(data_xr[lon].isel(time=-1).values[j, i], data_xr[lat].isel(time=-1).values[j, i])])]
                else:
                    projected_points = [point for point in
                                        gps_to_l93_func.itransform(
                                            [(data_xr[lon].values[j, i], data_xr[lat].values[j, i])])]
                X_L93[j, i], Y_L93[j, i] = projected_points[0]

        # Create a new variable with new coordinates
        data_xr["X_L93"] = (("yy", "xx"), X_L93)
        data_xr["Y_L93"] = (("yy", "xx"), Y_L93)

        return data_xr

    @staticmethod
    def project_coordinates(lon=None, lat=None, crs_in=4326, crs_out=2154):
        import pyproj
        gps_to_l93_func = pyproj.Transformer.from_crs(crs_in, crs_out, always_xy=True)
        projected_points = [point for point in gps_to_l93_func.itransform([(lon, lat)])][0]
        return projected_points

    def check_all_lon_and_lat_are_the_same_in_nwp(self):
        """
        Check that the lat/lon variables are the same in all files
        :param config: dict
        """
        if self.config["network"] == "local":

            print("\ncheck_all_lon_and_lat_are_the_same_in_nwp not called")
            print("Should be run on labia")

        elif self.config["network"] == "labia":

            for path in [self.config["path_nwp_alp"],
                         self.config["path_nwp_swiss"],
                         self.config["path_nwp_pyr"],
                         self.config["path_nwp_corse"]]:

                dims_x = []
                dims_y = []
                longitudes = []
                latitudes = []
                path = path.split("month/")[0]+"without_L93/"
                for file in os.listdir(path):
                    print(f"Check lat/lon in {file})")
                    nwp = xr.open_dataset(path + file)
                    dims_x.append(nwp.dims['xx'])
                    dims_y.append(nwp.dims['yy'])
                    if "longitude" in nwp:
                        print("selected longitude/latitude")

                        nwp["longitude"] = (("yy", "xx"), nwp.longitude.isel(time=-1).values)
                        nwp["latitude"] = (("yy", "xx"), nwp.latitude.isel(time=-1).values)
                        print("Replace longitude and latitude by last value")
                        nwp.to_netcdf(path + '_new' + file)
                        print("saved to netcdf")

                        try:
                            longitudes.append(nwp.longitude.isel(time=-1).values)
                            latitudes.append(nwp.latitude.isel(time=-1).values)
                            print("selected isel")
                            print(nwp.longitude.isel(time=-1).values)
                            print(nwp.latitude.isel(time=-1).values)
                        except:
                            longitudes.append(nwp.longitude.values)
                            latitudes.append(nwp.latitude.values)
                            print("didnt selected isel")
                            print(nwp.longitude)
                            print(nwp.latitude)
                    elif "LON" in nwp:
                        print("selected LON/LAT")
                        try:
                            longitudes.append(nwp.LON.isel(time=0).values)
                            latitudes.append(nwp.LAT.isel(time=0).values)
                        except:
                            longitudes.append(nwp.LON.values)
                            latitudes.append(nwp.LAT.values)
                assert len(set(dims_x)) == 1, print(dims_x)
                assert len(set(dims_y)) == 1, print(dims_y)

                print(set(dims_x))
                print(set(dims_y))

                for lon, lat in zip(longitudes, latitudes):
                    np.testing.assert_almost_equal(longitudes[0], lon, decimal=3)
                    np.testing.assert_almost_equal(latitudes[0], lat, decimal=3)

    def compute_l93(self, nwp, country="france"):
        try:
            X_Y_L93 = self.gps_to_l93(nwp, lon='longitude', lat='latitude')
        except (ValueError, KeyError):
            X_Y_L93 = self.gps_to_l93(nwp, lon='LON', lat='LAT')
        np.save(self.config[f"path_X_Y_L93_{country}"] + "X_L93.npy", X_Y_L93["X_L93"].values)
        np.save(self.config[f"path_X_Y_L93_{country}"] + "Y_L93.npy", X_Y_L93["Y_L93"].values)

    def _check_L93_in_folder(self):
        for country in ["alp", "swiss", "pyr", "corse"]:
            assert "X_L93.npy" in os.listdir(self.config[f"path_X_Y_L93_{country}"])
            assert "Y_L93.npy" in os.listdir(self.config[f"path_X_Y_L93_{country}"])
            print(f"X_L93.npy and Y_L93.npy exist in {country}")

    def save_L93_npy(self):
        self._check_L93_in_folder()

    def print_send_L93_npy_to_labia(self):
        if self.config["network"] == "local":
            print("X_L93.npy and Y_L93 need to be sent to labia")
        elif self.config["network"] == "labia":
            self._check_L93_in_folder()

    def downcast_to_float32(self):
        for country in ["alp", "swiss", "pyr", "corse"]:
            print(f"downscasting to float32 for country {country}")
            path_nwp = self.config[f"path_nwp_{country}"]
            path_nwp = path_nwp.split("month/")[0] + "with_L93_64bits/"
            for file_name in os.listdir(path_nwp):
                print(file_name)
                try:
                    file_name_short = file_name.split("_new_")[1]
                except IndexError:
                    file_name_short = file_name
                path_to_file = self.config[f"path_nwp_{country}"]+file_name_short
                xr.open_dataset(path_nwp+file_name).astype(np.float32).to_netcdf(path_to_file)

    def add_Z0_to_all_nwp_files(self):
        print("impossible because we don't have Z0 for Switzerland")
