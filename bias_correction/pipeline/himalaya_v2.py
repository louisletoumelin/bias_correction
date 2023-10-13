#import rioxarray # for the extension to load
#import imageio
#from tqdm import tqdm

import xarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")
from typing import Callable
import gzip
import gc
import os
from bias_correction.config.config_custom_devine import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.wind_utils import wind2comp, comp2speed, comp2dir


"""
Configuration
"""


def load_dem(small_dem_path: str, small_dem_epsg=None):
    # Specify the projection of the dem
    # The small dem is the target grid.
    small_dem = xarray.open_dataset(small_dem_path)
    if small_dem.rio.crs is None:
        small_dem = small_dem.rio.write_crs(small_dem_epsg)
    return small_dem


def select_sub_array_in_dem(large_dem):
    dem_array = large_dem.isel(band=0).band_data.values
    height_is_even = dem_array.shape[0] % 2 == 0
    length_is_even = dem_array.shape[1] % 2 == 0

    if height_is_even:
        dem_array = np.expand_dims(large_dem.isel(band=0).band_data.values[:-1, :], axis=-1)
    if length_is_even:
        dem_array = np.expand_dims(large_dem.isel(band=0).band_data.values[:, :-1], axis=-1)

    return dem_array


# Be sure to work on an inner domain that has an uneven number of pixels (easier to replace maps)
def select_sub_array_in_dem(large_dem):
    dem_array = large_dem.isel(band=0).band_data.values
    height_is_even = dem_array.shape[0] % 2 == 0
    length_is_even = dem_array.shape[1] % 2 == 0

    if height_is_even:
        dem_array = np.expand_dims(large_dem.isel(band=0).band_data.values[:-1, :], axis=-1)
    if length_is_even:
        dem_array = np.expand_dims(large_dem.isel(band=0).band_data.values[:, :-1], axis=-1)

    return dem_array


path = config["path_root"] + "Data/1_Raw/Himalaya/"
large_dem_path = path+"large_dem.nc"
small_dem_path = path+"KYZYLSU_DEM_T_C.tif"
large_dem_epsg = "EPSG:32442"
small_dem_epsg = "EPSG:32442"
# Name of the file with acceleration and alpha for each direction
result_name = "acceleration_alpha_tadjikistan_v2.nc"
# Name of the file with acceleration and alpha for each direction replaced on small dem
acceleration_name = "/home/letoumelinl/himalaya/acceleration_himalaya_v2.nc"
alpha_name = "/home/letoumelinl/himalaya/alpha_himalaya_v2.nc"
plot_ = False
create_gif = False


# The large dem is rotated, cropped and is the input of the cnn.
# This dem should be twice the size of the small dem in order to avoid nans.
large_dem = load_dem(large_dem_path, large_dem_epsg)
dem_array = select_sub_array_in_dem(large_dem)

print("\ndem_array shape should be uneven")
print(np.shape(dem_array))

# config
config["custom_input_shape"] = list(dem_array.shape)
config["use_scaling"] = False
config["type_of_output"] = "map_speed_acceleration"
batch_size = 1
config["custom_dataloader"] = True













"""
Prediction for each wind direction
"""

def prepare_custom_data(dem_array: np.ndarray):
    # Create topo_dict
    dict_topo_custom = {"custom": {"data": dem_array}}

    # Create time_series
    custom_time_series = pd.DataFrame()
    # Speed = [3, 3, 3, ...] and direction = [0, 1, 2, ...]
    custom_time_series["Wind"] = [3] * 360
    custom_time_series["Wind_DIR"] = list(range(360))

    # Order of 'Wind' and 'Wind_DIR' in dataset is important
    assert custom_time_series.columns[0] == "Wind"
    assert custom_time_series.columns[1] == "Wind_DIR"

    return dict_topo_custom, custom_time_series


def compute_devine_output_shape(dem_array: np.ndarray):
    min_shape = np.intp(np.min(np.squeeze(dem_array).shape) / np.sqrt(2))  # -1 est ajouté par moi pour enlever un bug
    output_shape = list(tuple([2]) + tuple([len(custom_time_series)]) + tuple([min_shape, min_shape]))
    return output_shape, min_shape


def save_results_devine(ds, path_to_results: str, result_name: str, use_compression: bool = True):
    if use_compression:
        comp = dict(zlib=True, complevel=3)
        encoding = {"acceleration": comp, "alpha": comp}

    print(f"saving to {path_to_results}{result_name}")
    ds.to_netcdf(path_to_results + f"{result_name}", encoding=encoding)


def results_to_dataset(results: np.ndarray):
    ds = xarray.Dataset(
        data_vars={"acceleration": (("angle", "y", "x"), results[0, :, :, :]),
                   "alpha": (("angle", "y", "x"), results[1, :, :, :])})
    return ds


def print_shape_information(dem_array, min_shape, output_shape):
    print(f"dem_array.shape: {dem_array.shape}")
    print("\n")
    print(
        "We are sure that given any rotation, we will find finite values in an array with min_shape, centered on the initial array")
    print(f"min shape: {min_shape}")
    print("\n")
    print(f"output_shape: [wind component, wind direction, x, y]")
    print(f"output_shape: {output_shape}")

dict_topo_custom, custom_time_series = prepare_custom_data(dem_array)

# Initialization
exp = None

# data_loader preprocess data and adapt their format to tensorflow (tf.data)
data_loader = CustomDataHandler(config)

# cm knows how to build a deep learning model, load weights, predict
cm = CustomModel(exp, config)

# Compute output shape
output_shape, min_shape = compute_devine_output_shape(dem_array)

# Print information
print_shape_information(dem_array, min_shape, output_shape)

# preprocess data
data_loader.prepare_custom_devine_data(custom_time_series, dict_topo_custom)
inputs = data_loader.get_tf_zipped_inputs(mode="custom",
                                          output_shapes=config["custom_input_shape"]).batch(batch_size)

cm.build_model_with_strategy()


# Predict
with timer_context("Predict test set"):
    results = cm.predict_multiple_batches(inputs, batch_size=batch_size, output_shape=output_shape, force_build=True)

with timer_context("Change dtype"):
    results = np.float16(results)

with timer_context("Save results"):
    ds = results_to_dataset(results)
    save_results_devine(ds, path, result_name, use_compression=True)







"""
Replace on large_dem
"""



def get_countinuous_data_boundaries(shape_large_dem: tuple):
    """Not called directly. Works with an inner domain with unevend dimensions."""
    # Get shape
    length_y = shape_large_dem[0]
    length_x = shape_large_dem[1]  # -1 before here

    # Define an area where we are sure there won't be any nans
    min_length = np.min([length_y, length_x])
    y_diff = np.intp((min_length / np.sqrt(2))) // 2
    x_diff = y_diff
    y_offset_left = length_y // 2 - y_diff
    y_offset_right = length_y // 2 + y_diff
    x_offset_left = length_x // 2 - x_diff
    x_offset_right = length_x // 2 + x_diff

    return y_offset_left, y_offset_right, x_offset_left, x_offset_right


def dem_to_wind(large_dem, results, shape_large_dem, str_variable: str = "acceleration"):
    y_offset_left, y_offset_right, x_offset_left, x_offset_right = get_countinuous_data_boundaries(shape_large_dem)
    large_dem = large_dem.expand_dims({"angles": 360})
    large_dem = large_dem.isel(band=0)
    large_dem[str_variable] = (("angles", "y", "x"), np.zeros((360,
                                                               large_dem.dims["y"],
                                                               large_dem.dims["x"]),
                                                              dtype=np.float32))  # length_x +1 before here
    large_dem[str_variable].values[:, y_offset_left:y_offset_right, x_offset_left:x_offset_right] = np.float32(results.acceleration.values)
    return large_dem[str_variable].to_dataset()


def reduce_size_large_dem(large_dem, small_dem, margin_in_meters: float = 100):
    large_dem = large_dem.rio.clip_box(
        minx=small_dem.x.min() - margin_in_meters,
        miny=small_dem.y.min() - margin_in_meters,
        maxx=small_dem.x.max() + margin_in_meters,
        maxy=small_dem.y.max() + margin_in_meters)
    return large_dem


def reproject_match(large_dem, resampling=1):
    return large_dem.rio.reproject_match(small_dem, resampling=resampling)


for str_variable, name_output_file in zip(["acceleration", "alpha"], [acceleration_name, alpha_name]):
    results = xarray.open_dataset(path+f"/{result_name}")
    large_dem = load_dem(large_dem_path, large_dem_epsg)
    dem_array = select_sub_array_in_dem(large_dem)

    shape_large_dem = dem_array.shape
    large_dem = dem_to_wind(large_dem, results, shape_large_dem, str_variable="str_variable")
    small_dem = load_dem(small_dem_path, small_dem_epsg)
    large_dem = reduce_size_large_dem(large_dem, small_dem)
    large_dem = reproject_match(large_dem, resampling=1)
    if str_variable == "alpha":
        large_dem["alpha"] = np.rad2deg(large_dem["alpha"], dtype=np.float32)
    large_dem.to_netcdf(name_output_file)
