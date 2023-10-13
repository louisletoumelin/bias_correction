try:
    import rioxarray # for the extension to load
except:
    pass

import xarray

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")
from typing import Callable
import gzip
import gc

large_dem = "/scratch/mrmn/letoumelinl/bias_correction/Data/1_Raw/DEM/domaine_grande_rousse_large.nc"
small_dem = "/scratch/mrmn/letoumelinl/bias_correction/Data/1_Raw/DEM/domaine_grande_rousse_small.nc"

# The large dem is rotated, cropped and is the input of the cnn.
# This dem should be twice the size of the small dem in order to avoid nans.
large_dem = xarray.open_dataarray(large_dem)

# Try except is used here because depending on how you save the dem into .nc file,
# elevation values can have different name this operation can be removed in the future.
try:
    dem_array = np.expand_dims(large_dem.isel(band=0).values[:, :-1], axis=-1)
except AttributeError:
    dem_array = np.expand_dims(large_dem.isel(band=0).band_data.values[:, :-1], axis=-1)

# The small dem is the target grid.
small_dem = xarray.open_dataarray(small_dem)

#large_dem = large_dem.rio.write_crs("EPSG:2154")
#small_dem = small_dem.rio.write_crs("EPSG:2154")

from bias_correction.config.config_custom_devine import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager

config["custom_input_shape"] = list(dem_array.shape)

# config
config["use_scaling"] = False
config["type_of_output"] = "map_speed_alpha"
batch_size = 1
config["custom_dataloader"] = True
result_name = "acceleration_grandes_rousses_v2"

# Create topo_dict
dict_topo_custom = {"custom": {"data": dem_array}}

# Create time_series
custom_time_series = pd.DataFrame()
# Speed = [3, 3, 3, ...] and direction = [0, 1, 2, ...]
custom_time_series["Wind"] = [3]*360
custom_time_series["Wind_DIR"] = list(range(360))

# Order of 'Wind' and 'Wind_DIR' in dataset is important
assert custom_time_series.columns[0] == "Wind"
assert custom_time_series.columns[1] == "Wind_DIR"

# Initialization
exp = None

# data_loader preprocess data and adapt their format to tensorflow (tf.data)
data_loader = CustomDataHandler(config)

# cm knows how to build a deep learning model, load weights, predict
cm = CustomModel(exp, config)

min_shape = np.intp(np.min(np.squeeze(dem_array).shape) / np.sqrt(2))
output_shape = list(tuple([2]) + tuple([len(custom_time_series)]) + tuple([min_shape, min_shape]))

print(f"dem_array.shape: {dem_array.shape}")
print("\n")
print("We are sure that given any rotation, we will find finite values in an array with min_shape, centered on the initial array")
print(f"min shape: {min_shape}")
print("\n")
print(f"output_shape: [wind component, wind direction, x, y]")
print(f"output_shape: {output_shape}")

# preprocess data
data_loader.prepare_custom_devine_data(custom_time_series, dict_topo_custom)
inputs = data_loader.get_tf_zipped_inputs(mode="custom", output_shapes=config["custom_input_shape"]).batch(batch_size)

# Predict
with timer_context("Predict test set"):
    results = cm.predict_multiple_batches(inputs, batch_size=batch_size, output_shape=output_shape, force_build=True)
    print(results)

print(np.shape(results))
with timer_context("results[0, :, :, :] = results[0, :, :, :]/3"):
    results[0, :, :, :] = results[0, :, :, :] / 3

with timer_context("Change dtype"):
    results[0, :, :, :] = np.float16(results[0, :, :, :])
    results[1, :, :, :] = np.float16(results[1, :, :, :])

with timer_context("Save results"):
    ds = xarray.Dataset(
        data_vars={"acceleration": (("angle", "y", "x"), results[0, :, :, :]),
                   "alpha": (("angle", "y", "x"), results[1, :, :, :])})

    comp = dict(zlib=True, complevel=3)
    encoding = {"acceleration": {"zlib": True, "complevel": 3},
                "alpha": {"zlib": True, "complevel": 3}}

    ds.to_netcdf(f"/scratch/mrmn/letoumelinl/bias_correction/Data/1_Raw/DEM/{result_name}.nc", encoding=encoding)
