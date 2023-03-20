try:
    import rioxarray # for the extension to load
    _rioxarray = True
except ModuleNotFoundError:
    _rioxarray = False
import xarray as xr
import numpy as np
import pandas as pd
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import sys

sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")
import gc
import uuid
import os

from bias_correction.config.config_custom_devine import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.wind_utils import comp2dir, comp2speed, wind2comp
"""
On GPU: 
2.88 min for prediction
5.24 min for saving
"""
# config
path = config["path_root"] + "Data/1_Raw/Himalaya/"
#path = os.getcwd()  # todo warning change path
# Unique identifier
config["uuid_str"] = str(uuid.uuid4())[:4]
config["use_scaling"] = False
config["type_of_output"] = "map"
config["batch_size"] = 3
config["angle_step"] = 1
config['name_cnn_outputs'] = "himalaya"
crs_nwp = "EPSG:4326"
name_era5_u = "ERA5Land_TJK_u10_2021.nc"
name_era5_v = "ERA5Land_TJK_v10_2021.nc"
name_large_dem = "KYZYLSU_DEM_larger_finer.tif"
name_large_dem = "large_dem.nc" #todo change here
name_small_dem = "KYZYLSU_DEM_T_C.tif"
name_wind_direction_small_dem = f"wind_direction_small_dem_{config['uuid_str']}.nc"
name_acceleration_small_dem = f"acceleration_small_dem_{config['uuid_str']}.nc"

# Predict with DEVINE
# shared between both parts
large_dem = xr.open_dataset(path + name_large_dem)

try:
    print("Initial large_dem shape:")
    print(large_dem.isel(band=0).values.shape)
    dem_array = np.expand_dims(large_dem.isel(band=0).values[:, :-1], axis=-1)
    print("large_dem shape after first reduction:")
    print(large_dem.isel(band=0).values.shape)
except AttributeError:
    print("Initial large_dem shape:")
    print(large_dem.isel(band=0).band_data.values.shape)
    dem_array = np.expand_dims(large_dem.isel(band=0).band_data.values[:, :-1], axis=-1)
    print("large_dem shape after first reduction:")
    print(large_dem.isel(band=0).band_data.values.shape)

config["custom_input_shape"] = dem_array.shape

# Create topo_dict
dict_topo_custom = {"custom": {"data": dem_array}}

# Create time_series
custom_time_series = pd.DataFrame()
custom_time_series["Wind"] = [3] * len(list(range(0, 360, config["angle_step"])))
custom_time_series["Wind_DIR"] = list(range(0, 360, config["angle_step"]))

# Input should be sorted: "Wind" then "Wind_DIR"
assert custom_time_series.columns[0] == "Wind"
assert custom_time_series.columns[1] == "Wind_DIR"

# Initialization
data_loader = CustomDataHandler(config)
cm = CustomModel(None, config)

# output_shape = min(initial_shape) / sqrt(2)
# this correspond to the shape of the data that go to CNN
min_shape = np.intp(np.min(np.squeeze(dem_array).shape) / np.sqrt(2))
output_shape = tuple([2]) + tuple([len(custom_time_series)]) + tuple([min_shape, min_shape])
print(f"Shape of topographic data for CNN predictions: {output_shape}")

# get data
data_loader.prepare_custom_devine_data(custom_time_series, dict_topo_custom)
batched_inputs = data_loader.get_tf_zipped_inputs(mode="custom",
                                                  output_shapes=config["custom_input_shape"])\
    .batch(config["batch_size"])

# Predict
with timer_context("Predict test set"):
    cnn_outputs = cm.predict_multiple_batches(batched_inputs,
                                              batch_size=config["batch_size"],
                                              output_shape=output_shape,
                                              force_build=True)

# This application consumes a lot of memory, so we call frequently
# the garbage collector to avoid "out of memory"" errors.
del batched_inputs
gc.collect()

# Accelerations = cnn_outputs / 3 (outputs are only valid for an input speed of 3 m/s)
with timer_context("cnn_outputs[0, :, :, :] = cnn_outputs[0, :, :, :]/3"):
    cnn_outputs[0, :, :, :] = cnn_outputs[0, :, :, :] / 3

# Reduce output type
with timer_context("Change dtype"):
    # saving in float16 and loading in float32 induces an error of approximately 0.0009763241 in terms of speed
    # saving in float16 and loading in float32 induces an error of approximately 0.125 in terms of direction
    # saving in int16 and loading in float32 induces an error of approximately 0.99 in terms of direction
    # file in float16: 5Go, file in float32: 12Go
    cnn_outputs[0, :, :, :] = np.float16(cnn_outputs[0, :, :, :])
    cnn_outputs[1, :, :, :] = np.float16(cnn_outputs[1, :, :, :])

gc.collect()

with timer_context("Save cnn_outputs"):
    ds = xr.Dataset(data_vars={"acceleration": (("angle", "y", "x"), cnn_outputs[0, :, :, :]),
                               "Wind_DIR": (("angle", "y", "x"), cnn_outputs[1, :, :, :])})
    comp = dict(zlib=True, complevel=3)
    encoding = {"acceleration": comp, "Wind_DIR": comp}
    name_cnn_outputs = f"cnn_outputs_{config['name_cnn_outputs']}_{config['uuid_str']}.nc"
    ds.to_netcdf(path + name_cnn_outputs, encoding=encoding)
    print(f"cnn_outputs saved to netcdf: {path + name_cnn_outputs}")

"""
name_cnn_outputs = "cnn_outputs_himalaya_aac7.nc"

try:
    large_dem = xr.open_dataset(path + name_large_dem).band_data
except AttributeError:
    large_dem = xr.open_dataset(path + name_large_dem)

# Load predictions
cnn_outputs = xr.open_dataset(name_cnn_outputs)

# Replace prediction on large_dem
shape_large_dem = large_dem.values.shape

# Get shape
if len(shape_large_dem) == 3:
    length_y = shape_large_dem[1]
    length_x = shape_large_dem[2] - 1
elif len(shape_large_dem) == 2:
    length_y = shape_large_dem[0]
    length_x = shape_large_dem[1] - 1

# Size where we are sure there will be no nans
min_length = np.min([length_y, length_x])
y_diff = np.intp((min_length / np.sqrt(2))) // 2
x_diff = y_diff

y_offset_left = length_y // 2 - y_diff
y_offset_right = length_y // 2 + y_diff + 1
x_offset_left = length_x // 2 - x_diff
x_offset_right = length_x // 2 + x_diff + 1

# Reduce large_dem to minimum architecture and add angles dimension
large_dem = large_dem.to_dataset()
large_dem = large_dem.drop_dims("band")
large_dem = large_dem.expand_dims({"angles": 360})

# Put acceleration in large_dem
large_dem["acceleration"] = (("angles", "y", "x"), np.zeros((360, length_y, length_x + 1), dtype=np.float32))
large_dem["acceleration"].values[:, y_offset_left:y_offset_right, x_offset_left:x_offset_right] = np.float32(cnn_outputs.acceleration.values)

del cnn_outputs
gc.collect()

# Reduce size of large_dem to save memory
large_dem = large_dem.rio.clip_box(
    minx=707305 - 100,
    miny=4319768 - 100,
    maxx=720806 + 100,
    maxy=4341368 + 100)

gc.collect()

# Reproject and match small_dem
small_dem = xr.open_dataarray(path + name_small_dem)
large_dem = large_dem.rio.reproject_match(small_dem, resampling=1)

# Save file acceleration
large_dem.to_netcdf(path+name_acceleration_small_dem)

del large_dem
gc.collect()

# Reload data
results = xr.open_dataset(path+name_cnn_outputs)
large_dem = xr.open_dataarray(path+name_large_dem)
large_dem = large_dem.to_dataset()
large_dem = large_dem.drop_dims("band")
large_dem = large_dem.expand_dims({"angles": 360})

# Put direction in large_dem
large_dem["Wind_DIR"] = (("angles", "y", "x"), np.zeros((360, length_y, length_x + 1), dtype=np.int16))
large_dem["Wind_DIR"].values[:, y_offset_left:y_offset_right, x_offset_left:x_offset_right] = np.int16(results.Wind_DIR.values)

del results
import gc

gc.collect()

large_dem = large_dem.rio.clip_box(
    minx=707305 - 100,
    miny=4319768 - 100,
    maxx=720806 + 100,
    maxy=4341368 + 100)

gc.collect()

# Reproject and match small_dem
large_dem = large_dem.rio.reproject_match(small_dem, resampling=1)

# Save file acceleration
large_dem.to_netcdf(path+name_wind_direction_small_dem)

del large_dem
gc.collect()

# Interpolate ERA5
era5_u = xr.open_dataarray(path+name_era5_u)
era5_v = xr.open_dataarray(path+name_era5_v)

# Be sure both files have CRS
if era5_u.rio.crs is None:
    era5_u.rio.write_crs(crs_nwp, inplace=True)

if era5_v.rio.crs is None:
    era5_v.rio.write_crs(crs_nwp, inplace=True)

assert era5_u.rio.crs is not None
assert era5_v.rio.crs is not None
assert small_dem.rio.crs is not None

era5_u_on_small_dem = era5_u.rio.reproject_match(small_dem, resampling=1)
era5_v_on_small_dem = era5_v.rio.reproject_match(small_dem, resampling=1)

del era5_u
del era5_v
gc.collect()

era5_uv_on_small_dem = comp2speed(era5_u_on_small_dem, era5_v_on_small_dem)
era5_wind_dir_on_small_dem = np.round(comp2dir(era5_u_on_small_dem, era5_v_on_small_dem))

acceleration = xr.open_dataset(path+name_acceleration_small_dem)
wind_direction = xr.open_dataset(path+name_wind_direction_small_dem)

# Unify wind direction
filter_direction = era5_wind_dir_on_small_dem == 360
era5_wind_dir = xr.where(filter_direction, 0, era5_wind_dir_on_small_dem)

# Get unique direction values (so that we don't compute lookup for directions that are not present in input files)
unique_wind_directions = np.intp(np.unique(era5_wind_dir.values))

# Create dataset for downscaled values
downscaled_uv = era5_uv_on_small_dem.copy(deep=True)
downscaled_wind_direction = era5_wind_dir_on_small_dem.copy(deep=True)

# Lookup and replace values
for direction in unique_wind_directions:
    filter_direction = era5_wind_dir == direction
    downscaled_wind_direction = xr.where(filter_direction,
                                         wind_direction.isel(angles=direction).Wind_DIR,
                                         downscaled_wind_direction)
    downscaled_uv = xr.where(filter_direction,
                             era5_uv_on_small_dem * acceleration.isel(angles=direction).acceleration,
                             downscaled_uv)

"""
"""
# test for lookup function

cnn_outputs = np.ones((2, 25, 5, 5))
for i in range(cnn_outputs.shape[1]):
    cnn_outputs[0, i, :, :] = cnn_outputs[0, i, :, :]*i+0.5
cnn_outputs = xr.Dataset(
    data_vars={"Wind_DIR": (("angle", "y", "x"), cnn_outputs[0, :, :, :]),
               "acceleration": (("angle", "y", "x"), 0.1*cnn_outputs[1, :, :, :])})
era5_dir = np.random.randint(25, size=(100, 5, 5))
era5_dir = xr.Dataset(data_vars={"Wind_DIR": (("time", "y", "x"), era5_dir)})
era5_speed = era5_dir.copy(deep=True)
downscaled_wind_dir = era5_dir.copy(deep=True)
downscaled_speed = era5_speed.copy(deep=True)
for direction in range(25):
    filter_direction = era5_dir == direction
    downscaled_wind_dir = xr.where(filter_direction, 
                                   cnn_outputs.isel(angle=direction).Wind_DIR,
                                   downscaled_wind_dir)
    downscaled_speed = xr.where(filter_direction, 
                                era5_speed*cnn_outputs.isel(angle=direction).acceleration,
                                downscaled_speed)

"""

