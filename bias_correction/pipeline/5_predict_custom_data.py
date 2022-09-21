import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")
from pprint import pprint


from bias_correction.config.config_custom_devine import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.visu_map import plot_quiver

# Create topo_dict
data_path = "/home/letoumelinl/bias_correction/Data/1_Raw/Custom/"
dem = np.loadtxt(data_path + "dem.asc", skiprows=6)
dict_topo_custom = {"custom": {"data": np.reshape(dem, (90, 88, 1))}}

# Create time_series
custom_time_series = pd.DataFrame()
custom_time_series["Wind"] = [2.3, 1.3, 1.3, 1.1, 1.4]
custom_time_series["Wind_DIR"] = [200, 235, 248, 221, 290]

# Initialization
exp = ExperienceManager(config)
data_loader = CustomDataHandler(config)
cm = CustomModel(exp, config)

# Load inputs and outputs
data_loader.prepare_custom_devine_data(custom_time_series, dict_topo_custom)

# Predict
with timer_context("Predict test set"):
    inputs = data_loader.get_tf_zipped_inputs(mode="custom",
                                              output_shapes=config["custom_input_shape"])\
        .batch(data_loader.length_custom)
    results = cm.predict_single_bath(inputs, force_build=True)

plt.figure()
initial_length_x = 88
initial_length_y = 90
x_offset = 31
y_offset = 31
y_offset_left = initial_length_y // 2 - y_offset
y_offset_right = initial_length_y // 2 + y_offset + 1
x_offset_left = initial_length_x // 2 - x_offset
x_offset_right = initial_length_x // 2 + x_offset + 1
plt.imshow(dem[y_offset_left:y_offset_right, x_offset_left:x_offset_right])

for idx_fig in range(5):
    speed = custom_time_series["Wind"].values[idx_fig]
    dir = custom_time_series["Wind_DIR"].values[idx_fig]

    U_final = np.full_like(dem, np.nan)
    V_final = np.full_like(dem, np.nan)
    W_final = np.full_like(dem, np.nan)
    UV_final = np.full_like(dem, np.nan)

    U = results[0][idx_fig, :, :, 0]
    U_final[y_offset_left:y_offset_right, x_offset_left:x_offset_right] = U
    U_final = np.where(np.isnan(U_final), -9999, U_final)
    # U = np.where(U>10, -9999, U)
    plt.figure()
    plt.imshow(U)
    plt.title(f"U_{idx_fig}_s_{speed}_d_{dir}")
    np.savetxt(f"U_{idx_fig}_s_{speed}_d_{dir}.asc", U_final, fmt='%.4e')
    plt.colorbar()

    V = results[1][idx_fig, :, :, 0]
    V_final[y_offset_left:y_offset_right, x_offset_left:x_offset_right] = V
    V_final = np.where(np.isnan(V_final), -9999, V_final)
    # V = np.where(V>8, -9999, V)
    plt.figure()
    plt.imshow(V)
    plt.title(f"V_{idx_fig}_s_{speed}_d_{dir}")
    np.savetxt(f"V_{idx_fig}_s_{speed}_d_{dir}.asc", V_final, fmt='%.4e')

    W = results[2][idx_fig, :, :, 0]
    W_final[y_offset_left:y_offset_right, x_offset_left:x_offset_right] = W
    W_final = np.where(np.isnan(W_final), -9999, W_final)
    plt.colorbar()
    plt.figure()
    plt.imshow(W)
    plt.title(f"W_{idx_fig}_s_{speed}_d_{dir}")
    np.savetxt(f"W_{idx_fig}_s_{speed}_d_{dir}.asc", W_final, fmt='%.4e')

    UV = np.sqrt(U**2+V**2)
    UV_final[y_offset_left:y_offset_right, x_offset_left:x_offset_right] = UV
    UV_final = np.where(np.isnan(UV_final), -9999, UV_final)
    plt.colorbar()
    plt.figure()
    plt.imshow(UV)
    plt.title(f"UV_{idx_fig}_s_{speed}_d_{dir}")
    np.savetxt(f"UV_{idx_fig}_s_{speed}_d_{dir}.asc", UV_final, fmt='%.4e')

    plt.colorbar()
    plt.figure()
    x = list(range(np.shape(results[0][idx_fig, :, :, 0])[1]))
    y = list(range(np.shape(results[0][idx_fig, :, :, 0])[0]))
    x, y = np.meshgrid(x, y)
    plot_quiver(x, y, U, V, W,
                cmap="coolwarm", norm=None, linewidths=1, scale=1 / 0.01, axis_equal=True, edgecolor=None)
    plt.title(f"UV_{idx_fig}_s_{speed}_d_{dir}")

    x = []
    for i in range(88):
        x.append(2777636 + i * 25)
    x = np.array(x)
    y = []
    for i in range(90):
        y.append(1185112 + i * 25)
    y = np.array(y)
    print(x[x_offset_left:x_offset_right])
    print(y[y_offset_left:y_offset_right])
    print(np.min(x[x_offset_left:x_offset_right]))
    print(np.min(y[y_offset_left:y_offset_right]))
    print(np.shape(dem[y_offset_left:y_offset_right, x_offset_left:x_offset_right]))
