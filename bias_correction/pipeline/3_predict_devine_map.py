import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LightSource
from matplotlib.dates import DateFormatter
import seaborn as sns

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint
from datetime import datetime, timedelta
import os
import glob
import contextlib
from PIL import Image

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_devine_map import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, Interpretability
from bias_correction.train.wind_utils import wind2comp, comp2speed
from bias_correction.train.visu import save_figure
from bias_correction.train.visu_map import VisuMap
from bias_correction.utils_bc.print_functions import print_intro, print_headline
from bias_correction.utils_bc.plot_utils import MidPointNorm
from bias_correction.train.windrose import plot_windrose
from bias_correction.train.metrics import abs_bias_direction

if config["restore_experience"]:
    print_headline("Restore experience", "")
    exp, config = ExperienceManager.from_previous_experience(config["restore_experience"])
    config["get_intermediate_output"] = False
    config["type_of_output"] = "map"
    config["map_variables"] = ["topos"]
    cm = CustomModel.from_previous_experience(exp, config, "last")
else:
    print_headline("Create a new experience", "")
    exp = ExperienceManager(config)
    config["get_intermediate_output"] = False
    config["type_of_output"] = "map"

data_loader = CustomDataHandler(config)

print("\nCurrent experience:")
print(exp.path_to_current_experience)

print("\nConfig")
print(pprint(config))

# Load inputs and outputs

data_loader.prepare_train_test_data()
data_loader.add_model("_D", mode="test")
data_loader.add_model("_A")

#inputs_and_labels_train = data_loader.get_tf_zipped_inputs_labels(mode="train")
#inputs_and_labels_val = data_loader.get_tf_zipped_inputs_labels(mode="val")

d0 = pd.to_datetime(datetime(2020, 1, 3, 15))
d = pd.to_datetime(datetime(2020, 1, 3, 17))
d1 = pd.to_datetime(datetime(2020, 1, 3, 18))

d0 = pd.to_datetime(datetime(2020, 1, 2, 15))
d = pd.to_datetime(datetime(2020, 1, 2, 17))
d1 = pd.to_datetime(datetime(2020, 1, 2, 18))

d0 = pd.to_datetime(datetime(2020, 1, 5, 4))
d = pd.to_datetime(datetime(2020, 1, 5, 5))
d1 = pd.to_datetime(datetime(2020, 1, 5, 6))

d = pd.to_datetime(datetime(2019, 12, 18, 4))
d = pd.to_datetime(datetime(2020, 6, 25, 7))
d = pd.to_datetime(datetime(2020, 5, 23, 17))  # marche bien à COV
d = pd.to_datetime(datetime(2019, 11, 12, 16))
d = pd.to_datetime(datetime(2019, 12, 22, 14))
d = pd.to_datetime(datetime(2019, 11, 23, 9))
d = pd.to_datetime(datetime(2019, 11, 24, 5))  # marche bien
d = pd.to_datetime(datetime(2020, 5, 3, 10))
d = pd.to_datetime(datetime(2019, 10, 8, 6))
#d = data_loader.get_inputs("test").index.iloc[0]

station = "LE GUA-NIVOSE"
station = "AGUIL. DU MIDI"
station = "Col du Lac Blanc"
station = "COV"
#BEH" et "MLS"
#station = "Col du Lac Blanc"

from datetime import date, timedelta


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(hours=n)

#for model in ["devine_only", "double_ann"]:

"""
for idx, n in enumerate(range(72)):

    d = datetime(2019, 10, 8, 1) + timedelta(hours=n)
    print(d)
    for model in ["devine_only"]:

        config["global_architecture"] = model
        data_loader.config["global_architecture"] = model
        cm.config["global_architecture"] = model
        if model == "devine_only":
            config["standardize"] = False
            data_loader.config["standardize"] = False
            cm.config["standardize"] = False
        else:
            config["standardize"] = True
            data_loader.config["standardize"] = True
            cm.config["standardize"] = True

        vm = VisuMap(exp, d, station, data_loader, model, config)
        vm.plot_quiver(cm,
                       edgecolor="black",
                       vert_exag=15_000,
                       scale=1/0.04,
                       nb_arrows_to_skip=3,
                       levels=10,
                       width=0.005,
                       vmin=0.41,
                       vmax=0.53,
                       cmap="viridis",
                       name=idx)
"""
"""
fp_0 = "/home/letoumelinl/Images/gifpres/"
fp_in = fp_0 + "*.png"
fp_out = f"/home/letoumelinl/Images/cov_gif_{station}.gif"

for file in glob.glob(fp_in):
    print(file.split('_'))
    os.rename(file, file.split('_')[0]+".png")
"""
"""    
# use exit stack to automatically close opened images
files = (fp_0+f"{i}.png" for i in range(35))
with contextlib.ExitStack() as stack:
    # lazily load images
    imgs = (stack.enter_context(Image.open(f)) for f in files)

    # extract  first image from iterator
    img = next(imgs)

    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=450, loop=0)
"""
"""
time_series = data_loader.loader.load_time_series_pkl()
time_series = time_series[time_series["name"] == station]
#analysis = data_loader.predicted_A
#devine = data_loader.predicted_D
nn_speed = pd.read_pickle(os.path.join(exp.path_to_current_experience, f"df_results_UV.pkl"))
nn_speed = nn_speed[nn_speed["name"] == station]
nn_dir = pd.read_pickle(os.path.join(exp.path_to_current_experience, f"df_results_UV_DIR.pkl"))
nn_dir = nn_dir[nn_dir["name"] == station]

nb_days = 1
fontsize = 12
sns.set_style("ticks", {'axes.grid': True})
date_form = DateFormatter("%Y-%m-%d-%H h")

filter_time_low = (time_series.index >= d - pd.Timedelta(nb_days, "d"))
filter_time_high = (time_series.index <= d + pd.Timedelta(nb_days, "d"))
nn_s_filter_time_low = (nn_speed.index >= d - pd.Timedelta(nb_days, "d"))
nn_s_filter_time_high = (nn_speed.index <= d + pd.Timedelta(nb_days, "d"))
nn_d_filter_time_low = (nn_dir.index >= d - pd.Timedelta(nb_days, "d"))
nn_d_filter_time_high = (nn_dir.index <= d + pd.Timedelta(nb_days, "d"))

filter_arome_one = time_series["Wind"] >= 1
filter_obs_one = time_series["vw10m(m/s)"] >= 1
wind_is_positive = filter_arome_one & filter_obs_one

#
#
# Time series speed
#
#
plt.figure()
ax = plt.gca()
#time_series[["Wind", "vw10m(m/s)"]][filter_time_low & filter_time_high].plot(ax=ax, color=["C1", "black"])
#analysis[["UV_A"]][(analysis.index >= d - pd.Timedelta(nb_days, "d")) & (analysis.index <= d + pd.Timedelta(nb_days, "d")) & (analysis["name"] == "COV")].plot(ax=ax)
#devine[["UV_D"]][(devine.index >= d - pd.Timedelta(nb_days, "d")) & (devine.index <= d + pd.Timedelta(nb_days, "d")) & (devine["name"] == "COV")].plot(ax=ax)
nn_speed[['UV_AROME', "UV_obs", "UV_nn"]][nn_s_filter_time_low & nn_s_filter_time_high].plot(ax=ax, color="C2")
ax.xaxis.set_major_formatter(date_form)
#nn_speed["ones"] = 1
#nn_speed[["ones"]][nn_s_filter_time_low & nn_s_filter_time_high].plot(color="black", ax=ax)
plt.legend(("$AROME_{forecast}$", "Observations", "$Neural\:Network+DEVINE$"), fontsize=fontsize)
plt.ylabel(("Wind speed [$m\:s^{-1}$]"), fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)

#time_series[["Wind", "vw10m(m/s)", "name"]][(time_series.index == d) & (time_series["name"] == "COV")].plot(ax=ax, marker="x", markersize=10, linestyle="")
x = np.squeeze([time_series.index[time_series.index == (d - pd.Timedelta(1, "H"))], time_series.index[time_series.index == (d + pd.Timedelta(1, "H"))]])
y1 = [0, 0]
y2 = [12, 12]
plt.fill_between(x, y1, y2, color="black", alpha=0.1)
save_figure(f"Map_wind_fields/time_series_speed", exp=exp, svg=True)

#
#
# Time series direction
#
#
plt.figure()
fig = plt.gcf()
ax = plt.gca()
#time_series[["Wind_DIR", "winddir(deg)"]][filter_time_low & filter_time_high & wind_is_positive].plot(marker="x", linestyle="", color=["C1", "black"], ax=ax)
#analysis[["UV_DIR_A"]][(analysis.index >= d - pd.Timedelta(nb_days, "d")) & (analysis.index <= d + pd.Timedelta(nb_days, "d")) & (analysis["name"] == "COV")].plot(marker="x", linestyle="",ax=ax)
#devine[["UV_DIR_D"]][(devine.index >= d - pd.Timedelta(nb_days, "d")) & (devine.index <= d + pd.Timedelta(nb_days, "d")) & (devine["name"] == "COV")].plot(marker="x", linestyle="",ax=ax)
nn_dir[["UV_DIR_AROME", "UV_DIR_obs", "UV_DIR_nn"]][nn_d_filter_time_low & nn_d_filter_time_high].plot(marker="x", linestyle="", color=["C1", "black", "C2"], ax=ax)
ax.xaxis.set_major_formatter(date_form)
plt.legend(("$AROME_{forecast}$", "Observations", "$Neural\:Network+DEVINE$"), fontsize=fontsize)
plt.ylabel(("Wind direction [°]"), fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
fig.autofmt_xdate()

x = np.squeeze([time_series.index[time_series.index == (d - pd.Timedelta(1, "H"))], time_series.index[time_series.index == (d + pd.Timedelta(1, "H"))]])
y1 = [0, 0]
y2 = [360, 360]
plt.fill_between(x, y1, y2, color="black", alpha=0.1)
save_figure(f"Map_wind_fields/time_series_dir", exp=exp, svg=True)

#
#
# Time series bias direction
#
#

plt.figure()
ax = plt.gca()
#time_series["bias_nn"] = np.nan
#time_series["bias_nn"][time_series.index.isin(time_series.index.intersection(nn_dir.index))] = abs_bias_direction(time_series["winddir(deg)"][time_series.index.isin(time_series.index.intersection(nn_dir.index))].sort_index().values, nn_dir["UV_DIR_nn"][nn_dir.index.isin(nn_dir.index.intersection(time_series.index))].sort_index().values)
#time_series["bias_arome"] = abs_bias_direction(time_series["winddir(deg)"].values, time_series["Wind_DIR"].values)
#time_series[["bias_arome", "bias_nn"]][filter_time_low & filter_time_high & wind_is_positive].plot(marker="x", linestyle="", color=["C1", "C2"], ax=ax)


index_intersection = nn_speed.index.intersection(nn_dir.index)
#filter_time = nn_dir.index.isin(index_intersection)
#filter_arome_one = (nn_speed["UV_AROME"] >= 1) & nn_speed.index.isin(index_intersection)
#filter_obs_one = (nn_speed["UV_obs"] >= 1)
#wind_is_positive = filter_arome_one & filter_obs_one & filter_time

nn_dir[["abs_bias_direction_AROME", "abs_bias_direction_nn"]][nn_d_filter_time_low & nn_d_filter_time_high].plot(marker="x", linestyle="", color=["C1", "C2"], ax=ax)
ax.xaxis.set_major_formatter(date_form)
plt.legend(("$AROME_{forecast}$", "$Neural\:Network+DEVINE$"), fontsize=fontsize)
plt.ylabel(("Wind direction \n absolute error [°]"), fontsize=fontsize)
ax.tick_params(axis='both', which='major', labelsize=fontsize)
x = np.squeeze([time_series.index[time_series.index == (d - pd.Timedelta(1, "H"))], time_series.index[time_series.index == (d + pd.Timedelta(1, "H"))]])
y1 = [0, 0]
y2 = [360, 360]
plt.fill_between(x, y1, y2, color="black", alpha=0.1)
save_figure(f"Map_wind_fields/time_series_dir_error", exp=exp, svg=True)
"""
# Modified DEVINE

#cm.build_model_with_strategy(True)
# cm.load_weights(os.getcwd()+"/last_weights/")
model = "devine_only"
config["global_architecture"] = model
data_loader.config["global_architecture"] = model
cm.config["global_architecture"] = model
config["standardize"] = False
data_loader.config["standardize"] = False
cm.config["standardize"] = False
vm = VisuMap(exp, d, station, data_loader, model, config)
vm.plot_quiver(cm,
               edgecolor="black",
               vert_exag=15_000,
               scale=1 / 0.04,
               nb_arrows_to_skip=3,
               levels=10,
               width=0.005,
               vmin=0.41,
               vmax=0.53,
               path_to_last_weights="/home/letoumelinl/bias_correction/Data/3_Predictions/tuning_DEVINE/2023_5_17_labia_v5/last_weights/"
               )
plt.title("Modified_DEVINE")

print_intro()
