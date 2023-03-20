import numpy as np
import pandas as pd
import seaborn as sns
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_train import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, StaticEval

# Initialization
print("\nConfig")
print(pprint(config))

data_loader = CustomDataHandler(config, load_dict_topo=False)
s_eval = StaticEval()

# Load data
stations = data_loader.get_stations()
time_series = data_loader.get_time_series()
time_series, stations = data_loader.reject_stations(time_series, stations, country_to_reject=["pyr", "corse"])
data_loader.prepare_train_test_data()

# Modify time_series and stations
stations = data_loader.add_mode_to_df(stations)
stations = data_loader.add_nwp_stats_to_stations(stations, time_series, metrics=["rmse", "mbe", "corr", "mae"])

# Figure 1: Pair plot parameters
s_eval.plot_pair_plot_parameters(stations)

# Figure 2: Pair plot metrics
s_eval.plot_pair_plot_metrics(stations)

# Figure 3: Pair plot metrics
s_eval.plot_pairplot_all(stations)

# Evaluation of metrics in the train/test/val dataset
time_series = data_loader.add_mode_to_df(time_series)
s_eval.print_train_test_val_stats_above_elevation(stations, time_series)
s_eval.print_train_test_val_stats_by_elevation_category(stations, time_series)
"""
# Boxplot
time_series = data_loader.add_elevation_category_to_df(time_series, list_min=[0, 1000, 2000, 3000], list_max=[1000, 2000, 3000, 5000])
time_series["bias"] = time_series["Wind"] - time_series["vw10m(m/s)"]
time_series = time_series.rename(columns={"bias": "Bias [$m\:s^{-1}$]"})
time_series = time_series.sort_values("cat_zs")
g = sns.catplot(data=time_series,
                y="Bias [$m\:s^{-1}$]",
                x="mode",
                col="cat_zs",
                col_wrap=2,
                kind="box",
                fliersize=0)
g.set(ylim=(-10, 10))
g.set_titles(col_template='{col_name}')
"""