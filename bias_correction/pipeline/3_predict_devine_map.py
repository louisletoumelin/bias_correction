import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LightSource

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint
from datetime import datetime

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
from bias_correction.train.visu_map import VisuMap

# Initialization
if config["restore_experience"]:
    exp, config = ExperienceManager.from_previous_experience(config["restore_experience"])
    cm = CustomModel.from_previous_experience(exp, config)
else:
    exp = ExperienceManager(config)
    cm = CustomModel(exp, config)

data_loader = CustomDataHandler(config)

print("\nCurrent experience:")
print(exp.path_to_current_experience)

print("\nConfig")
print(pprint(config))

# Load inputs and outputs
data_loader.prepare_train_test_data()
inputs_and_labels_train = data_loader.get_tf_zipped_inputs_labels(mode="train")
inputs_and_labels_val = data_loader.get_tf_zipped_inputs_labels(mode="val")

time_series = data_loader.get_time_series(prepared=False)
d0 = pd.to_datetime(datetime(2020, 1, 5, 4))
d = pd.to_datetime(datetime(2020, 1, 5, 5))
d1 = pd.to_datetime(datetime(2020, 1, 5, 6))

d0 = pd.to_datetime(datetime(2020, 1, 2, 15))
d = pd.to_datetime(datetime(2020, 1, 2, 17))
d1 = pd.to_datetime(datetime(2020, 1, 2, 18))

d0 = pd.to_datetime(datetime(2020, 1, 3, 15))
d = pd.to_datetime(datetime(2020, 1, 3, 17))
d1 = pd.to_datetime(datetime(2020, 1, 3, 18))
station = "Col du Lac Blanc"
station = "AGUIL. DU MIDI"
station = "LE GUA-NIVOSE"

vm = VisuMap(exp, d0, d, d1, station, config)
vm.plot_quiver(time_series, data_loader, cm)
