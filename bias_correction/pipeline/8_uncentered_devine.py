import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm as cm_plt
from tensorflow.keras.models import Model

try:
    import seaborn as sns
    _sns = True
except ModuleNotFoundError:
    _sns = False

# matplotlib.use('Agg')

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
import os
from pprint import pprint
import gc
import sys


sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_uncentered_devine import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.utils_bc.print_functions import print_intro, print_headline
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.visu_map import plot_quiver


#names = ["Col du Lac Blanc", "Col du Lac Blanc", "Col du Lac Blanc"]
#idx_x = [0, 25, -13]
#idx_y = [0, 25, 28]
#gen = MapGeneratorUncentered(names, idx_x, idx_y, config)

#for i, ix, iy in zip(gen(), idx_x, idx_y):
#    print(ix, iy)
#    plt.figure()
#    plt.imshow(i)
#    plt.title(f"idx_x={ix} idx_y={iy}")

print_headline("Create a new experience", "")
exp = ExperienceManager(config)
print("exp.path_debug")
print(exp.path_debug)

"""
tf.debugging.experimental.enable_dump_debug_info(
    exp.path_debug+"tfdbg2_logdir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=50_000)
"""

print("\nCurrent experience:", flush=True)
print(exp.path_to_current_experience)

print("\nConfig", flush=True)
print(pprint(config), flush=True)

cm = CustomModel(exp, config)
cm.build_model_with_strategy()

#try:
#    cm.load_weights(path="/scratch/mrmn/letoumelinl/bias_correction/Data/3_Predictions/Experiences/2023_5_12_labia_v0/last_weights/")
#except:
#    cm.load_weights(path="/home/letoumelinl/bias_correction/Data/3_Predictions/tuning_DEVINE/2023_5_12_labia_v0/last_weights/")
# Data
with timer_context("Prepare data"):
    data_loader = CustomDataHandler(config)
    data_loader.prepare_train_test_data()

#try:
#    cm.load_weights(path="/scratch/mrmn/letoumelinl/bias_correction/Data/3_Predictions/Experiences/2023_3_30_labia_v4/last_weights/")
#except:
#    cm.load_weights(path="/home/letoumelinl/bias_correction/Data/3_Predictions/tuning_DEVINE/2023_3_30_labia_v4/last_weights/")

# Fit
with tf.device('/GPU:0'), timer_context("fit"):
    _ = cm.fit_with_strategy(data_loader.get_batched_inputs_labels(mode="train"),
                             dataloader=data_loader,
                             mode_callback="train")
    # Save weights
    exp.save_model(cm)

config["modified_latent"] = True
#config["activation_dense_modified"]
#config["initializer_modified"]
#config["dropout_rate_modified"]
print_intro()
