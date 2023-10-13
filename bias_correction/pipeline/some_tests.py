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

from bias_correction.config.config_double_v1 import config
from bias_correction.train.dataloader import MapGeneratorUncentered


names = ["Col du Lac Blanc", "Col du Lac Blanc", "Col du Lac Blanc"]
idx_x = [0, 25, -13]
idx_y = [0, 25, 28]
gen = MapGeneratorUncentered(names, idx_x, idx_y, config)

for i, ix, iy in zip(gen(), idx_x, idx_y):
    print(ix, iy)
    plt.figure()
    plt.imshow(i)
    plt.title(f"idx_x={ix} idx_y={iy}")

config["random_idx"] = True
"idx_x"
"idx_y"
# assert idx_x and idx_y before wind speed and wind direction
config["labels"] = ""
