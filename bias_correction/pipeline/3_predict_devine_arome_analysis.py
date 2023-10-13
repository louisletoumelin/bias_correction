import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_devine import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, Interpretability

# Change path to time_series with path to analysis
config["path_to_analysis"] = config["path_root"] + "Data/2_Pre_processed/AROME_analysis/"
config["time_series"] = config["path_to_analysis"] + "time_series_bc_a.pkl"

# Initialization
#  x[:, 0] is nwp wind speed.
#  x[:, 1] is wind direction.
exp = ExperienceManager(config)
data_loader = CustomDataHandler(config)
cm = CustomModel(exp, config)
cm.build_model_with_strategy()

print("\nCurrent experience:")
print(exp.path_to_current_experience)

print("\nConfig")
print(pprint(config))

# Load inputs and outputs
data_loader.prepare_train_test_data()

with tf.device('/GPU:0'):

    # Predict
    with timer_context("Predict test set"):
        inputs_test = data_loader.get_tf_zipped_inputs(mode="test").batch(data_loader.length_test)
        results_test = cm.predict_single_bath(inputs_test, force_build=True)

    # Predict
    with timer_context("Predict Pyrénées and Corsica"):
        inputs_other_countries = data_loader.get_tf_zipped_inputs(mode="other_countries")\
            .batch(data_loader.length_other_countries)

        results_other_countries = cm.predict_single_bath(inputs_other_countries)

for mode, result in zip(["test", "other_countries"], [results_test, results_other_countries]):
    data_loader.set_predictions(result, mode=mode, str_model="_D")
    df = data_loader.get_predictions(mode)
    print("exp.path_to_predictions")
    print(exp.path_to_predictions+f"devine_arome_analysis_2023_03_16_{config['type_of_output']}_{mode}.pkl")
    df.to_pickle(exp.path_to_predictions+f"devine_arome_analysis_2023_03_16_{config['type_of_output']}_{mode}.pkl")
