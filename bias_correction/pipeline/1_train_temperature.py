import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_train_temperature import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation

# Initialization
exp = ExperienceManager(config)
data_loader = CustomDataHandler(config)
cm = CustomModel(config, exp)

print("\nCurrent experience:")
print(exp.path_to_current_experience)

print("\nConfig")
print(pprint(config))

# Load inputs and outputs
data_loader.prepare_train_test_data()

inputs_and_labels_train = data_loader.get_tf_zipped_inputs_labels(mode="train")
inputs_and_labels_val = data_loader.get_tf_zipped_inputs_labels(mode="val")

with tf.device('/GPU:0'):
    with timer_context("fit"):
        results_train = cm.fit_with_strategy(inputs_and_labels_train,
                                             validation_data=inputs_and_labels_val)

    # Predict
    with timer_context("Predict"):
        inputs_test = data_loader.get_tf_zipped_inputs(mode="test").batch(data_loader.length_test)
        results_test = cm.predict_with_batch(inputs_test)

data_loader.set_predictions(results_test)

# Evaluation
c_eval = CustomEvaluation(exp, data_loader)
c_eval.df2correlation()
exp.save_all(data_loader, c_eval, cm)

# print results
print(c_eval.df_results[[c_eval.key_obs, c_eval.key_nn, c_eval.key_arome]].round(2))

# Save figures
c_eval.plot_1_1_by_station(c_eval.df_results, c_eval.key_nn)
c_eval.plot_1_1_all(c_eval.df_results, c_eval.key_nn)
