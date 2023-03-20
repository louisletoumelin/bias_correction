import numpy as np
import pandas as pd
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
from bias_correction.train.eval import CustomEvaluation, Interpretability
from bias_correction.utils_bc.print_functions import print_intro

# Initialization
exp = ExperienceManager(config)
data_loader = CustomDataHandler(config)
cm = CustomModel(exp, config)

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
                                             validation_data=inputs_and_labels_val,
                                             dataloader=data_loader,
                                             mode="test")

    # Predict
    with timer_context("Predict test set"):
        inputs_test = data_loader.get_tf_zipped_inputs(mode="test").batch(data_loader.length_test)
        results_test = cm.predict_with_batch(inputs_test)

    # Predict
    with timer_context("Predict Pyrénées and Corsica"):
        inputs_other_countries = data_loader.get_tf_zipped_inputs(mode="other_countries")\
            .batch(data_loader.length_other_countries)
        results_other_countries = cm.predict_with_batch(inputs_other_countries)

data_loader.set_predictions(results_test, mode="test")
data_loader.set_predictions(results_other_countries, mode="other_countries")

# Evaluation
c_eval = CustomEvaluation(exp, data_loader, mode="test", keys=["_AROME", "_nn"])
c_eval_other_countries = CustomEvaluation(exp, data_loader, mode="other_countries", keys=["_AROME", "_nn"])
c_eval.df2correlation()
c_eval_other_countries.df2correlation()
exp.save_all(data_loader, c_eval, cm)

# print results
print(c_eval.df_results[[c_eval.key_obs] + c_eval.keys].round(2))

# Save figures
c_eval.plot_1_1_all(c_eval.df_results, c_eval.keys)
c_eval.plot_1_1_by_station(c_eval.df_results, c_eval.keys)
c_eval.plot_seasonal_evolution(c_eval.df_results, keys=c_eval.keys)
c_eval.plot_seasonal_evolution_by_station(c_eval.df_results, keys=c_eval.keys)
c_eval.plot_lead_time(c_eval.df_results, keys=c_eval.keys)
c_eval.plot_boxplot_topo_carac(c_eval.df_results)
c_eval_other_countries.plot_1_1_all(c_eval_other_countries.df_results, c_eval_other_countries.keys)
c_eval_other_countries.plot_1_1_by_station(c_eval_other_countries.df_results, c_eval_other_countries.keys)
c_eval_other_countries.plot_seasonal_evolution(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
c_eval_other_countries.plot_seasonal_evolution_by_station(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
c_eval_other_countries.plot_lead_time(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)


i = Interpretability(data_loader, cm, exp)

with timer_context("plot_feature_importance"):
    i.plot_feature_importance("test")

with timer_context("partial_dependence_plot"):
    i.plot_partial_dependence("test")

print_intro()
