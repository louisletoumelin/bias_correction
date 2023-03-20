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

for model in ["last", "best"]:

    with tf.device('/GPU:0'):

        """
        # Predict
        with timer_context("Predict train set"):
            inputs_train = data_loader.get_tf_zipped_inputs(mode="train").batch(data_loader.length_train)
            results_train = cm.predict_with_batch(inputs_train, model_str=model)
            del inputs_train
        """

        # Predict
        with timer_context("Predict test set"):
            inputs_test = data_loader.get_tf_zipped_inputs(mode="test").batch(data_loader.length_test)
            results_test = cm.predict_with_batch(inputs_test, model_str=model)
            del inputs_test

        """
        # Predict
        with timer_context("Predict Pyrénées and Corsica"):
            inputs_other_countries = data_loader.get_tf_zipped_inputs(mode="other_countries")\
                .batch(data_loader.length_other_countries)
            results_other_countries = cm.predict_with_batch(inputs_other_countries, model_str=model)
            del inputs_other_countries
        """
    # Test
    print(f"\n\nTest statistics: model={model}")
    data_loader.set_predictions(results_test, mode="test")
    data_loader.add_other_model("_D")
    c_eval = CustomEvaluation(exp, data_loader, mode="test", keys=["_AROME", "_nn"], other_models=["_D"])
    c_eval.print_stats_model()
    exp.save_all(data_loader, c_eval, cm)

    # Test without Aiguille du midi
    print(f"\n\nTest statistics: model={model}")
    data_loader.set_predictions(results_test, mode="test")
    c_eval_no_aiguille = CustomEvaluation(exp, data_loader,
                              mode="test",
                              stations_to_remove=['AGUIL. DU MIDI'],
                              keys=["_AROME", "_nn"])
    c_eval.print_stats_model()
    exp.save_all(data_loader, c_eval_no_aiguille, cm)

    """
    # Train
    print(f"\n\nTrain statistics: model={model}")
    data_loader.set_predictions(results_train, mode="train")
    c_eval_train = CustomEvaluation(exp, data_loader, mode="train", keys=["_AROME", "_nn"])
    c_eval_train.print_stats_model()

    # Other countries
    print(f"\n\nOther countries statistics: model={model}")
    data_loader.set_predictions(results_other_countries, mode="other_countries")
    c_eval_other_countries = CustomEvaluation(exp, data_loader, mode="other_countries", keys=["_AROME", "_nn"])
    c_eval_other_countries.print_stats_model()
    """
    # Save figures
    c_eval.plot_1_1_all(c_eval.df_results, c_eval.keys)
    c_eval.plot_1_1_by_station(c_eval.df_results, c_eval.keys)
    c_eval.plot_seasonal_evolution(c_eval.df_results, keys=c_eval.keys)
    c_eval.plot_seasonal_evolution_by_station(c_eval.df_results, keys=c_eval.keys)
    c_eval.plot_lead_time(c_eval.df_results, keys=c_eval.keys)
    c_eval.plot_boxplot_topo_carac(c_eval.df_results)
    """
    c_eval_other_countries.plot_1_1_all(c_eval_other_countries.df_results, c_eval_other_countries.keys)
    c_eval_other_countries.plot_1_1_by_station(c_eval_other_countries.df_results, c_eval_other_countries.keys)
    c_eval_other_countries.plot_seasonal_evolution(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
    c_eval_other_countries.plot_seasonal_evolution_by_station(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
    c_eval_other_countries.plot_lead_time(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
    """

    i = Interpretability(data_loader, cm, exp)

    with timer_context("plot_feature_importance"):
        i.plot_feature_importance("test")

"""
with timer_context("partial_dependence_plot"):
    i.plot_partial_dependence("test")
"""

print_intro()
