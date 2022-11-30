import numpy as np
import pandas as pd
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint
import gc

import sys

sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_some_tests import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, Interpretability
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.utils_bc.print_functions import print_intro, print_headline
from bias_correction.utils_bc.utils_config import detect_variable

# Initialization
if config["restore_experience"]:
    exp, config = ExperienceManager.from_previous_experience(config["restore_experience"])
    cm = CustomModel.from_previous_experience(exp, config, "last")
else:
    exp = ExperienceManager(config)
    cm = CustomModel(exp, config)

print("\nCurrent experience:")
print(exp.path_to_current_experience)

print("\nConfig")
print(pprint(config))

# Load inputs and outputs
data_loader = CustomDataHandler(config)
data_loader.prepare_train_test_data()

"""

if not config["restore_experience"]:
    with tf.device('/GPU:0'), timer_context("fit"):
        _ = cm.fit_with_strategy(data_loader.get_batched_inputs_labels(mode="train"),
                                 validation_data=data_loader.get_batched_inputs_labels(mode="val"),
                                 dataloader=data_loader,
                                 mode_callback="test")
exp.save_all(data_loader, cm)
"""
for type_of_output in ["output_speed", "output_direction"]:

    if config["type_of_output"] != type_of_output:
        config["type_of_output"] = type_of_output
        config = detect_variable(config)
        cm.build_model_with_strategy()
        cm.model.load_weights(cm.exp.path_to_last_model)
        cm.model_version = "last"

    for model in ["last"]:  # ["last", "best"]
        print_headline("Model", model)

        # Train
        # with timer_context("Predict train set"):
        #    inputs_train = data_loader.get_tf_zipped_inputs(mode="train").batch(data_loader.length_train)
        #    results_train = cm.predict_single_bath(inputs_train, model_str=model)
        #    del inputs_train

        # print(f"\n\n_______________________")
        # print(f"Train statistics: model={model}")
        # print(f"_______________________\n\n")
        # data_loader.set_predictions(results_train, mode="train")
        # c_eval_train = CustomEvaluation(exp, data_loader, mode="train", keys=("_AROME", "_nn"))
        # c_eval_train.print_stats()

        # Test
        with tf.device('/GPU:0'), timer_context("Predict test set"):
            inputs_test = data_loader.get_tf_zipped_inputs(mode="test").batch(data_loader.length_test)
            results_test = cm.predict_single_bath(inputs_test, model_version=model)
            del inputs_test

        print_headline("Test statistics", model)
        data_loader.set_predictions(results_test, mode="test")
        del results_test
        data_loader.add_model("_D", mode="test")
        data_loader.add_model("_A")
        c_eval = CustomEvaluation(exp, data_loader, mode="test", keys=("_AROME", "_nn"), other_models=("_D", "_A"))
        c_eval.print_stats()
        exp.save_results(c_eval)

        """
        # Predict
        with tf.device('/GPU:0'), timer_context("Predict Pyrénées and Corsica"):
            inputs_other_countries = data_loader.get_tf_zipped_inputs(mode="other_countries") \
                .batch(data_loader.length_other_countries)
            results_other_countries = cm.predict_single_bath(inputs_other_countries, model_version=model)
            del inputs_other_countries
        
        # Other countries
        print_headline("Other countries statistics", model)
        data_loader.set_predictions(results_other_countries, mode="other_countries")
        del results_other_countries
        data_loader.add_model("_D", mode="other_countries")
        c_eval_other_countries = CustomEvaluation(exp,
                                                  data_loader,
                                                  mode="other_countries",
                                                  keys=("_AROME", "_nn"),
                                                  other_models=("_D", "_A"))
        c_eval_other_countries.print_stats()
        del c_eval_other_countries
        """

        # Save figures
        cv = config["current_variable"]
        with timer_context("1-1 plots"):
            c_eval.plot_1_1_all(c_eval.df_results,
                                keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                name=f"1_1_all_{model}",
                                print_=True)
            c_eval.plot_1_1_by_station(c_eval.df_results,
                                       keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                       name=model,
                                       print_=True)
        with timer_context("Seasonal evolution"):
            c_eval.plot_seasonal_evolution(c_eval.df_results,
                                           keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                           name=f"Seasonal_evolution_{model}",
                                           print_=True)
            c_eval.plot_seasonal_evolution_by_station(c_eval.df_results,
                                                      keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                                      name=model,
                                                      print_=True)
        with timer_context("Lead time"):
            c_eval.plot_lead_time(c_eval.df_results,
                                  keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                  name=f"Lead_time_{model}",
                                  print_=True)
        with timer_context("Boxplot"):
            c_eval.plot_boxplot_topo_carac(c_eval.df_results,
                                           name=f"Boxplot_topo_carac_{model}",
                                           dict_keys={"_nn": "Neural Network + DEVINE",
                                                      "_AROME": "$AROME_{forecast}$",
                                                      "_D": "DEVINE",
                                                      "_A": "$AROME_{analysis}$"},
                                           print_=True)

        del c_eval
        gc.collect()

"""
    c_eval_other_countries.plot_1_1_all(c_eval_other_countries.df_results, c_eval_other_countries.keys)
    c_eval_other_countries.plot_1_1_by_station(c_eval_other_countries.df_results, c_eval_other_countries.keys)
    c_eval_other_countries.plot_seasonal_evolution(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
    c_eval_other_countries.plot_seasonal_evolution_by_station(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
    c_eval_other_countries.plot_lead_time(c_eval_other_countries.df_results, keys=c_eval_other_countries.keys)
"""

"""
    i = Interpretability(data_loader, cm, exp)
    with timer_context("plot_feature_importance"):
        i.plot_feature_importance("test", name=f"Feature_importance_{model}")
"""

"""
with timer_context("partial_dependence_plot"):
    i.plot_partial_dependence("test", name=f"Partial_dependence_plot_{model}")
"""

print_intro()
