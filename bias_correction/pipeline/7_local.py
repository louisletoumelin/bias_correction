import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
import os

# matplotlib.use('Agg')

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint
import gc

import sys

sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_local import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, Interpretability
from bias_correction.train.config_handler import PersistentConfig
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.utils_bc.print_functions import print_intro, print_headline

# Initialization
persistent_config = PersistentConfig(config)
if config["restore_experience"]:
    print_headline("Restore experience", "")
    exp, config = ExperienceManager.from_previous_experience(config["restore_experience"])
    cm = CustomModel.from_previous_experience(exp, config, "last")
else:
    print_headline("Create a new experience", "")
    exp = ExperienceManager(config)

config = persistent_config.restore_persistent_data(("quick_test", "quick_test_stations"), config)

print("\nCurrent experience:", flush=True)
print(exp.path_to_current_experience)

print("\nConfig", flush=True)
print(pprint(config), flush=True)


# Data
with timer_context("Prepare data"):
    data_loader = CustomDataHandler(config)
    data_loader.prepare_train_test_data()

if config["get_intermediate_output"] and not cm.has_intermediate_outputs:
    cm = CustomModel(exp, config)
    cm.build_model_with_strategy(print_=False)
    cm.load_weights()
    cm.model_version = "last"

"""
zip(["output_speed", "output_direction"],
                                          [("bias", "n_bias", "ae", "n_ae"), ("bias_direction", "abs_bias_direction")],
                                          [['vw10m(m/s)'], ['winddir(deg)']])
"""

for type_of_output, metrics, label in zip(["output_speed", "output_direction"],
                                          [("bias", "ae"), ("bias_direction", "abs_bias_direction")],
                                          [['vw10m(m/s)'], ['winddir(deg)']]):

    print_headline("Type of output", type_of_output)

    if cm.type_of_output != type_of_output:
        config_predict_speed = persistent_config.config_predict_parser(type_of_output, config)

        # Model
        cm = CustomModel(exp, config_predict_speed)
        cm.build_model_with_strategy(print_=False)
        cm.model.load_weights(cm.exp.path_to_last_model)
        cm.model_version = "last"
        print(cm)

    for model in ["last"]:  # "best"
        print_headline("Model", model)
        if type_of_output == "output_speed":
            cv = "UV"
        else:
            cv = "UV_DIR"
        # with tf.device('/GPU:0'):
        # Predict
        # with timer_context("Predict train set"):
        #    inputs_train = data_loader.get_tf_zipped_inputs(mode="train").batch(data_loader.length_train)
        #    results_train = cm.predict_single_bath(inputs_train, model_str=model)
        #    del inputs_train

        # Train
        # print(f"\n\n_______________________")
        # print(f"Train statistics: model={model}")
        # print(f"_______________________\n\n")
        # data_loader.set_predictions(results_train, mode="train")
        # c_eval_train = CustomEvaluation(exp, data_loader, mode="train", keys=("_AROME", "_nn"))
        # c_eval_train.print_stats()

        # Test
        print_headline("Test statistics", model)

        with timer_context("CustomEvaluation"):
            c_eval = CustomEvaluation(exp,
                                      data_loader,
                                      mode="test",
                                      keys=("_AROME", "_nn"),
                                      other_models=("_D", "_A"),
                                      metrics=metrics,
                                      key_df_results=cv)

        if type_of_output == "output_speed":
            with timer_context("Print statistics"):
                c_eval.print_means(keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'))
                mae, rmse, mbe, corr = c_eval.print_stats()
        else:
            c_eval.df2ae_dir(print_=True)

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

    if type_of_output == "output_speed":
        try:
            with timer_context("1-1 plots"):
                c_eval.plot_1_1_all(c_eval.df_results,
                                    keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                    color=("C1", "C0", "C2", "C3", "C4"),
                                    name=f"1_1_all_{model}_{cv}",
                                    plot_text=False,
                                    fontsize=20,
                                    density=True,
                                    xlabel="Observed wind speed \n[$m\:s^{-1}$]",
                                    ylabel="Modeled wind speed \n[$m\:s^{-1}$]",
                                    print_=True)
                c_eval.plot_1_1_by_station(c_eval.df_results,
                                           keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                           color=("C1", "C0", "C2", "C3", "C4"),
                                           name=f"{cv}_{model}",
                                           print_=True)
        except Exception as e:
            print(f"\nWARNING Exception for 1-1 plots: {e}", flush=True)

    if type_of_output == "output_direction":
        try:
            with timer_context("plot_wind_direction_all"):
                c_eval.plot_wind_direction_all(c_eval.df_results,
                                               keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                               metrics=("abs_bias_direction",),
                                               name=f"wind_direction_all",
                                               print_=True)
        except Exception as e:
            print(f"\nWARNING Exception for plot_wind_direction_all: {e}", flush=True)

        try:
            with timer_context("plot_wind_direction_all"):
                c_eval.plot_1_1_by_station(c_eval.df_results,
                                           keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                           name=f"{cv}_{model}",
                                           print_=True)
        except Exception as e:
            print(f"\nWARNING Exception for plot_wind_direction_all: {e}", flush=True)

    try:
        with timer_context("Seasonal evolution"):
            c_eval.plot_seasonal_evolution(c_eval.df_results,
                                           keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                           metrics=metrics,
                                           name=f"Seasonal_evolution_{model}_{cv}",
                                           print_=True)
            # c_eval.plot_seasonal_evolution_by_station(c_eval.df_results,
            #                                          keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
            #                                          metrics=metrics,
            #                                          name=f"{cv}_{model}",
            #                                          print_=True)

    except Exception as e:
        print(f"\nWARNING Exception for Seasonal: {e}", flush=True)

    try:
        with timer_context("Lead time"):
            c_eval.plot_lead_time(c_eval.df_results,
                                  keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                  color=("C1", "C0", "C2", "C3", "C4"),
                                  metrics=metrics,
                                  name=f"Lead_time_{model}_{cv}",
                                  print_=True,
                                  yerr=True)
    except Exception as e:
        print(f"\nWARNING Exception for Lead time: {e}", flush=True)
    try:
        c_eval.plot_lead_time_shadow(c_eval.df_results,
                                     metrics=metrics,
                                     list_x=('lead_time',),
                                     dict_keys={"_AROME": "$AROME_{forecast}$",
                                                "_D": "DEVINE",
                                                "_nn": "Neural Network + DEVINE",
                                                "_int": "Neural Network",
                                                "_A": "$AROME_{analysis}$",
                                                },
                                     figsize=(15, 10),
                                     name="LeadTimeCI",
                                     hue_order=("$AROME_{forecast}$",
                                                "DEVINE",
                                                "Neural Network + DEVINE",
                                                "$AROME_{analysis}$"),
                                     palette=("C1", "C0", "C2", "C4"),
                                     print_=False,
                                     fontsize=20)
    except Exception as e:
        print(f"\nWARNING Exception for Lead time: {e}", flush=True)

    try:
        with timer_context("Boxplot"):
            c_eval.plot_boxplot_topo_carac(c_eval.df_results,
                                           name=f"Boxplot_topo_carac_{model}_{cv}",
                                           dict_keys={"_nn": "Neural Network + DEVINE",
                                                      "_AROME": "$AROME_{forecast}$",
                                                      "_D": "DEVINE",
                                                      "_A": "$AROME_{analysis}$"},
                                           metrics=metrics,
                                           print_=True)

    except Exception as e:
        print(f"\nWARNING Exception for Boxplot: {e}", flush=True)

    """
    if cv == "UV":
        with timer_context("ALE plot"):
            config_predict_speed = persistent_config.config_predict_parser("output_speed", config)
            config_predict_speed["get_intermediate_output"] = False

            # Model
            cm = CustomModel(exp, config_predict_speed)
            cm.build_model_with_strategy(print_=False)
            cm.model.load_weights(cm.exp.path_to_last_model)
            cm.model_version = "last"
            print(cm)
    
            c_eval.plot_ale(cm,
                            data_loader,
                            ["alti",
                             "ZS",
                             "Tair",
                             "LWnet",
                             "SWnet",
                             "CC_cumul",
                             "BLH",
                             "tpi_500",
                             "curvature",
                             "mu",
                             "laplacian",
                             'Wind90',
                             'Wind87',
                             'Wind84',
                             'Wind75',
                             'aspect',
                             'tan(slope)',
                             "Wind",
                             "Wind_DIR"],
                            10,
                            monte_carlo=False,
                            rugplot_lim=1000,
                            cmap="viridis",
                            marker='x',
                            markersize=1,
                            linewidth=1)
    """
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
