import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')

logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint
import gc

import sys

sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_double_v1 import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, Interpretability
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.utils_bc.print_functions import print_intro, print_headline
from bias_correction.utils_bc.utils_config import detect_variable

# Initialization
get_intermediate_output = config["get_intermediate_output"]
quick_test = config["quick_test"]
quick_test_stations = config["quick_test_stations"]
initial_loss = config["loss"]

if config["restore_experience"]:
    print_headline("Restore experience", "")
    exp, config = ExperienceManager.from_previous_experience(config["restore_experience"])
    cm = CustomModel.from_previous_experience(exp, config, "last")
else:
    print_headline("Create a new experience", "")
    exp = ExperienceManager(config)

config["quick_test"] = quick_test
config["quick_test_stations"] = quick_test_stations

print("\nCurrent experience:", flush=True)
print(exp.path_to_current_experience)

print("\nConfig", flush=True)
print(pprint(config), flush=True)

if not config["restore_experience"]:
    #
    #
    # First fit
    #
    #
    config["labels"] = ['winddir(deg)']  # ["vw10m(m/s)"] or ["U_obs", "V_obs"] or ['T2m(degC)'] or ['winddir(deg)']
    config["type_of_output"] = "output_direction"
    config["loss"] = "cosine_distance"
    config["remove_null_speeds"] = True
    config["callbacks"].append("CSVLogger_dir")
    cm = CustomModel(exp, config)

    # Load inputs and outputs
    with timer_context("Prepare data"):
        data_loader = CustomDataHandler(config)
        data_loader.prepare_train_test_data()

    print_headline("Launch training direction", "")
    print(config["labels"], config["type_of_output"], config["loss"])
    print(config["input_variables"])

    # We don't fit a model with two outputs because it cause error in the loss function
    config["get_intermediate_output"] = False
    data_loader.config["get_intermediate_output"] = False
    cm.config["get_intermediate_output"] = False
    exp.config["get_intermediate_output"] = False

    cm.build_model_with_strategy()

    for layer in cm.model.layers:
        # selecting layer by name
        if "speed_ann" in layer.name:  # "speed_ann"
            cm.model.get_layer(layer.name).trainable = False
        if "dir_ann" in layer.name:  # "speed_ann"
            cm.model.get_layer(layer.name).trainable = True

    with tf.device('/GPU:0'), timer_context("fit"):
        _ = cm.fit_with_strategy(data_loader.get_batched_inputs_labels(mode="train"),
                                 dataloader=data_loader,
                                 mode_callback="train")
        exp.save_model(cm)
    #
    #
    # Second fit
    #
    #
    config["labels"] = ["vw10m(m/s)"]  # ["vw10m(m/s)"] or ["U_obs", "V_obs"] or ['T2m(degC)'] or ['winddir(deg)']
    config["type_of_output"] = "output_speed"
    config["loss"] = initial_loss
    config["remove_null_speeds"] = False
    config["callbacks"].remove("CSVLogger_dir")
    config["callbacks"].remove("CSVLogger")

    # Load inputs and outputs
    with timer_context("Prepare data"):
        data_loader = CustomDataHandler(config)
        data_loader.prepare_train_test_data()

    cm = CustomModel(exp, config)
    print_headline("Launch training speed", "")
    print(config["labels"], config["type_of_output"], config["loss"])

    # We don't fit a model with two outputs because it cause error in the loss function
    config["get_intermediate_output"] = False
    data_loader.config["get_intermediate_output"] = False
    cm.config["get_intermediate_output"] = False
    exp.config["get_intermediate_output"] = False

    cm.build_model_with_strategy()

    print("Launch load_weights")
    cm.model.load_weights(cm.exp.path_to_last_model)
    print(f"Restore weights from {cm.exp.path_to_last_model}")
    cm.model_version = "last"

    with timer_context("Detect layers to freeze"):
        for layer in cm.model.layers:
            # selecting layer by name
            if "speed_ann" in layer.name or "speed_cnn" in layer.name:  # "speed_ann"
                cm.model.get_layer(layer.name).trainable = True
                print(f"Training == True on: {layer.name}")
            if "dir_ann" in layer.name or "dir_cnn" in layer.name:  # "speed_ann"
                cm.model.get_layer(layer.name).trainable = False
                print(f"Freezing: {layer.name}")

    with tf.device('/GPU:0'), timer_context("fit"):
        _ = cm.fit_with_strategy(data_loader.get_batched_inputs_labels(mode="train"),
                                 dataloader=data_loader,
                                 mode_callback="train")
        exp.save_model(cm)
else:
    print_headline("No training", "")
    with timer_context("Prepare data"):
        data_loader = CustomDataHandler(config)
        data_loader.prepare_train_test_data()

config["get_intermediate_output"] = get_intermediate_output
exp.config["get_intermediate_output"] = get_intermediate_output
cm.config["get_intermediate_output"] = get_intermediate_output
data_loader.config["get_intermediate_output"] = get_intermediate_output
if config["get_intermediate_output"]:
    cm.build_model_with_strategy(print_=False)
    print("Launch load_weights")
    cm.model.load_weights(cm.exp.path_to_last_model)
    print(f"Restore weights from {cm.exp.path_to_last_model}")
    cm.model_version = "last"

exp.save_all(data_loader, cm)
"""
zip(["output_speed", "output_direction"],
                                          [("bias", "n_bias", "ae", "n_ae"), ("bias_direction", "abs_bias_direction")],
                                          [['vw10m(m/s)'], ['winddir(deg)']])
"""
for type_of_output, metrics, label in zip(["output_direction"],
                                          [("bias_direction", "abs_bias_direction")],
                                          [['winddir(deg)']]):

    print_headline("Type of output", type_of_output)

    if config["type_of_output"] != type_of_output:
        config["type_of_output"] = type_of_output
        config = detect_variable(config)

        current_var = config["current_variable"]
        exp.config["current_variable"] = current_var
        cm.config["current_variable"] = current_var
        data_loader.config["current_variable"] = current_var

        config["labels"] = label  # ["vw10m(m/s)"]
        exp.config["labels"] = label
        cm.config["labels"] = label
        data_loader.config["labels"] = label

        if type_of_output == "output_speed":
            remove_null_speeds = False
        else:
            remove_null_speeds = True

        config["remove_null_speeds"] = remove_null_speeds
        exp.config["remove_null_speeds"] = remove_null_speeds
        cm.config["remove_null_speeds"] = remove_null_speeds
        # Load inputs and outputs
        with timer_context("Prepare data"):
            data_loader = CustomDataHandler(config)
            data_loader.prepare_train_test_data()

        cm.build_model_with_strategy(print_=False)
        cm.model.load_weights(cm.exp.path_to_last_model)
        cm.model_version = "last"

    cv = config["current_variable"]
    for model in ["last"]:  # "best"
        print_headline("Model", model)

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

        # Predict
        with tf.device('/GPU:0'), timer_context("Predict test set"):
            inputs_test = data_loader.get_tf_zipped_inputs(mode="test").batch(data_loader.length_test)
            results_test = cm.predict_single_bath(inputs_test, model_version=model)
            del inputs_test

        # Test
        print_headline("Test statistics", model)

        with timer_context("set_predictions"):
            data_loader.set_predictions(results_test, mode="test")
        del results_test

        with timer_context("add_model"):
            data_loader.add_model("_D", mode="test")
            data_loader.add_model("_A")

        with timer_context("CustomEvaluation"):
            c_eval = CustomEvaluation(exp,
                                      data_loader,
                                      mode="test",
                                      keys=("_AROME", "_nn"),
                                      other_models=("_D", "_A"),
                                      metrics=metrics)

        if type_of_output == "output_speed":
            with timer_context("Print statistics"):
                print("\nMean observations:", flush=True)
                print(c_eval.df_results[f"{cv}_obs"].mean(), flush=True)
                print("\nMean AROME", flush=True)
                print(c_eval.df_results[f"{cv}_AROME"].mean(), flush=True)
                print("\nMean int", flush=True)
                print(c_eval.df_results[f"{cv}_int"].mean(), flush=True)
                print("\nMean NN", flush=True)
                print(c_eval.df_results[f"{cv}_nn"].mean(), flush=True)
                print("\nMean D", flush=True)
                print(c_eval.df_results[f"{cv}_D"].mean(), flush=True)
                print("\nMean A", flush=True)
                print(c_eval.df_results[f"{cv}_A"].mean(), flush=True)
                mae, rmse, mbe, corr = c_eval.print_stats()

            exp.save_results(c_eval, mae, rmse, mbe, corr)
            exp.save_metrics_current_experience((mae, rmse, mbe, corr),
                                                ("mae", "rmse", "mbe", "corr"),
                                                keys=tuple(['_' + key.split('_')[-1] for key in c_eval.keys]))
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

    # Save figures
    """
    if cv == "UV":
        with timer_context("ALE plot"):
            c_eval.plot_ale(cm,
                            data_loader,
                            ['Wind', 'Wind_DIR'],
                            10,
                            monte_carlo=False,
                            rugplot_lim=1000,
                            cmap="viridis",
                            marker='x',
                            markersize=1,
                            linewidth=1)
    """

    if cv == "UV":
        try:
            with timer_context("1-1 plots"):
                c_eval.plot_1_1_all(c_eval.df_results,
                                    keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                    name=f"1_1_all_{model}_{cv}",
                                    print_=True)
                c_eval.plot_1_1_by_station(c_eval.df_results,
                                           keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                           name=f"{cv}_{model}",
                                           print_=True)
        except Exception as e:
            print(f"\nWARNING Exception for 1-1 plots: {e}", flush=True)

    if cv == "UV_DIR":
        try:
            with timer_context("plot_wind_direction_all"):
                c_eval.plot_wind_direction_all(c_eval.df_results,
                                               keys=(f'{cv}_AROME', f'{cv}_D', f'{cv}_nn', f'{cv}_int', f'{cv}_A'),
                                               metrics=("abs_bias_direction",),
                                               name=f"wind_direction_all",
                                               print_=True)
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
                                  metrics=metrics,
                                  name=f"Lead_time_{model}_{cv}",
                                  print_=True)
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
