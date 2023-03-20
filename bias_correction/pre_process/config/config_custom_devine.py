from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables, adapt_distribution_strategy_to_available_devices, init_learning_rate_adapted, detect_variable
from bias_correction.config._config import config

# Architecture
config["details"] = "new_test"                                          # Str. Some details about the experiment
config["global_architecture"] = "devine_only"                                # Str. Default="ann_v0", "dense_only", "dense_temperature", "devine_only"
config["restore_experience"] = False
config["type_of_output"] = "map_u_v_w"                               # Str. "output_speed" or "output_components", "map", "map_components", "map_u_v_w"
config["custom_unet"] = True
config["custom_input_shape"] = (90, 88, 1)
config["disable_training_cnn"] = True
config["sliding_mean"] = True

# General
config["distribution_strategy"] = None                                  # "MirroredStrategy", "Horovod" or None
config["prefetch"] = "auto"                                                # Default="auto", else = Int

config["input_variables"] = ['Wind', 'Wind_DIR']
config["wind_nwp_variables"] = ["Wind", "Wind_DIR"]                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]
config["get_intermediate_output"] = False
config["standardize"] = False
config["labels"] = []
config["batch_size"] = 32
config["initializer"] = None
config["args_initializer"] = []
config["kwargs_initializer"] = {}
config["loss"] = "mse"
config["args_loss"] = {"mse": []}
config["kwargs_loss"] = {"mse": {}}
config["optimizer"] = "Adam"
config["learning_rate"] = 0.01
config["args_optimizer"] = [config["learning_rate"]]  # List.
config["kwargs_optimizer"] = {}  # Dict.

# Do not modify: assert inputs are correct
config = adapt_distribution_strategy_to_available_devices(config)
config = init_learning_rate_adapted(config)
config["nb_input_variables"] = len(config["input_variables"])
config = detect_variable(config)
