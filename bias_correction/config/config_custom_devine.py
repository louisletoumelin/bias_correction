from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables, adapt_distribution_strategy_to_available_devices, init_learning_rate_adapted, detect_variable
from bias_correction.config._config import config

# Architecture
config["details"] = "new_test"                                          # Str. Some details about the experiment
config["global_architecture"] = "devine_only"                                # Str. Default="ann_v0", "dense_only", "dense_temperature", "devine_only"
config["restore_experience"] = False
config["type_of_output"] = "map_u_v_w"                               # Str. "output_speed" or "output_components", "map", "map_components", "map_u_v_w"

# General
config["distribution_strategy"] = None                                  # "MirroredStrategy", "Horovod" or None
config["prefetch"] = "auto"                                                # Default="auto", else = Int

# Input variables
config["input_variables"] = ['Wind', 'Wind_DIR']
config["wind_nwp_variables"] = ["Wind", "Wind_DIR"]                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]

# Do not modify: assert inputs are correct
config = adapt_distribution_strategy_to_available_devices(config)
config = init_learning_rate_adapted(config)
config["nb_input_variables"] = len(config["input_variables"])
config = detect_variable(config)
