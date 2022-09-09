from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables, adapt_distribution_strategy_to_available_devices, init_learning_rate_adapted, detect_variable
from bias_correction.config._config import config

# Architecture
config["details"] = "new_test"                                          # Str. Some details about the experiment
config["global_architecture"] = "devine_only"                                # Str. Default="ann_v0", "dense_only", "dense_temperature", "devine_only"
config["restore_experience"] = False
config["type_of_output"] = "output_speed"                               # Str. "output_speed" or "output_components", "map", "map_components"

# General
config["distribution_strategy"] = None                                  # "MirroredStrategy", "Horovod" or None
config["prefetch"] = "auto"                                                # Default="auto", else = Int

# Quick test
config["quick_test"] = False                                             # Bool. Quicktest case (fast training)
config["quick_test_stations"] = ["ALPE-D'HUEZ", 'LES ECRINS-NIVOSE', 'SOUM COUY-NIVOSE', 'SPONDE-NIVOSE']

# Input variables
config["input_variables"] = ['Wind', 'Wind_DIR']
config["labels"] = ['vw10m(m/s)']                                       # ["vw10m(m/s)"] or ["U_obs", "V_obs"] or ['T2m(degC)']
config["wind_nwp_variables"] = ["Wind", "Wind_DIR"]                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]

# Split
config["split_strategy_test"] = "time_and_space"                         # "time", "space", "time_and_space", "random"
config["split_strategy_val"] = "time_and_space"

# Random split
config["random_split_state_test"] = 50                                  # Float. Default = 50
config["random_split_state_val"] = 55                                   # Float. Default = 55

# Time split
config["date_split_train_test"] = "2019-10-01"                          # Str. Split train/test around this date
#                                                                         e.g. "2018-11-01"
config["date_split_train_val"] = "2019-10-01"

# Space split
# 2022_08_04 v4
config["stations_test"] = ['Col du Lac Blanc', 'GOE', 'WAE', 'TGKAL', 'LAG', 'AND', 'CHU', 'SMM', 'ULR', 'WFJ', 'TICAM', 'SCM', 'MMMEL', 'INNRED', 'MMBIR', 'MMHIW', 'MMLOP', 'TGALL', 'GAP', 'BAS', 'STK', 'PLF', 'MVE', 'SAG', 'MLS', 'MAR', 'MTE', 'MTR', 'CHZ', 'SIA', 'COV', 'MMSTA', 'BIV', 'ANT', 'TGDIE', 'CHM', 'TGARE', 'TALLARD', 'LE CHEVRIL-NIVOSE', 'GOR', 'MMMUE', 'INT', 'BIE', 'EIN', 'RUE', 'QUI', 'NEU', 'MMNOI', 'LE GUA-NIVOSE', 'GIH', 'AEG', 'MOE', 'LUG', 'TGNUS', 'BEH']
config["stations_val"] = ['WYN', 'BER', 'ARH', 'ELM', 'MMMES', 'ASCROS', 'GRANDE PAREI NIVOSE', 'SAM', 'JUN', 'SCU', 'MMSVG', 'GALIBIER-NIVOSE', 'MEYTHET', 'BRZ', 'OBR', 'FAH', 'MMRIG', 'PMA']
config["stations_to_reject"] = ["Vallot", "Dome Lac Blanc", "MFOKFP"]

# Do not modify: assert inputs are correct
config = adapt_distribution_strategy_to_available_devices(config)
config = init_learning_rate_adapted(config)
config["nb_input_variables"] = len(config["input_variables"])
config = detect_variable(config)
