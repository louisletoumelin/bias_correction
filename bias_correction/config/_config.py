from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables, adapt_distribution_strategy_to_available_devices, init_learning_rate_adapted, detect_variable

config = {"network": detect_network()}

# Path root
if config["network"] == "local":
    config["path_root"] = "/home/letoumelinl/bias_correction/"
else:
    config["path_root"] = "//scratch/mrmn/letoumelinl/bias_correction/"

# Path downscale
if config["network"] == "local":
    config["path_module_downscale"] = "/home/letoumelinl/wind_downscaling_cnn/src/downscale_/"
else:
    config["path_module_downscale"] = "//home/mrmn/letoumelinl/downscale_/"

# Path to CNN
config["cnn_name"] = "date_21_12_2021_name_simu_classic_all_low_epochs_0_model_UNet/"
config["path_experience"] = config["path_root"] + "Data/1_Raw/CNN/"
config["unet_path"] = config["path_experience"] + config["cnn_name"]

# Input data
# Path
config["folder_obs"] = "2022_06_10"
config["path_stations_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/stations/{config['folder_obs']}/"
config["path_time_series_pre_processed"] = config[
                                               "path_root"] + f"Data/2_Pre_processed/time_series/{config['folder_obs']}/"
config["path_topos_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/topos/{config['folder_obs']}/"
config["path_experiences"] = config["path_root"] + "Data/3_Predictions/Experiences/"
config["path_to_devine"] = config["path_root"] + "Data/3_Predictions/DEVINE/"
config["path_to_analysis"] = config["path_root"] + "Data/2_Pre_processed/AROME_analysis/"
config["path_to_topographic_parameters"] = config["path_root"] + "Data/1_Raw/topographic_parameters/"

# Input data
# Filename
config["time_series"] = config["path_time_series_pre_processed"] + "time_series_bc.pkl"
config["time_series_int"] = config["path_time_series_pre_processed"] + "time_series_bc_interpolated.pkl"
config["stations"] = config["path_stations_pre_processed"] + "stations_bc.pkl"
config["topos_near_station"] = config["path_topos_pre_processed"] + "dict_topo_near_station_2022_10_26.pickle"
config["topos_near_nwp"] = config["path_topos_pre_processed"] + "dict_topo_near_nwp.pickle"
config["topos_near_nwp_int"] = config["path_topos_pre_processed"] + "dict_topo_near_nwp_inter.pickle"

config["aspect_near_station"] = config["path_topos_pre_processed"] + "dict_aspect_near_station_2022_10_26.pickle"
config["tan_slope_near_station"] = config["path_topos_pre_processed"] + "dict_tan_slope_near_station_2022_10_26.pickle"
config["tpi_300_near_station"] = config["path_topos_pre_processed"] + "dict_tpi_300_near_station_2022_10_26.pickle"
config["tpi_600_near_station"] = config["path_topos_pre_processed"] + "dict_tpi_600_near_station_2022_10_26.pickle"
