from bias_correction.utils_bc.network import detect_network

config = dict(
    name_nwp="AROME_analysis",
    name_dem="DEM",
    resolution_dem=30,
    verbose=True,
    number_of_neighbors=1,
    nb_pixel_topo_x=140,
    nb_pixel_topo_y=140,
)

# Network
config["network"] = detect_network()

# Path root
if config["network"] == "local":
    config["path_root"] = "/home/letoumelinl/bias_correction/"
else:
    config["path_root"] = "//scratch/mrmn/letoumelinl/bias_correction/"

# Path
config["folder_obs"] = "2022_06_10"
config["path_station"] = config["path_root"] + f"Data/1_Raw/Observation/stations/{config['folder_obs']}/"
config["path_time_series"] = config["path_root"] + f"Data/1_Raw/Observation/time_series/{config['folder_obs']}/"
config["path_dem"] = config["path_root"] + "Data/1_Raw/DEM/"
config["path_nwp_alp"] = config["path_root"] + "Data/1_Raw/AROME_analysis/alp/month/"
config["path_nwp_france"] = config["path_nwp_alp"]
config["path_nwp_swiss"] = config["path_root"] + "Data/1_Raw/AROME_analysis/swiss/month/"
config["path_nwp_pyr"] = config["path_root"] + "Data/1_Raw/AROME_analysis/pyr/month/"
config["path_nwp_corse"] = config["path_root"] + "Data/1_Raw/AROME_analysis/corse/month/"
config["path_X_Y_L93_alp"] = config["path_root"] + "Data/1_Raw/AROME_analysis/alp/X_Y_L93/"
config["path_X_Y_L93_france"] = config["path_X_Y_L93_alp"]
config["path_X_Y_L93_swiss"] = config["path_root"] + "Data/1_Raw/AROME_analysis/swiss/X_Y_L93/"
config["path_X_Y_L93_pyr"] = config["path_root"] + "Data/1_Raw/AROME_analysis/pyr/X_Y_L93/"
config["path_X_Y_L93_corse"] = config["path_root"] + "Data/1_Raw/AROME_analysis/corse/X_Y_L93/"

# Output paths
config["path_stations_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/stations/{config['folder_obs']}/"
config["path_time_series_pre_processed"] = config[
                                               "path_root"] + f"Data/2_Pre_processed/time_series/{config['folder_obs']}/"
config["path_topos_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/topos/{config['folder_obs']}/"

# Path downscale
if config["network"] == "local":
    config["path_module_downscale"] = "/home/letoumelinl/wind_downscaling_cnn/src/downscale_/"
else:
    config["path_module_downscale"] = "//home/mrmn/letoumelinl/downscale_/"

# Variables
config["variables_nwp"] = ["Tair", "T1", "ts", "Tmin", "Tmax", "Qair", "Q1", "RH2m", "Wind", "Wind_Gust",
                           "Wind_DIR", "PSurf", "ZS", "BLH", "Wind90", "Wind87", "Wind84", "Wind75", "TT90",
                           "TT87", "TT84", "TT75"]

config["variables_time_series"] = ['date', 'name', 'lon', 'lat', 'alti', 'T2m(degC)', 'vw10m(m/s)', 'winddir(deg)',
                                   'HTN(cm)'] + config["variables_nwp"] + ["U_obs", "V_obs", "U_AROME", "V_AROME"]

# Preprocess
config["pre_process_stations"] = False
config["pre_process_nwp"] = False
config["pre_process_time_series"] = True
config["pre_process_topos"] = False
