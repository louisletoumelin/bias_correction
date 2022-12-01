from bias_correction.utils_bc.network import detect_network

config = dict(
    name_nwp="AROME",
    name_dem="DEM",
    resolution_dem=30,
    interp=2,
    method="linear",
    verbose=True,
    number_of_neighbors=4,
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
config["path_nwp_alp"] = config["path_root"] + "Data/1_Raw/AROME/alp/month/"
config["path_nwp_swiss"] = config["path_root"] + "Data/1_Raw/AROME/swiss/month/"
config["path_nwp_pyr"] = config["path_root"] + "Data/1_Raw/AROME/pyr/month/"
config["path_nwp_corse"] = config["path_root"] + "Data/1_Raw/AROME/corse/month/"
config["path_X_Y_L93_alp"] = config["path_root"] + "Data/1_Raw/AROME/alp/X_Y_L93/"
config["path_X_Y_L93_swiss"] = config["path_root"] + "Data/1_Raw/AROME/swiss/X_Y_L93/"
config["path_X_Y_L93_pyr"] = config["path_root"] + "Data/1_Raw/AROME/pyr/X_Y_L93/"
config["path_X_Y_L93_corse"] = config["path_root"] + "Data/1_Raw/AROME/corse/X_Y_L93/"

# Output paths
config["path_stations_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/stations/{config['folder_obs']}/"
config["path_time_series_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/time_series/{config['folder_obs']}/"
config["path_topos_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/topos/{config['folder_obs']}/"

# Path downscale
if config["network"] == "local":
    config["path_module_downscale"] = "/home/letoumelinl/wind_downscaling_cnn/src/downscale_/"
else:
    config["path_module_downscale"] = "//home/mrmn/letoumelinl/downscale_/"

# Variables
config["variables_nwp"] = ['Tair', 'T1', 'ts', 'Tmin', 'Tmax', 'Qair', 'Q1', 'RH2m', 'Wind_Gust', 'PSurf', 'ZS',
                     'BLH', 'Rainf', 'Snowf', 'LWdown', 'LWnet', 'DIR_SWdown', 'SCA_SWdown', 'SWnet', 'SWD',
                     'SWU', 'LHF', 'SHF', 'CC_cumul', 'CC_cumul_low', 'CC_cumul_middle', 'CC_cumul_high',
                     'Wind90', 'Wind87', 'Wind84', 'Wind75', 'TKE90', 'TKE87', 'TKE84', 'TKE75', 'TT90', 'TT87',
                     'TT84', 'TT75', 'SWE', 'snow_density', 'snow_albedo', 'vegetation_fraction', 'Wind', 'Wind_DIR']

config["variables_time_series"] = ['date', 'name', 'lon', 'lat', 'alti', 'T2m(degC)', 'vw10m(m/s)', 'winddir(deg)',
                                   'HTN(cm)'] + config["variables_nwp"] + ["U_obs", "V_obs", "U_AROME", "V_AROME"]

# Preprocess
config["pre_process_stations"] = False
config["pre_process_nwp"] = False
config["pre_process_time_series"] = False
config["pre_process_topos"] = True
