from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables, adapt_distribution_strategy_to_available_devices, init_learning_rate_adapted, detect_variable


config = {}
config["network"] = detect_network()

"""
├── ./Data
│   │
│   ├── ./Data/1_Raw
│   │   │
│   │   ├── ./Data/1_Raw/AROME
│   │   │   └── ...
│   │   │
│   │   ├── ./Data/1_Raw/CNN
│   │   │   └── ./Data/1_Raw/CNN/date_21_12_2021_name_simu_classic_all_low_epochs_0_model_UNet
│   │   │
│   │   ├── ./Data/1_Raw/DEM
│   │   │   └── ...
│   │   │
│   │   │── ./Data/1_Raw/Observation
│   │   └── └── ...
│   │
│   ├── ./Data/2_Pre_processed
│   │   │
│   │   ├── ./Data/2_Pre_processed/stations
│   │   │   └── ./Data/2_Pre_processed/stations/2022_03_25
│   │   ├── ./Data/2_Pre_processed/time_series
│   │   │   └── ./Data/2_Pre_processed/time_series/2022_03_25
│   │   └── ./Data/2_Pre_processed/topos
│   │       └── ./Data/2_Pre_processed/topos/2022_03_25
│   │
│   └── ./Data/3_Predictions
│       │
│       ├── ./Data/3_Predictions/Experiences
│       │   ├── ./Data/3_Predictions/Experiences/2022_5_4_local_v0
│       │   ├── ./Data/3_Predictions/Experiences/2022_5_4_local_v1
│       │   ├── ./Data/3_Predictions/Experiences/2022_5_4_local_v2
│       │
│       └── ./Data/3_Predictions/Model
"""

# Path root
if config["network"] == "local":
    config["path_root"] = "/home/letoumelinl/bias_correction/"
else:
    config["path_root"] = "//scratch/mrmn/letoumelinl/bias_correction/"

# Path to CNN
config["cnn_name"] = "date_21_12_2021_name_simu_classic_all_low_epochs_0_model_UNet/"
config["path_experience"] = config["path_root"] + "Data/1_Raw/CNN/"
config["model_path"] = config["path_experience"] + config["cnn_name"]

# Path to inputs and labels data
config["folder_obs"] = "2022_03_25"
config["path_stations_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/stations/{config['folder_obs']}/"
config["path_time_series_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/time_series/{config['folder_obs']}/"
config["path_topos_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/topos/{config['folder_obs']}/"
config["path_experiences"] = config["path_root"] + "Data/3_Predictions/Experiences/"

# Filename inputs and labels data
config["time_series"] = config["path_time_series_pre_processed"] + "time_series_bc.pkl"
config["time_series_int"] = config["path_time_series_pre_processed"] + "time_series_bc_interpolated.pkl"
config["stations"] = config["path_stations_pre_processed"] + "stations_bc.pkl"
config["topos_near_station"] = config["path_topos_pre_processed"] + "dict_topo_near_station.pickle"
config["topos_near_nwp"] = config["path_topos_pre_processed"] + "dict_topo_near_nwp.pickle"
config["topos_near_nwp_int"] = config["path_topos_pre_processed"] + "dict_topo_near_nwp_inter.pickle"

# Architecture
config["details"] = "t_power"                                          # Str. Some details about the experiment
config["global_architecture"] = "dense_temperature"                 # Str. Default="ann_v0", "dense_only", "dense_temperature"

# ann_v0
config["disable_training_cnn"] = True                                   # Bool. Default=True
config["type_of_output"] = "output_speed"                               # Str. "output_speed" or "output_components"
config["nb_units"] = [15]                                               # List. Each member is a unit [40, 20, 10, 5]
config["use_bias"] = True

# General
config["batch_normalization"] = False                                   # Bool. Apply batch_norm or not
config["activation_dense"] = "selu"                                     # Bool. Activation in dense network
config["dropout_rate"] = 0.25                                            # Int. or False. Dropout rate or no dropout
config["final_skip_connection"] = True                                        # Use skip connection with speed/direction
config["distribution_strategy"] = None                                  # "MirroredStrategy", "Horovod" or None
config["prefetch"] = "auto"                                                # Default="auto", else = Int

# Skip connections in dense network
config["dense_with_skip_connection"] = False

# Hyperparameters
config["batch_size"] = 32                                                # Int.
config["epochs"] = 50                                                    # Int.
config["learning_rate"] = 0.001

# Optimizer
config["optimizer"] = "RMSprop"                                         # Str.
config["args_optimizer"] = [config["learning_rate"]]                                        # List.
config["kwargs_optimizer"] = {}                                         # Dict.

# Initializer
config["initializer"] = "GlorotUniform"                                 # Str. Default = "GlorotUniform"
config["args_initializer"] = []                                         # List.
config["kwargs_initializer"] = {"seed": 42}                             # Dict.

# Input CNN
config["input_cnn"] = False

# Inputs pre-processing
config["standardize"] = True                                            # Bool. Apply standardization
config["shuffle"] = True                                                # Bool. Shuffle inputs

# Quick test
config["quick_test"] = False                                             # Bool. Quicktest case (fast training)
config["quick_test_stations"] = ["ALPE-D'HUEZ", 'LA MURE- RADOME', 'LES ECRINS-NIVOSE']

# Input variables
config["input_variables"] = ["tpi_500", "curvature", "mu", "laplacian", 'ZS', 'Wind', 'Wind_DIR', 'TT90', 'TT75',
                             'Wind75', "LWnet", "SWnet", 'CC_cumul', 'BLH', 'lat', 'lon', 'alti']

# Labels
config["labels"] = ['T2m(degC)']                                       # ["vw10m(m/s)"] or ["U_obs", "V_obs"] or ['T2m(degC)']
config["wind_nwp_variables"] = ["Wind", "Wind_DIR"]                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]
config["wind_temp_variables"] = ['Tair']                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]

# Dataset
config["unbalanced_dataset"] = False
config["unbalanced_threshold"] = 5

# Callbacks
config["callbacks"] = ["TensorBoard", "ReduceLROnPlateau", "EarlyStopping", "CSVLogger", "ModelCheckpoint"]
config["kwargs_callbacks"] = {"ReduceLROnPlateau": {"monitor": "val_loss",
                                                    "factor": 0.5,      # new_lr = lr * factor
                                                    "patience": 3,
                                                    "min_lr": 0.0001},

                              "EarlyStopping": {"monitor": "val_loss",
                                                "min_delta": 0.01,
                                                "patience": 5,
                                                "mode": "min",
                                                "restore_best_weights": False},

                              "ModelCheckpoint": {"min_delta": 0.01,
                                                  "monitor": "val_loss",
                                                  "save_best_only": True,
                                                  "save_weights_only": False,
                                                  "mode": "min",
                                                  "save_freq": "epoch"},

                              "TensorBoard": {"profile_batch": '20, 50',
                                              "histogram_freq": 1},

                              "LearningRateWarmupCallback": {"warmup_epochs": 5,
                                                                "verbose": 1}
                              }

# Split
config["split_strategy_test"] = "time_and_space"                         # "time", "space", "time_and_space", "random"
config["split_strategy_val"] = "time_and_space"

# Random split
config["parameters_split_test"] = ["alti", "tpi_500_NN_0", "mu_NN_0", "laplacian_NN_0", "Y", "X"]
config["parameters_split_val"] = ["alti", "tpi_500_NN_0"]
config["country_to_reject_during_training"] = ["pyr", "corse"]
config["metric_split"] = "rmse"
config["random_split_state_test"] = 50                                  # Float. Default = 50
config["random_split_state_val"] = 55                                   # Float. Default = 55

# Time split
config["date_split_train_test"] = "2018-11-01"                          # Str. Split train/test around this date
#                                                                         e.g. "2018-11-01"
config["date_split_train_val"] = "2018-11-01"

# Space split
config["stations_test"] = ['AIGUILLES ROUGES-NIVOSE', 'LE GRAND-BORNAND', 'MEYTHET', 'LE PLENAY', 'Saint-Sorlin',
                           'Argentiere', 'Col du Lac Blanc', 'CHA', 'CMA', 'DOL', "GALIBIER-NIVOSE", 'LA MURE-ARGENS',
                           'ARVIEUX', 'PARPAILLON-NIVOSE', 'EMBRUN', 'LA FAURIE', 'GAP', 'LA MEIJE-NIVOSE',
                           'COL AGNEL-NIVOSE', 'HOE', 'TGILL', 'INT', 'EGO', 'EIN', 'ELM', 'MMERZ', 'INNESF',
                           'EVO', 'FAH', 'FLU', 'MMFRS', 'TGFRA', 'GRA', 'FRU', 'MFOFKP', 'GVE', 'GES', 'GIH']
config["stations_val"] = ['La Muzelle Lac Blanc', 'MMKSB', 'KOP', 'TGKRE', 'ALBERTVILLE JO', 'BONNEVAL-NIVOSE', 'SHA',
                          'SRS', 'SCM', 'WAE', 'TGWEI', 'WFJ', 'KAWEG', 'WYN', 'ZER', 'MMZNZ', 'MMZOZ',
                          'MMZWE', 'REH', 'SMA', 'KLO']

# Intermediate output
config["get_intermediate_output"] = False

# Custom loss
config["loss"] = "mse_power"                                                  # Str. Default=mse. Used for gradient descent
config["args_loss"] = []
config["kwargs_loss"] = {"penalized_mse": {"penalty": 5,
                                           "speed_threshold": 7},
                         "mse_proportional": {"penalty": 1},
                         "mse_power": {"penalty": 1,
                                       "power": 2}}

# todo assert station validation not in station test
# todo assert kwargs split strategy are defined
# todo assert a seed is given for any random input
# todo extract DEVINE predictions


# Do not modify: assert inputs are correct
config = assert_input_for_skip_connection(config)
config = sort_input_variables(config)
config = adapt_distribution_strategy_to_available_devices(config)
config = init_learning_rate_adapted(config)
config["nb_input_variables"] = len(config["input_variables"])
config = detect_variable(config)

list_variables = ['name', 'date', 'lon', 'lat', 'alti', 'T2m(degC)', 'vw10m(m/s)',
                  'winddir(deg)', 'HTN(cm)', 'Tair', 'T1', 'ts', 'Tmin', 'Tmax', 'Qair',
                  'Q1', 'RH2m', 'Wind_Gust', 'PSurf', 'ZS', 'BLH', 'Rainf', 'Snowf',
                  'LWdown', 'LWnet', 'DIR_SWdown', 'SCA_SWdown', 'SWnet', 'SWD', 'SWU',
                  'LHF', 'SHF', 'CC_cumul', 'CC_cumul_low', 'CC_cumul_middle',
                  'CC_cumul_high', 'Wind90', 'Wind87', 'Wind84', 'Wind75', 'TKE90',
                  'TKE87', 'TKE84', 'TKE75', 'TT90', 'TT87', 'TT84', 'TT75', 'SWE',
                  'snow_density', 'snow_albedo', 'vegetation_fraction', 'Wind',
                  'Wind_DIR', 'U_obs', 'V_obs', 'U_AROME', 'V_AROME']

all_stations = ['BARCELONNETTE', 'DIGNE LES BAINS', 'RESTEFOND-NIVOSE',
       'LA MURE-ARGENS', 'ARVIEUX', 'PARPAILLON-NIVOSE', 'EMBRUN',
       'LA FAURIE', 'GAP', 'LA MEIJE-NIVOSE', 'COL AGNEL-NIVOSE',
       'GALIBIER-NIVOSE', 'ORCIERES-NIVOSE', 'RISTOLAS',
       'ST JEAN-ST-NICOLAS', 'TALLARD', "VILLAR D'ARENE",
       'VILLAR ST PANCRACE', 'ASCROS', 'PEIRA CAVA', 'PEONE',
       'MILLEFONTS-NIVOSE', 'CHAPELLE-EN-VER', 'LUS L CROIX HTE',
       'ST ROMAN-DIOIS', 'AIGLETON-NIVOSE', 'CREYS-MALVILLE',
       'LE GUA-NIVOSE', "ALPE-D'HUEZ", 'LA MURE- RADOME',
       'LES ECRINS-NIVOSE', 'GRENOBLE-ST GEOIRS', 'ST HILAIRE-NIVOSE',
       'ST-PIERRE-LES EGAUX', 'GRENOBLE - LVD', 'VILLARD-DE-LANS',
       'CHAMROUSSE', 'ALBERTVILLE JO', 'BONNEVAL-NIVOSE', 'MONT DU CHAT',
       'BELLECOTE-NIVOSE', 'GRANDE PAREI NIVOSE', 'COL-DES-SAISIES',
       'ALLANT-NIVOSE', 'LA MASSE', 'LE CHEVRIL-NIVOSE',
       'LES ROCHILLES-NIVOSE', 'LE TOUR', 'AGUIL. DU MIDI',
       'AIGUILLES ROUGES-NIVOSE', 'LE GRAND-BORNAND', 'MEYTHET',
       'LE PLENAY', 'Vallot', 'Saint-Sorlin', 'Argentiere',
       'Dome Lac Blanc', 'Col du Lac Blanc', 'La Muzelle Lac Blanc',
       'Col de Porte', 'Col du Lautaret', 'TAE', 'AGAAR', 'COM', 'ABO',
       'AIG', 'TIAIR', 'TGALL', 'ALT', 'ARH', 'TGAMR', 'AND', 'ANT',
       'TGARE', 'ARO', 'RAG', 'BAS', 'LAT', 'BER', 'BEZ', 'BIA', 'BIN',
       'TIBIO', 'BIZ', 'BIV', 'BIE', 'BLA', 'BOL', 'MMBOY', 'BRZ', 'BUS',
       'BUF', 'FRE', 'TICAM', 'CEV', 'CHZ', 'MMCPY', 'CHA', 'CHM', 'CHU',
       'CHD', 'CIM', 'CDM', 'GSB', 'COY', 'CMA', 'CRM', 'DAV', 'DEM',
       'MMDAS', 'TGDIE', 'DIS', 'TGDUS', 'INNEBI', 'EBK', 'EGH', 'TGEGN',
       'EGO', 'EIN', 'ELM', 'MMERZ', 'INNESF', 'EVO', 'FAH', 'FLU',
       'MMFRS', 'TGFRA', 'GRA', 'FRU', 'MFOFKP', 'GVE', 'GES', 'GIH',
       'GLA', 'GOR', 'GOS', 'GOE', 'GRE', 'GRH', 'GRO', 'MMGTT', 'GUE',
       'GUT', 'HLL', 'MMHIR', 'MMHIW', 'HOE', 'TGILL', 'INT', 'MMIBG',
       'JUN', 'TGKAL', 'MMKSB', 'KOP', 'TGKRE', 'CDF', 'DOL', 'LAE',
       'LAG', 'TGLAN', 'MMLAF', 'MLS', 'MMNOI', 'LEI', 'MMLEN', 'CHB',
       'DIA', 'MAR', 'MMLIN', 'OTL', 'TGLOM', 'MMLOP', 'LUG', 'MMBIR',
       'LUZ', 'MAG', 'MAS', 'MMMAT', 'MAH', 'MTR', 'MER', 'MMMEL',
       'MMMES', 'TIMOL', 'MOE', 'MOB', 'MVE', 'GEN', 'MRP', 'MOA', 'MTE',
       'MMMOU', 'MUB', 'MMMUM', 'MMMUE', 'NAS', 'NAP', 'NEU', 'TGNOL',
       'TGNUS', 'CGI', 'OBR', 'AEG', 'MMOES', 'ORO', 'TGOTT', 'BEH',
       'PAY', 'PIL', 'PIO', 'COV', 'PMA', 'PLF', 'ROB', 'QUI', 'MMRAF',
       'INNRED', 'MMRIC', 'TGRIC', 'MMRIG', 'ROE', 'MMROM', 'RUE',
       'MMSAA', 'MMSAF', 'MMSAS', 'HAI', 'SAM', 'SAE', 'MMSRL', 'SBE',
       'SAG', 'MMSVG', 'MMSAX', 'SHA', 'SRS', 'SCM', 'AGSUA', 'SPF',
       'SIM', 'SIO', 'SBO', 'MMSSE', 'SCU', 'SIA', 'STG', 'SMM', 'MMSTA',
       'STK', 'PRE', 'THU', 'MMPRV', 'TIT', 'MMTIT', 'MMTRG', 'FLTRB',
       'ULR', 'MMUNS', 'TGUSH', 'VAD', 'VLS', 'VEV', 'VIO', 'VIT', 'VIS',
       'WAE', 'TGWEI', 'WFJ', 'KAWEG', 'WYN', 'ZER', 'MMZNZ', 'MMZOZ',
       'MMZWE', 'REH', 'SMA', 'KLO']
