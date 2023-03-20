from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables, adapt_distribution_strategy_to_available_devices, init_learning_rate_adapted, detect_variable

config = {"network": detect_network()}

"""
├── ./Data
│   │
│   ├── ./Data/1_Raw
│   │   │
│   │   ├── ./Data/1_Raw/AROME
│   │   │   └── ...
│   │   │
│   │   ├── ./Data/1_Raw/CNN
│   │   │   └── ./Data/1_Raw/CNN/date_21_12_2021_name_simu_classic_all_low_0_model_UNet
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

# Path downscale
if config["network"] == "local":
    config["path_module_downscale"] = "/home/letoumelinl/wind_downscaling_cnn/src/downscale_/"
else:
    config["path_module_downscale"] = "//home/mrmn/letoumelinl/downscale_/"


# Path to CNN
config["cnn_name"] = "date_21_12_2021_name_simu_classic_all_low_epochs_0_model_UNet/"
config["path_experience"] = config["path_root"] + "Data/1_Raw/CNN/"
config["model_path"] = config["path_experience"] + config["cnn_name"]

# Path to inputs and labels data
config["folder_obs"] = "2022_06_10"
config["path_stations_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/stations/{config['folder_obs']}/"
config["path_time_series_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/time_series/{config['folder_obs']}/"
config["path_topos_pre_processed"] = config["path_root"] + f"Data/2_Pre_processed/topos/{config['folder_obs']}/"
config["path_experiences"] = config["path_root"] + "Data/3_Predictions/Experiences/"
config["path_to_devine"] = config["path_root"] + "Data/3_Predictions/DEVINE/"
config["path_to_analysis"] = config["path_root"] + "Data/2_Pre_processed/AROME_analysis/"

# Filename inputs and labels data
config["time_series"] = config["path_time_series_pre_processed"] + "time_series_bc.pkl"
config["time_series_int"] = config["path_time_series_pre_processed"] + "time_series_bc_interpolated.pkl"
config["stations"] = config["path_stations_pre_processed"] + "stations_bc.pkl"
config["topos_near_station"] = config["path_topos_pre_processed"] + "dict_topo_near_station.pickle"
config["topos_near_nwp"] = config["path_topos_pre_processed"] + "dict_topo_near_nwp.pickle"
config["topos_near_nwp_int"] = config["path_topos_pre_processed"] + "dict_topo_near_nwp_inter.pickle"

# Architecture
config["details"] = "pp_50_20"                                            # Str. Some details about the experiment
config["global_architecture"] = "ann_v0"                                # Str. Default="ann_v0", "dense_only", "dense_temperature", "devine_only"
config["restore_experience"] = False #"2022_7_27_labia_v4"

# ann_v0
config["disable_training_cnn"] = True                                   # Bool. Default=True
config["type_of_output"] = "output_speed"                               # Str. "output_speed" or "output_components"
config["nb_units"] = [25, 10]       #25, 10                                    # List. Each member is a unit [40, 20, 10, 5]
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
config["nb_layers_skip_connection"] = 3

# Hyperparameters
config["batch_size"] = 128                                                # Int.
config["epochs"] = 20                                                      # Int.
config["learning_rate"] = 0.001

# Optimizer
config["optimizer"] = "Adam"                                         # Str.
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
config["quick_test_stations"] = ["ALPE-D'HUEZ", 'Col du Lac Blanc', 'SOUM COUY-NIVOSE', 'SPONDE-NIVOSE']

# Input variables
config["input_variables"] = ['alti', 'ZS', 'Wind', 'Wind_DIR', "Tair",
                              "LWnet", "SWnet", 'CC_cumul', 'BLH']
# ["tpi_500", "curvature", "mu", "laplacian", 'alti', 'ZS', 'Wind', 'Wind_DIR', "Tair",
#                              "LWnet", "SWnet", 'CC_cumul', 'BLH']

# Labels
config["labels"] = ['vw10m(m/s)']                                       # ["vw10m(m/s)"] or ["U_obs", "V_obs"] or ['T2m(degC)']
config["wind_nwp_variables"] = ["Wind", "Wind_DIR"]                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]
config["wind_temp_variables"] = ['Tair']                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]

# Dataset
config["unbalanced_dataset"] = False
config["unbalanced_threshold"] = 2

# Callbacks
config["callbacks"] = ["TensorBoard",
                       "ReduceLROnPlateau",
                       "EarlyStopping",
                       "CSVLogger",
                       "ModelCheckpoint"]  # "FeatureImportanceCallback"

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

                              "HVDLearningRateWarmupCallback": {"warmup_epochs": 5,
                                                                "verbose": 1},

                              "FeatureImportanceCallback": {}
                              }

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

# Select test and val stations
config["parameters_split_test"] = ["alti", "tpi_500_NN_0", "mu_NN_0", "laplacian_NN_0", "Y", "X"]
config["parameters_split_val"] = ["alti", "tpi_500_NN_0"]
config["country_to_reject_during_training"] = ["pyr", "corse"]
config["metric_split"] = "rmse"

# Space split
config["stations_test"] = ['MMMES', 'EGH', 'LA MURE-ARGENS', 'BLA', 'EVO', 'TGAMR', 'TIT',
                           'MILLEFONTS-NIVOSE', 'COY', 'ULR', 'AGUIL. DU MIDI', 'SIA', 'PLF',
                           'BIN', 'ZER', 'MMMUE', 'EGO', 'LUZ', 'DIGNE LES BAINS', 'TGILL',
                           'CIM', 'Col du Lac Blanc', 'TICAM', 'CHB', 'LE PLENAY',
                           'MONT DU CHAT', 'VAD', 'ARH', 'MTR', 'EMBRUN', 'MMKSB', 'MOB',
                           'MMIBG', 'MMSAF', 'MMSAX', 'BIZ', 'BER', 'TALLARD', 'GRO',
                           'INNEBI', 'MLS', 'PEONE', 'SRS', 'FLU', 'GVE', 'BIA', 'MMRIC',
                           'PIL', 'MMSAS', 'La Muzelle Lac Blanc', 'TGDIE', 'TIAIR', 'MMNOI',
                           'ALBERTVILLE JO']


config["stations_val"] = ['LA MASSE', 'LE GRAND-BORNAND', 'MER', 'TGRIC', 'ARVIEUX',
                          'BONNEVAL-NIVOSE', 'MMERZ', 'INNESF', 'TGLAN', 'MMLAF', 'SHA',
                          'AND', 'Argentiere', 'BEH', 'TGNOL', 'GOS', 'PARPAILLON-NIVOSE',
                          'MOA']

config["stations_to_reject"] = ["Vallot", "Dome Lac Blanc", "MFOKFP"]

# Intermediate output
config["get_intermediate_output"] = False

# Custom loss
config["loss"] = "pinball_proportional"                                                  # Str. Default=mse. Used for gradient descent
config["args_loss"] = []
config["kwargs_loss"] = {"penalized_mse": {"penalty": 10,
                                           "speed_threshold": 5},
                         "mse_proportional": {"penalty": 1},
                         "mse_power": {"penalty": 1,
                                       "power": 2},
                         "pinball": {"tho": 0.85},
                         "pinball_proportional": {"tho": 0.6},
                         "pinball_weight": {"tho": 0.95}}

# todo assert station validation not in station test
# todo assert kwargs split strategy are defined
# todo assert a seed is given for any random input


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

all_stations = ['ABO', 'AEG', 'AGAAR', 'AGSUA', 'AGUIL. DU MIDI', 'AIG',
                'AIGLETON-NIVOSE', 'AIGUILLES ROUGES-NIVOSE',
                'AIGUILLETTES_NIVOSE', 'ALBERTVILLE JO', 'ALISTRO',
                'ALLANT-NIVOSE', "ALPE-D'HUEZ", 'ALT', 'AND', 'ANDORRE-LA-VIEILLE',
                'ANT', 'ARH', 'ARO', 'ARVIEUX', 'ASCROS', 'ASTON', 'Argentiere',
                'BARCELONNETTE', 'BAS', 'BAZUS-AURE', 'BEH', 'BELLECOTE-NIVOSE',
                'BER', 'BEZ', 'BIA', 'BIARRITZ-PAYS-BASQUE', 'BIE', 'BIN', 'BIV',
                'BIZ', 'BLA', 'BOL', 'BONNEVAL-NIVOSE', 'BRZ', 'BUF', 'BUS',
                'CALVI', 'CANIGOU-NIVOSE', 'CAP CORSE', 'CAP PERTUSATO',
                'CAP SAGRO', 'CDF', 'CDM', 'CEV', 'CGI', 'CHA', 'CHAMROUSSE',
                'CHAPELLE-EN-VER', 'CHB', 'CHD', 'CHM', 'CHU', 'CHZ', 'CIM',
                'CLARAC', 'CMA', 'COL AGNEL-NIVOSE', 'COL-DES-SAISIES', 'COM',
                'CONCA', 'COS', 'COV', 'COY', 'CREYS-MALVILLE', 'CRM',
                'Col de Porte', 'Col du Lac Blanc', 'Col du Lautaret', 'DAV',
                'DEM', 'DIA', 'DIGNE LES BAINS', 'DIS', 'DOL', 'Dome Lac Blanc',
                'EBK', 'EGH', 'EGO', 'EIN', 'ELM', 'EMBRUN', 'EVO', 'FAH',
                'FIGARI', 'FLU', 'FORMIGUERES', 'FRE', 'FRU', 'GALIBIER-NIVOSE',
                'GAP', 'GEN', 'GES', 'GIH', 'GLA', 'GOE', 'GOR', 'GOS', 'GRA',
                'GRANDE PAREI NIVOSE', 'GRE', 'GRENOBLE - LVD',
                'GRENOBLE-ST GEOIRS', 'GRH', 'GRO', 'GSB', 'GUE', 'GUT', 'GVE',
                'HAI', 'HLL', 'HOE', 'HOSPITALET-NIVOSE', 'ILE ROUSSE', 'INNEBI',
                'INNESF', 'INNRED', 'INT', 'IRATY ORGAMBIDE', 'JUN', 'KAWEG',
                'KLO', 'KOP', 'LA CHIAPPA', 'LA FAURIE', 'LA MASSE',
                'LA MEIJE-NIVOSE', 'LA MURE- RADOME', 'LA MURE-ARGENS',
                "LAC D'ARDIDEN-NIVOSE", 'LAE', 'LAG', 'LAT', 'LE CHEVRIL-NIVOSE',
                'LE GRAND-BORNAND', 'LE GUA-NIVOSE', 'LE PLENAY', 'LE TOUR', 'LEI',
                'LES ECRINS-NIVOSE', 'LES ROCHILLES-NIVOSE', 'LOUDERVIELLE',
                'LUCHON', 'LUG', 'LUS L CROIX HTE', 'LUZ', 'La Muzelle Lac Blanc',
                'MAG', 'MAH', 'MANICCIA-NIVOSE', 'MAR', 'MAS', 'MAUPAS-NIVOSE',
                'MER', 'MEYTHET', 'MFOFKP', 'MILLEFONTS-NIVOSE', 'MLS', 'MMBIR',
                'MMBOY', 'MMCPY', 'MMDAS', 'MMERZ', 'MMFRS', 'MMGTT', 'MMHIR',
                'MMHIW', 'MMIBG', 'MMKSB', 'MMLAF', 'MMLEN', 'MMLIN', 'MMLOP',
                'MMMAT', 'MMMEL', 'MMMES', 'MMMOU', 'MMMUE', 'MMMUM', 'MMNOI',
                'MMOES', 'MMPRV', 'MMRAF', 'MMRIC', 'MMRIG', 'MMROM', 'MMSAF',
                'MMSAS', 'MMSAX', 'MMSRL', 'MMSSE', 'MMSTA', 'MMSVG', 'MMTIT',
                'MMTRG', 'MMUNS', 'MMZNZ', 'MMZOZ', 'MMZWE', 'MOA', 'MOB',
                'MOCA-CROCE', 'MOE', 'MONT DU CHAT', 'MRP', 'MTE', 'MTR', 'MUB',
                'MVE', 'NAP', 'NAS', 'NEU', 'OBR', 'OLETTA', 'ORCIERES-NIVOSE',
                'ORO', 'OTL', 'PARPAILLON-NIVOSE', 'PAY', 'PEIRA CAVA', 'PEONE',
                'PIETRALBA', 'PIL', 'PILA-CANALE', 'PIO', 'PLF', 'PMA',
                "PORT D'AULA-NIVOSE", 'PRE', 'PUIGMAL-NIVOSE', 'QUI', 'RAG', 'REH',
                'RENNO', 'RESTEFOND-NIVOSE', 'RISTOLAS', 'ROB', 'ROE', 'RUE',
                'SAE', 'SAG', 'SALINES', 'SAM', 'SARTENE', 'SBE', 'SBO', 'SCM',
                'SCU', 'SERRALONGUE', 'SHA', 'SIA', 'SIM', 'SIO', 'SMA', 'SMM',
                'SOLENZARA', 'SOUM COUY-NIVOSE', 'SPF', 'SPONDE-NIVOSE', 'SRS',
                'ST HILAIRE-NIVOSE', 'ST JEAN-ST-NICOLAS', 'ST ROMAN-DIOIS',
                'ST-PIERRE-LES EGAUX', 'STG', 'STK', 'Saint-Sorlin', 'TAE',
                'TALLARD', 'TGALL', 'TGAMR', 'TGARE', 'TGDIE', 'TGDUS', 'TGEGN',
                'TGFRA', 'TGILL', 'TGKAL', 'TGKRE', 'TGLAN', 'TGLOM', 'TGNOL',
                'TGNUS', 'TGOTT', 'TGRIC', 'TGUSH', 'TGWEI', 'THU', 'TIAIR',
                'TIBIO', 'TICAM', 'TIMOL', 'TIT', 'TOULOUSE-BLAGNAC', 'ULR', 'VAD',
                'VEV', "VILLAR D'ARENE", 'VILLAR ST PANCRACE', 'VILLARD-DE-LANS',
                'VIO', 'VIS', 'VIT', 'VLS', 'Vallot', 'WAE', 'WFJ', 'WYN', 'ZER']


"""
config["stations_test"] = ['AIGUILLES ROUGES-NIVOSE', 'LE GRAND-BORNAND', 'MEYTHET', 'LE PLENAY', 'Saint-Sorlin',
                           'Argentiere', 'Col du Lac Blanc', 'CHA', 'CMA', 'DOL', "GALIBIER-NIVOSE", 'LA MURE-ARGENS',
                           'ARVIEUX', 'PARPAILLON-NIVOSE', 'EMBRUN', 'LA FAURIE', 'GAP', 'LA MEIJE-NIVOSE',
                           'COL AGNEL-NIVOSE', 'HOE', 'TGILL', 'INT', 'EGO', 'EIN', 'ELM', 'MMERZ', 'INNESF',
                           'EVO', 'FAH', 'FLU', 'MMFRS', 'TGFRA', 'GRA', 'FRU', 'MFOFKP', 'GVE', 'GES', 'GIH']

config["stations_val"] = ['MMKSB', 'KOP', 'TGKRE', 'ALBERTVILLE JO', 'BONNEVAL-NIVOSE', 'SHA',
                          'SRS', 'SCM', 'WAE', 'TGWEI', 'WFJ', 'KAWEG', 'WYN', 'ZER', 'MMZNZ', 'MMZOZ',
                          'MMZWE', 'REH', 'SMA', 'KLO']

config["stations_test"] = ['INNRED', 'TICAM', 'STK', 'OBR', 'EIN', 'AND', 'TGOTT', 'PIO', 'LES ROCHILLES-NIVOSE',
                           'ULR', 'SBE', 'BEH', 'ABO', 'MMCPY', 'MMZOZ', 'INNESF', 'CREYS-MALVILLE', 'SBO', 'TGKRE',
                           'AGAAR', 'MMLAF', 'PLF', 'Col de Porte', 'DIA', 'DEM', 'VAD', 'TAE', 'TGEGN', 'MVE',
                           'MMRIC', 'ROE', 'EMBRUN', 'MMSRL', 'BER', 'Col du Lac Blanc', 'GSB', 'MOB',
                           'ORCIERES-NIVOSE', 'PARPAILLON-NIVOSE', 'GALIBIER-NIVOSE', 'LAT', 'FRE', 'CHB', 'GUE',
                           'FAH', 'TGWEI', 'COY', 'CHA', 'LA MURE-ARGENS', 'MMROM', 'BIE', 'AGUIL. DU MIDI', 'WYN',
                           'HLL', 'EGO', 'ANT', 'TGRIC', 'SCM', 'MMHIW', 'CMA']


config["stations_val"] = ['BUS', 'MMBIR', 'AGSUA', 'GRENOBLE-ST GEOIRS', 'DIS', 'RUE', 'GOS', 'CDF',
                          'ARVIEUX', 'COL AGNEL-NIVOSE', 'PEIRA CAVA', 'PMA']
"""
