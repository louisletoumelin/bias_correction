from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables,\
    adapt_distribution_strategy_to_available_devices,\
    init_learning_rate_adapted,\
    detect_variable,\
    get_idx_speed_and_dir_variables,\
    define_input_variables
from bias_correction.config._config import config

# Architecture
config["details"] = "d_speed_dir_v0"  # Str. Some details about the experiment
config["global_architecture"] = "double_ann"  # Str. Default="ann_v0", "dense_only", "dense_temperature", "devine_only", "double_ann"
config["restore_experience"] = "2022_11_29_labia_v5"

# ann_v0
config["disable_training_cnn"] = True  # Bool. Default=True
config["type_of_output"] = "output_speed"  # Str. "output_speed" or "output_components"
config["nb_units"] = [25, 10, 50]  # 25, 10
config["nb_units_speed"] = [25, 10, 50]  # 25, 10
config["nb_units_dir"] = [50, 10]  # 25, 10 or 1024, 256, 32
config["use_bias"] = True

# General
config["batch_normalization"] = False  # Bool. Apply batch_norm or not
config["activation_dense"] = "gelu"  # Bool. Activation in dense network, before selu
config["activation_dense_speed"] = "relu"  # Bool. Activation in dense network, before selu
config["activation_dense_dir"] = "gelu"  # Bool. Activation in dense network, before selu
config["dropout_rate"] = 0.35  # Int. or False. Dropout rate or no dropout
config["final_skip_connection"] = True  # Use skip connection with speed/direction
config["distribution_strategy"] = None  # "MirroredStrategy", "Horovod" or None
config["prefetch"] = "auto"  # Default="auto", else = Int

# Skip connections in dense network
config["dense_with_skip_connection"] = False

# Hyperparameters
config["batch_size"] = 128  # Int.
config["epochs"] = 5  # Int.
config["learning_rate"] = 0.001

# Optimizer
config["optimizer"] = "RMSprop"  # Str.
config["args_optimizer"] = [config["learning_rate"]]  # List.
config["kwargs_optimizer"] = {}  # Dict.

# Initializer
config["initializer"] = "GlorotUniform"  # Str. Default = "GlorotUniform"
config["args_initializer"] = []  # List.
config["kwargs_initializer"] = {"seed": 42}  # Dict.

# Input CNN
config["input_cnn"] = False
config["use_input_cnn_dir"] = False
config["use_batch_norm_cnn"] = False
config["activation_cnn"] = "gelu"
config["threshold_null_speed"] = 1
config["use_normalization_cnn_inputs"] = True

# Inputs pre-processing
config["standardize"] = True  # Bool. Apply standardization
config["shuffle"] = True  # Bool. Shuffle inputs

# Quick test
config["quick_test"] = False  # Bool. Quicktest case (fast training)
config["quick_test_stations"] = ["ALPE-D'HUEZ"]
# config["quick_test_stations"] = ["ALPE-D'HUEZ", 'Col du Lac Blanc', 'SOUM COUY-NIVOSE', 'SPONDE-NIVOSE']

# Input variables
#config["input_variables"] = ['alti', 'ZS', 'Wind', 'Wind_DIR', "Tair",
#                             "LWnet", "SWnet", 'CC_cumul', 'BLH',
#                             'Wind90', 'Wind87', 'Wind84', 'Wind75',
#                             "tpi_500", "curvature", "mu", "laplacian", 'aspect', 'tan(slope)']
config["input_speed"] = ["tpi_500", "curvature", "mu", "laplacian", 'alti', 'ZS', "Tair",
                         "LWnet", "SWnet", 'CC_cumul', 'BLH',  'Wind90', 'Wind87', 'Wind84', 'Wind75',
                         'Wind', 'Wind_DIR']
config["input_dir"] = ['aspect', 'tan(slope)', 'Wind', 'Wind_DIR']
# todo write a test that checks that topos, aspect and tan_slope are in the correct order
config["map_variables"] = ["topos", "aspect", "tan_slope", "tpi_300", "tpi_600"]
config["compute_product_with_wind_direction"] = True

# ['alti', 'ZS', 'Wind', 'Wind_DIR', "Tair",
#                              "LWnet", "SWnet", 'CC_cumul', 'BLH',
#                              'Wind90', 'Wind87', 'Wind84', 'Wind75',
#                              "tpi_500", "curvature", "mu", "laplacian",
#                              'dir_canyon_w0_1_w1_10',
#                              'dir_canyon_w0_5_w1_10',
#                              'dir_canyon_w0_1_w1_3_thresh5',
#                              'dir_canyon_w0_4_w1_20_thresh20',
#                              'diag_7', 'diag_13', 'diag_21', 'diag_31',
#                              'diag_7_r', 'diag_13_r', 'diag_21_r', 'diag_31_r',
#                              'side_7', 'side_13', 'side_21', 'side_31',
#                              'side_7_r', 'side_13_r', 'side_21_r', 'side_31_r',
#                              'aspect', 'tan(slope)']

#list_variables = ['name', 'date', 'lon', 'lat', 'alti', 'T2m(degC)', 'vw10m(m/s)',
#                  'winddir(deg)', 'HTN(cm)', 'Tair', 'T1', 'ts', 'Tmin', 'Tmax', 'Qair',
#                  'Q1', 'RH2m', 'Wind_Gust', 'PSurf', 'ZS', 'BLH', 'Rainf', 'Snowf',
#                  'LWdown', 'LWnet', 'DIR_SWdown', 'SCA_SWdown', 'SWnet', 'SWD', 'SWU',
#                  'LHF', 'SHF', 'CC_cumul', 'CC_cumul_low', 'CC_cumul_middle',
#                  'CC_cumul_high', 'Wind90', 'Wind87', 'Wind84', 'Wind75', 'TKE90',
#                  'TKE87', 'TKE84', 'TKE75', 'TT90', 'TT87', 'TT84', 'TT75', 'SWE',
#                  'snow_density', 'snow_albedo', 'vegetation_fraction', 'Wind',
#                  'Wind_DIR', 'U_obs', 'V_obs', 'U_AROME', 'V_AROME', "month", "hour"]

# Labels
config["labels"] = ['vw10m(m/s)']  # ["vw10m(m/s)"] or ["U_obs", "V_obs"] or ['T2m(degC)'] or ['winddir(deg)']
config["wind_nwp_variables"] = ["Wind", "Wind_DIR"]  # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]
config["wind_temp_variables"] = ['Tair']  # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]

# Dataset
config["unbalanced_dataset"] = False
config["unbalanced_threshold"] = 2

# Callbacks
config["callbacks"] = ["TensorBoard",
                       "ModelCheckpoint"]  # "FeatureImportanceCallback", "EarlyStopping",
config["args_callbacks"] = {"ReduceLROnPlateau": [],
                            "EarlyStopping": [],
                            "ModelCheckpoint": [],
                            "TensorBoard": [],
                            "CSVLogger": [],
                            "CSVLogger_dir": [],
                            "LearningRateWarmupCallback": [],
                            "FeatureImportanceCallback": [],
                            "BroadcastGlobalVariablesCallback": [],
                            "MetricAverageCallback": [],
                            "learning_rate_decay": [],
                            }

config["kwargs_callbacks"] = {"ReduceLROnPlateau": {"monitor": "val_loss",
                                                    "factor": 0.5,  # new_lr = lr * factor
                                                    "patience": 3,
                                                    "min_lr": 0.0001},

                              "EarlyStopping": {"monitor": "val_loss",
                                                "min_delta": 0.01,
                                                "patience": 5,
                                                "mode": "min",
                                                "restore_best_weights": False},

                              "ModelCheckpoint": {"min_delta": 0.01,
                                                  "monitor": "loss",
                                                  "save_best_only": True,
                                                  "save_weights_only": False,
                                                  "mode": "min",
                                                  "save_freq": "epoch"},

                              "TensorBoard": {"profile_batch": '20, 50',
                                              "histogram_freq": 1},

                              "LearningRateWarmupCallback": {"warmup_epochs": 5,
                                                             "verbose": 1},

                              "FeatureImportanceCallback": {},

                              "BroadcastGlobalVariablesCallback": {},

                              "MetricAverageCallback": {},

                              "CSVLogger": {},

                              "CSVLogger_dir": {},

                              "learning_rate_decay": {},
                              }

# Metrics
config["metrics"] = ["tf_mae", "tf_rmse", "tf_mbe"]

# Split
config["split_strategy_test"] = "time_and_space"  # "time", "space", "time_and_space", "random"
config["split_strategy_val"] = "time_and_space"

# Random split
config["random_split_state_test"] = 50  # Float. Default = 50
config["random_split_state_val"] = 55  # Float. Default = 55

# Time split
config["date_split_train_test"] = "2019-10-01"  # Str. Split train/test around this date
#                                                                         e.g. "2018-11-01"
config["date_split_train_val"] = "2019-10-01"

# Select test and val stations
config["parameters_split_test"] = ["alti", "tpi_500_NN_0", "mu_NN_0", "laplacian_NN_0", "Y", "X"]
config["parameters_split_val"] = ["alti", "tpi_500_NN_0"]
config["country_to_reject_during_training"] = ["pyr", "corse"]
config["metric_split"] = "rmse"

# Space split
# 2022_08_04 v4
# Warning: not the same as usual v2
config["stations_test"] = ['Col du Lac Blanc', 'GOE', 'WAE', 'TGKAL', 'LAG', 'AND', 'CHU', 'SMM', 'ULR', 'WFJ', 'TICAM',
                           'SCM', 'MMMEL', 'INNRED', 'MMBIR', 'MMHIW', 'MMLOP', 'TGALL', 'GAP', 'BAS', 'STK', 'PLF',
                           'MVE', 'SAG', 'MLS', 'MAR', 'MTE', 'MTR', 'CHZ', 'SIA', 'COV', 'MMSTA', 'BIV', 'ANT',
                           'TGDIE', 'CHM', 'TGARE', 'TALLARD', 'LE CHEVRIL-NIVOSE', 'GOR', 'MMMUE', 'INT', 'BIE', 'EIN',
                           'RUE', 'QUI', 'NEU', 'MMNOI', 'LE GUA-NIVOSE', 'GIH', 'AEG', 'MOE', 'LUG', 'TGNUS', 'BEH']
# usual
"""
['Col du Lac Blanc', 'GOE', 'WAE', 'TGKAL', 'LAG', 'AND', 'CHU', 'SMM', 'ULR', 'WFJ', 'TICAM',
                           'SCM', 'MMMEL', 'INNRED', 'MMBIR', 'MMHIW', 'MMLOP', 'TGALL', 'GAP', 'BAS', 'STK', 'PLF',
                           'MVE', 'SAG', 'MLS', 'MAR', 'MTE', 'MTR', 'CHZ', 'SIA', 'COV', 'MMSTA', 'BIV', 'ANT',
                           'TGDIE', 'CHM', 'TGARE', 'TALLARD', 'LE CHEVRIL-NIVOSE', 'GOR', 'MMMUE', 'INT', 'BIE', 'EIN',
                           'RUE', 'QUI', 'NEU', 'MMNOI', 'LE GUA-NIVOSE', 'GIH', 'AEG', 'MOE', 'LUG', 'TGNUS', 'BEH']
"""
config["stations_val"] = []

# Before merging
# ['Col du Lac Blanc', 'GOE', 'WAE', 'TGKAL', 'LAG', 'AND', 'CHU', 'SMM', 'ULR', 'WFJ', 'TICAM',
#                           'SCM', 'MMMEL', 'INNRED', 'MMBIR', 'MMHIW', 'MMLOP', 'TGALL', 'GAP', 'BAS', 'STK', 'PLF',
#                           'MVE', 'SAG', 'MLS', 'MAR', 'MTE', 'MTR', 'CHZ', 'SIA', 'COV', 'MMSTA', 'BIV', 'ANT',
#                           'TGDIE', 'CHM', 'TGARE', 'TALLARD', 'LE CHEVRIL-NIVOSE', 'GOR', 'MMMUE', 'INT', 'BIE', 'EIN',
#                           'RUE', 'QUI', 'NEU', 'MMNOI', 'LE GUA-NIVOSE', 'GIH', 'AEG', 'MOE', 'LUG', 'TGNUS', 'BEH']
# ['WYN', 'BER', 'ARH', 'ELM', 'MMMES', 'ASCROS', 'GRANDE PAREI NIVOSE', 'SAM', 'JUN', 'SCU',
#                          'MMSVG', 'GALIBIER-NIVOSE', 'MEYTHET', 'BRZ', 'OBR', 'FAH', 'MMRIG', 'PMA']

config["stations_to_reject"] = ["Vallot", "Dome Lac Blanc", "MFOKFP"]

# Intermediate output
config["get_intermediate_output"] = True

# Custom loss
config["loss"] = "pinball_proportional"  # Str. Default=mse. Used for gradient descent
config["args_loss"] = {"mse": [],
                       "penalized_mse": [],
                       "mse_proportional": [],
                       "mse_power": [],
                       "pinball": [],
                       "pinball_proportional": [],
                       "pinball_weight": [],
                       "cosine_distance": []}
config["kwargs_loss"] = {"mse": {},
                         "penalized_mse": {"penalty": 10,
                                           "speed_threshold": 5},
                         "mse_proportional": {"penalty": 1},
                         "mse_power": {"penalty": 1,
                                       "power": 2},
                         "pinball": {"tho": 0.85},
                         "pinball_proportional": {"tho": 0.6},
                         "pinball_weight": {"tho": 0.95},
                         "cosine_distance": {"power": 1}}

# Do not modify: assert inputs are correct
config = define_input_variables(config)
config = assert_input_for_skip_connection(config)
config = sort_input_variables(config)
config = adapt_distribution_strategy_to_available_devices(config)
config = init_learning_rate_adapted(config)
config["nb_input_variables"] = len(config["input_variables"])
config = detect_variable(config)
config = get_idx_speed_and_dir_variables(config)

list_variables = ['name', 'date', 'lon', 'lat', 'alti', 'T2m(degC)', 'vw10m(m/s)',
                  'winddir(deg)', 'HTN(cm)', 'Tair', 'T1', 'ts', 'Tmin', 'Tmax', 'Qair',
                  'Q1', 'RH2m', 'Wind_Gust', 'PSurf', 'ZS', 'BLH', 'Rainf', 'Snowf',
                  'LWdown', 'LWnet', 'DIR_SWdown', 'SCA_SWdown', 'SWnet', 'SWD', 'SWU',
                  'LHF', 'SHF', 'CC_cumul', 'CC_cumul_low', 'CC_cumul_middle',
                  'CC_cumul_high', 'Wind90', 'Wind87', 'Wind84', 'Wind75', 'TKE90',
                  'TKE87', 'TKE84', 'TKE75', 'TT90', 'TT87', 'TT84', 'TT75', 'SWE',
                  'snow_density', 'snow_albedo', 'vegetation_fraction', 'Wind',
                  'Wind_DIR', 'U_obs', 'V_obs', 'U_AROME', 'V_AROME', "month", "hour"]

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
# De même, marche bien mais DEVINE a de mauvaises statistiques
# 2022_08_04 v5
config["stations_test"] = ['Col du Lac Blanc', 'CREYS-MALVILLE', 'CEV', 'TGKAL', 'EIN', 'AEG', 'CHB', 'ZER', 'ULR', 'CMA', 'TICAM', 'AND', 'LES ROCHILLES-NIVOSE', 'HAI', 'MAH', 'BEH', 'LUS L CROIX HTE', 'BONNEVAL-NIVOSE', 'PIL', 'MMZNZ', 'KOP', 'ARH', 'INNEBI', 'GRA', 'ASCROS', 'ST JEAN-ST-NICOLAS', 'BER', 'AIGUILLES ROUGES-NIVOSE', 'TGNOL', 'VILLAR ST PANCRACE', 'SAE', 'PAY', 'MRP', 'BIZ', 'ARVIEUX', 'AGSUA', 'MTR', 'ST-PIERRE-LES EGAUX', 'GRANDE PAREI NIVOSE', 'Argentiere', 'COM', 'MMPRV', 'TIT', 'WYN', 'WAE', 'TGALL', 'DIGNE LES BAINS', 'RISTOLAS', 'DIA', 'BUS', 'BLA', 'MMRAF', 'SMM', 'MMSVG', 'MMSAX']
config["stations_val"] = ['LUG', 'TGUSH', 'RAG', 'SRS', 'SAG', 'MER', 'BIV', 'PARPAILLON-NIVOSE', 'EGH', 'MOB', 'MMDAS', 'INNESF', 'DEM', 'GRE', 'MMSSE', 'TGDIE', 'DAV', 'LA MASSE']

# Marche bien mais DEVINE trop biaisé
# 2022_08_04 v4
config["stations_test"] = ['Col du Lac Blanc', 'GOE', 'WAE', 'TGKAL', 'LAG', 'AND', 'CHU', 'SMM', 'ULR', 'WFJ', 'TICAM', 'SCM', 'MMMEL', 'INNRED', 'MMBIR', 'MMHIW', 'MMLOP', 'TGALL', 'GAP', 'BAS', 'STK', 'PLF', 'MVE', 'SAG', 'MLS', 'MAR', 'MTE', 'MTR', 'CHZ', 'SIA', 'COV', 'MMSTA', 'BIV', 'ANT', 'TGDIE', 'CHM', 'TGARE', 'TALLARD', 'LE CHEVRIL-NIVOSE', 'GOR', 'MMMUE', 'INT', 'BIE', 'EIN', 'RUE', 'QUI', 'NEU', 'MMNOI', 'LE GUA-NIVOSE', 'GIH', 'AEG', 'MOE', 'LUG', 'TGNUS', 'BEH']
config["stations_val"] = ['WYN', 'BER', 'ARH', 'ELM', 'MMMES', 'ASCROS', 'GRANDE PAREI NIVOSE', 'SAM', 'JUN', 'SCU', 'MMSVG', 'GALIBIER-NIVOSE', 'MEYTHET', 'BRZ', 'OBR', 'FAH', 'MMRIG', 'PMA']

# 2022_08_04 v3
['Col du Lac Blanc', 'LUZ', 'REH', 'MMSAX', 'TIAIR', 'MMDAS', 'MER', 'LAT', 'PEONE', 'NAS', 'ARVIEUX', 'SCM', 'LES ECRINS-NIVOSE', 'MMERZ', 'STG', 'Col du Lautaret', 'WAE', "ALPE-D'HUEZ", 'LAE', 'GIH', "VILLAR D'ARENE", 'TAE', 'MMMUM', 'RUE', 'MMBOY', 'SCU', 'GOS', 'TGDUS', 'COL-DES-SAISIES', 'TGAMR', 'COV', 'HAI', 'VAD', 'ULR', 'ST-PIERRE-LES EGAUX', 'LE PLENAY', 'CHU', 'LUG', 'LES ROCHILLES-NIVOSE', 'AIG', 'MMLEN', 'MMPRV', 'DAV', 'KLO', 'GRE', 'OBR', 'TALLARD', 'BIE', 'AIGLETON-NIVOSE', 'MVE', 'ROE', 'ANT', 'EBK', 'CEV', 'TGARE']
['GRENOBLE - LVD', 'MMBIR', 'MMTRG', 'DIS', 'MMRIC', 'TGOTT', 'MMSAF', 'PEIRA CAVA', 'GUE', 'LAG', 'MMCPY', 'CDF', 'MMMAT', 'TGWEI', 'MMHIW', 'PAY', 'BIZ', 'MONT DU CHAT']


# 2022_08_04 v2
['Col du Lac Blanc', 'GIH', 'LEI', 'QUI', 'VIT', 'MMCPY', 'CHM', 'SMM', 'RISTOLAS', 'JUN', 'MMSAS', 'MMSRL', 'La Muzelle Lac Blanc', 'TGLAN', 'MMPRV', 'SIA', 'FRE', 'ALLANT-NIVOSE', 'LA MASSE', 'HAI', "VILLAR D'ARENE", 'MAG', 'FAH', 'TGNUS', 'GUE', 'BUF', 'LE CHEVRIL-NIVOSE', 'HOE', 'MMLOP', 'MMTIT', 'AIGUILLES ROUGES-NIVOSE', 'SRS', 'MMDAS', 'LA MURE- RADOME', 'MMLIN', 'SPF', 'GAP', 'CREYS-MALVILLE', 'PEIRA CAVA', 'GSB', 'ARO', 'BOL', 'BIE', 'KLO', 'EGO', 'AGAAR', 'PAY', 'GRANDE PAREI NIVOSE', 'AIGLETON-NIVOSE', 'INNRED', 'GES', 'MOE', 'MMFRS', 'TGILL', 'DAV']
['GRO', 'TGRIC', 'TGEGN', 'FLU', 'MMUNS', 'CHU', 'BIN', 'PEONE', 'TIT', 'MOB', 'AND', 'INNESF', 'MMERZ', 'MMRAF', 'MMSSE', 'CGI', 'TGAMR', 'PIL']

# 2022_08_04
config["stations_test"] = ['Col du Lac Blanc', 'MMLAF', 'WAE', 'AGAAR', 'DIS', 'MMDAS', 'EMBRUN', 'ZER', 'LE TOUR',
                           'GRH', 'SRS', 'MMSRL', 'BER', 'GIH', 'VAD', 'TAE', 'EGO', 'TGAMR', 'JUN', 'DEM', 'BRZ',
                           'MMHIW', 'COL-DES-SAISIES', 'LE PLENAY', 'BONNEVAL-NIVOSE', 'MMCPY', 'Col de Porte', 'DIA',
                           'TGNOL', 'INNESF', 'LA MEIJE-NIVOSE', 'LEI', 'STK', 'SIO', 'ST-PIERRE-LES EGAUX', 'MAS',
                           'Argentiere', 'MEYTHET', 'BELLECOTE-NIVOSE', 'ASCROS', 'MMZWE', 'ROB', 'MFOFKP', 'WYN',
                           'MMBIR', 'TGEGN', 'CDM', "VILLAR D'ARENE", 'PARPAILLON-NIVOSE', 'INNRED', 'HLL', 'ULR',
                           'TGUSH', 'TGNUS', 'SIA']

config["stations_val"] = ['SBO', 'CEV', 'MMBOY', 'THU', 'AND', 'MMTIT', 'ROE', 'RESTEFOND-NIVOSE',
                          'La Muzelle Lac Blanc', 'BARCELONNETTE', 'GRANDE PAREI NIVOSE', 'GALIBIER-NIVOSE',
                          'ALBERTVILLE JO', 'MER', 'ARH', 'SPF', 'PLF', 'AIGLETON-NIVOSE']


# 2022_08_03
['MMMES', 'EGH', 'LA MURE-ARGENS', 'BLA', 'EVO', 'TGAMR', 'TIT',
                           'MILLEFONTS-NIVOSE', 'COY', 'ULR', 'AGUIL. DU MIDI', 'SIA', 'PLF',
                           'BIN', 'ZER', 'MMMUE', 'EGO', 'LUZ', 'DIGNE LES BAINS', 'TGILL',
                           'CIM', 'Col du Lac Blanc', 'TICAM', 'CHB', 'LE PLENAY',
                           'MONT DU CHAT', 'VAD', 'ARH', 'MTR', 'EMBRUN', 'MMKSB', 'MOB',
                           'MMIBG', 'MMSAF', 'MMSAX', 'BIZ', 'BER', 'TALLARD', 'GRO',
                           'INNEBI', 'MLS', 'PEONE', 'SRS', 'FLU', 'GVE', 'BIA', 'MMRIC',
                           'PIL', 'MMSAS', 'La Muzelle Lac Blanc', 'TGDIE', 'TIAIR', 'MMNOI',
                           'ALBERTVILLE JO']

['LA MASSE', 'LE GRAND-BORNAND', 'MER', 'TGRIC', 'ARVIEUX',
  'BONNEVAL-NIVOSE', 'MMERZ', 'INNESF', 'TGLAN', 'MMLAF', 'SHA',
  'AND', 'Argentiere', 'BEH', 'TGNOL', 'GOS', 'PARPAILLON-NIVOSE',
  'MOA']    

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










General results
mbe: -0.09720373898744583
rmse: 1.9583745002746582
corr: 0.5967392491028242
mae: 1.3460997343063354
 Alti category = (0, 1000)m
Mode: Validation,  Nb stations: 10,  bias:  0.21,  rmse:  1.57,  corr:  0.59,  mae:  1.11
Mode: Training,  Nb stations: 114,  bias:  0.26,  rmse:  1.58,  corr:  0.64,  mae:  1.15
Mode: Test,  Nb stations: 31,  bias:  0.32,  rmse:  1.52,  corr:  0.65,  mae:  1.11
 Alti category = (1000, 2000)m
Mode: Validation,  Nb stations: 3,  bias:  0.05,  rmse:  1.53,  corr:  0.72,  mae:  1.10
Mode: Training,  Nb stations: 59,  bias: -0.35,  rmse:  2.02,  corr:  0.57,  mae:  1.37
Mode: Test,  Nb stations: 15,  bias: -0.10,  rmse:  1.76,  corr:  0.57,  mae:  1.29
 Alti category = (2000, 3000)m
Mode: Validation,  Nb stations: 5,  bias:  0.34,  rmse:  2.53,  corr:  0.51,  mae:  1.85
Mode: Training,  Nb stations: 22,  bias: -1.40,  rmse:  3.07,  corr:  0.60,  mae:  2.20
Mode: Test,  Nb stations: 6,  bias: -1.10,  rmse:  2.76,  corr:  0.61,  mae:  1.93
 Alti category = (3000, 5000)m
Mode: Validation,  Nb stations: 0,  bias:  nan,  rmse:  nan,  corr:  nan,  mae:  nan
Mode: Training,  Nb stations: 4,  bias: -2.28,  rmse:  3.84,  corr:  0.55,  mae:  2.83
Mode: Test,  Nb stations: 3,  bias: -0.16,  rmse:  2.64,  corr:  0.70,  mae:  1.96









General results
Training_mbe: -0.026106510311365128
Test_mbe: -0.31169164180755615
Validation_mbe: -0.26853638887405396
Training_rmse: 1.94012451171875
Test_rmse: 2.045722246170044
Validation_rmse: 1.8953019380569458
Training_corr: 0.5990634506707512
Test_corr: 0.5785604500111888
Validation_corr: 0.6536016792513266
Training_mae: 1.3388566970825195
Test_mae: 1.3758349418640137
Validation_mae: 1.3379265069961548
 Alti category = (0, 1000)m
Mode: Training,  Nb stations: 117,  bias:  0.31,  rmse:  1.58,  corr:  0.65,  mae:  1.14
Mode: Test,  Nb stations: 29,  bias:  0.19,  rmse:  1.54,  corr:  0.59,  mae:  1.11
Mode: Validation,  Nb stations: 10,  bias: -0.03,  rmse:  1.54,  corr:  0.65,  mae:  1.13
 Alti category = (1000, 2000)m
Mode: Training,  Nb stations: 57,  bias: -0.30,  rmse:  2.04,  corr:  0.57,  mae:  1.39
Mode: Test,  Nb stations: 17,  bias: -0.39,  rmse:  1.78,  corr:  0.61,  mae:  1.24
Mode: Validation,  Nb stations: 3,  bias:  0.62,  rmse:  1.34,  corr:  0.49,  mae:  1.02
 Alti category = (2000, 3000)m
Mode: Training,  Nb stations: 22,  bias: -0.91,  rmse:  2.93,  corr:  0.54,  mae:  2.08
Mode: Test,  Nb stations: 6,  bias: -1.56,  rmse:  3.16,  corr:  0.61,  mae:  2.28
Mode: Validation,  Nb stations: 5,  bias: -1.36,  rmse:  2.71,  corr:  0.68,  mae:  1.98
 Alti category = (3000, 5000)m
Mode: Training,  Nb stations: 5,  bias: -0.90,  rmse:  2.87,  corr:  0.63,  mae:  2.10
Mode: Test,  Nb stations: 2,  bias: -2.65,  rmse:  4.50,  corr:  0.48,  mae:  3.44
Mode: Validation,  Nb stations: 0,  bias:  nan,  rmse:  nan,  corr:  nan,  mae:  nan







"""
