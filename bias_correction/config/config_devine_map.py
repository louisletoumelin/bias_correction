from bias_correction.utils_bc.network import detect_network
from bias_correction.utils_bc.utils_config import assert_input_for_skip_connection, \
    sort_input_variables, adapt_distribution_strategy_to_available_devices, init_learning_rate_adapted, detect_variable
from bias_correction.config._config import config

# Architecture
config["details"] = "new_test"                                          # Str. Some details about the experiment
config["global_architecture"] = "devine_only"                                # Str. Default="ann_v0", "dense_only", "dense_temperature", "devine_only"
config["restore_experience"] = False

# ann_v0
config["type_of_output"] = "map"                               # Str. "output_speed" or "output_components", "map", "map_components"

# General
config["distribution_strategy"] = None                                  # "MirroredStrategy", "Horovod" or None
config["prefetch"] = "auto"                                                # Default="auto", else = Int

# Inputs pre-processing
config["standardize"] = False                                            # Bool. Apply standardization
config["shuffle"] = False                                                # Bool. Shuffle inputs

# Quick test
config["quick_test"] = False                                             # Bool. Quicktest case (fast training)
config["quick_test_stations"] = ["ALPE-D'HUEZ", 'LES ECRINS-NIVOSE', 'SOUM COUY-NIVOSE', 'SPONDE-NIVOSE']

# Input variables
config["input_variables"] = ['Wind', 'Wind_DIR']

# Labels
config["labels"] = ['vw10m(m/s)']                                       # ["vw10m(m/s)"] or ["U_obs", "V_obs"] or ['T2m(degC)']
config["wind_nwp_variables"] = ["Wind", "Wind_DIR"]                     # ["Wind", "Wind_DIR"] or ["U_AROME", "V_AROME"]

# Dataset
config["unbalanced_dataset"] = False

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
config["stations_test"] = ['Col du Lac Blanc', 'MMLAF', 'WAE', 'AGAAR', 'DIS', 'MMDAS', 'EMBRUN', 'ZER', 'LE TOUR', 'GRH', 'SRS', 'MMSRL', 'BER', 'GIH', 'VAD', 'TAE', 'EGO', 'TGAMR', 'JUN', 'DEM', 'BRZ', 'MMHIW', 'COL-DES-SAISIES', 'LE PLENAY', 'BONNEVAL-NIVOSE', 'MMCPY', 'Col de Porte', 'DIA', 'TGNOL', 'INNESF', 'LA MEIJE-NIVOSE', 'LEI', 'STK', 'SIO', 'ST-PIERRE-LES EGAUX', 'MAS', 'Argentiere', 'MEYTHET', 'BELLECOTE-NIVOSE', 'ASCROS', 'MMZWE', 'ROB', 'MFOFKP', 'WYN', 'MMBIR', 'TGEGN', 'CDM', "VILLAR D'ARENE", 'PARPAILLON-NIVOSE', 'INNRED', 'HLL', 'ULR', 'TGUSH', 'TGNUS', 'SIA']
config["stations_val"] = ['SBO', 'CEV', 'MMBOY', 'THU', 'AND', 'MMTIT', 'ROE', 'RESTEFOND-NIVOSE', 'La Muzelle Lac Blanc', 'BARCELONNETTE', 'GRANDE PAREI NIVOSE', 'GALIBIER-NIVOSE', 'ALBERTVILLE JO', 'MER', 'ARH', 'SPF', 'PLF', 'AIGLETON-NIVOSE']
config["stations_to_reject"] = ["Vallot", "Dome Lac Blanc", "MFOKFP"]

# Do not modify: assert inputs are correct
config = adapt_distribution_strategy_to_available_devices(config)
config = init_learning_rate_adapted(config)
config["nb_input_variables"] = len(config["input_variables"])
config = detect_variable(config)

list_variables = ['name', 'date', 'lon', 'lat', 'alti', 'T2m(degC)', 'vw10m(m/s)',
                  'winddir(deg)', 'HTN(cm)','Tair', 'T1', 'ts', 'Tmin', 'Tmax', 'Qair',
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
