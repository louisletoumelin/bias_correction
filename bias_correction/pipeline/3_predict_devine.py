import numpy as np
import pandas as pd
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint

import sys
sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_devine import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, Interpretability

# Initialization
#  x[:, 0] is nwp wind speed.
#  x[:, 1] is wind direction.
exp = ExperienceManager(config)
data_loader = CustomDataHandler(config)
cm = CustomModel(exp, config)
cm.build_model_with_strategy()

print("\nCurrent experience:")
print(exp.path_to_current_experience)

print("\nConfig")
print(pprint(config))

# Load inputs and outputs
data_loader.prepare_train_test_data(variables_needed=["Wind", "Wind_DIR", "vw10m(m/s)", "name"], add_topo_carac=False)

with tf.device('/GPU:0'):

    # Predict
    with timer_context("Predict test set"):
        inputs_test = data_loader.get_tf_zipped_inputs(mode="train").batch(data_loader.length_train)  #data_loader.length_train
        results_test = [cm.model.predict(i) for i in inputs_test]
        #results_test = cm.predict_single_bath(inputs_test, force_build=True)

        #results_test = cm.predict_single_bath(inputs_test, force_build=True)

    # Predict
    #with timer_context("Predict Pyrénées and Corsica"):
    #    inputs_other_countries = data_loader.get_tf_zipped_inputs(mode="other_countries")\
    #        .batch(data_loader.length_other_countries)

    #    results_other_countries = cm.predict_single_bath(inputs_other_countries)

for mode, result in zip(["train"], [results_test]):
    data_loader.set_predictions(result, mode=mode, str_model="_D")
    df = data_loader.get_predictions(mode)
    print("exp.path_to_predictions")
    print(exp.path_to_predictions)
    df.to_pickle(exp.path_to_predictions+f"devine_{mode}_split_v2.pkl")


#for mode, result in zip(["other_countries"], [results_other_countries]):
#    data_loader.set_predictions(result, mode=mode, str_model="_D")
#    df = data_loader.get_predictions(mode)
#    print("exp.path_to_predictions")
#    print(exp.path_to_predictions)
#    df.to_pickle(exp.path_to_predictions+f"devine_2023_01_25_ddsdir_{mode}.pkl")

# Create input files with information about stations
# 1. Create stations.csv with columns name, lat, lon, alt and country and move the file to the station folder
# 2. Change the path to stations.csv in 0_preprocess.py.
# 3.0 comment #s.update_station_with_topo_characteristics()
# 3.1 comment #s.change_dtype_stations()

# 4. Add time_series.pkl to the 0_preprocess.py and modify line that read time_series in config_preprocess.py
# 5. Modify stations = pd.read_pickle(config["path_stations_pre_processed"] + f"stations_bc{NAME}.pkl")
# 6. Modify time_series = pd.read_pickle(config["path_time_series_pre_processed"] + "time_series_with_clb.pkl")

# Preprocess time_series.pkl file (not necessary if you already have NWP columns in time_series.pkl)
# 7. set config["pre_process_time_series"] = True in config_preprocess.py
# 8. set config["pre_process_stations"] = False in config_preprocess.py
# send stations.pkl and time_series.pkl to labia in 1_Raw/
# send stations.pkl in 2_Pre_Processed/
# send files to labia

# Modify config_devine.py

