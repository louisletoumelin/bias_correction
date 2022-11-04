import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns

    _sns = True
except ModuleNotFoundError:
    _sns = False
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from pprint import pprint
import sys
import pickle

sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")

from bias_correction.config.config_train import config
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.eval import CustomEvaluation, Interpretability
from bias_correction.utils_bc.print_functions import print_intro
from bias_correction.utils_bc.load_my_modules import append_module_path
append_module_path(config, names=["downscale"])

path = config["path_time_series_pre_processed"]


with open(path + "dict_/dict_all.pickle", 'rb') as handle:
    results = pickle.load(handle)


#
#
# Figure 1
#
#
plt.figure()
list_df = []
for i in range(5):
    #plt.figure()
    i = str(i)
    values = results['length_constant_speed'][i]['<1m/s']['values']
    df = pd.DataFrame()
    df["values"] = values
    df["resolution"] = i
    print(len(values))
    #plt.hist(values, bins=100, density=True)
    list_df.append(df)
df = pd.concat(list_df)
df.index = list(range(len(df)))
g = sns.boxplot(data=df, x="resolution", y="values", palette="viridis")
for i in range(5):
    i = str(i)
    criteria = results['length_constant_speed'][i]['<1m/s']['stats']['criteria']
    P95 = results['length_constant_speed'][i]['<1m/s']['stats']['P95']
    P75 = results['length_constant_speed'][i]['<1m/s']['stats']['P75']
    P25 = results['length_constant_speed'][i]['<1m/s']['stats']['P25']
    criteria_theoretical = P95 + 7.5 * 1.5 * (P75-P25)
    plt.plot(i, criteria, color='red', marker='_', markersize=20)
    plt.plot(i, criteria_theoretical, color='orange', marker='_', markersize=20)
ax = plt.gca()
#ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel("Length of observed wind speed \n constant sequence [hours]\n for wind speed < 1 $m\:s^{-1}$")
plt.xlabel("r such as $10^{-r}$ = resolution of the observed signal \n (as interpreted by the quality control algorithm)")
plt.tight_layout()



#
#
# Figure 2
#
#
plt.figure()
list_df = []
for i in range(5):
    i = str(i)
    values = results['length_constant_speed'][i]['>=1m/s']['values']
    df = pd.DataFrame()
    df["values"] = values
    df["resolution"] = i
    print(len(values))
    #plt.hist(values, bins=100, density=True)
    list_df.append(df)
df = pd.concat(list_df)
df.index = list(range(len(df)))
g = sns.boxplot(data=df, x="resolution", y="values", palette="viridis")
for i in range(5):
    i = str(i)
    criteria = results['length_constant_speed'][i]['>=1m/s']['stats']['criteria']
    P95 = results['length_constant_speed'][i]['>=1m/s']['stats']['P95']
    P75 = results['length_constant_speed'][i]['>=1m/s']['stats']['P75']
    P25 = results['length_constant_speed'][i]['>=1m/s']['stats']['P25']
    criteria_theoretical = P95 + 7.5 * 1.5 * (P75-P25)
    plt.plot(i, criteria, color='red', marker='_', markersize=20)
    plt.plot(i, criteria_theoretical, color='orange', marker='_', markersize=20)
ax = plt.gca()
#ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel("Length of observed wind speed \n constant sequence [hours]\n for wind speed >= 1 $m\:s^{-1}$")
plt.xlabel("r such as $10^{-r}$ = resolution of the observed signal \n (as interpreted by the quality control algorithm)")
plt.tight_layout()




#
#
# Figure 3
#
#
plt.figure()
list_df = []
for i in ['10', '5', '1', '0.1']:
    # plt.figure()
    values = results['length_constant_direction'][i]['!=0']['values']
    df = pd.DataFrame()
    df["values"] = values
    df["resolution"] = i
    print(len(values))
    list_df.append(df)
df = pd.concat(list_df)
df.index = list(range(len(df)))
g = sns.boxplot(data=df, x="resolution", y="values", palette="viridis")
for i in ['10', '5', '1', '0.1']:
    criteria = results['length_constant_direction'][i]['!=0']['stats']['criteria']
    P95 = results['length_constant_direction'][i]['!=0']['stats']['P95']
    P75 = results['length_constant_direction'][i]['!=0']['stats']['P75']
    P25 = results['length_constant_direction'][i]['!=0']['stats']['P25']
    criteria_theoretical = P95 + 15 * 1.5 * (P75-P25)
    plt.plot(i, criteria, color='red', marker='_', markersize=20)
    plt.plot(i, criteria_theoretical, color='orange', marker='_', markersize=20)

ax = plt.gca()
# ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel("Length of observed wind direction \n constant sequence [hours]\n for wind speed > 0 $m\:s^{-1}$")
plt.xlabel(
    "r such as $10^{-r}$ = resolution of the observed signal \n (as interpreted by the quality control algorithm)")
plt.tight_layout()


#
#
# Figure 4 High variability
#
#
plt.figure()
with open(path + "dict_/dict_high_variability_small.pickle", 'rb') as handle:
    results = pickle.load(handle)

list_df = []
for time_step in [1, 2, 5, 10, 15, 20, 23]:
    values = []
    for station in results.keys():
        values.extend(results[station][time_step]["values"]['>=0'])
    df = pd.DataFrame()
    df["values"] = values
    time_step = str(time_step)
    df["time_step"] = time_step
    list_df.append(df)
df = pd.concat(list_df)
df.index = list(range(len(df)))
g = sns.boxplot(data=df, x="time_step", y="values", palette="viridis")
for time_step in [1, 2, 5, 10, 15, 20, 23]:
    P95 = np.nanquantile(results[station][time_step]["values"]['>=0'], 0.95)
    P75 = np.nanquantile(results[station][time_step]["values"]['>=0'], 0.75)
    P25 = np.nanquantile(results[station][time_step]["values"]['>=0'], 0.25)
    criteria = P95 + 8.9 * (P75-P25)
    time_step = str(time_step)
    plt.plot(time_step, 7.5, color='orange', marker='_', markersize=20)
    plt.plot(time_step, criteria, color='red', marker='_', markersize=20)

ax = plt.gca()
# ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel("Magnitude of wind speed increases [$m\:s^{-1}$]")
plt.xlabel("$\Delta_{t}$ [hour]")
plt.tight_layout()


#
#
# Figure 5 High variability
#
#
plt.figure()
list_df = []
for time_step in [1, 2, 5, 10, 15, 20, 23]:
    values = []
    for station in results.keys():
        values.extend(results[station][time_step]["values"]['<0'])
    df = pd.DataFrame()
    df["values"] = values
    time_step = str(time_step)
    df["time_step"] = time_step
    list_df.append(df)
df = pd.concat(list_df)
df.index = list(range(len(df)))
g = sns.boxplot(data=df, x="time_step", y="values", palette="viridis")
for time_step in [1, 2, 5, 10, 15, 20, 23]:
    P95 = np.nanquantile(results[station][time_step]["values"]['<0'], 0.95)
    P75 = np.nanquantile(results[station][time_step]["values"]['<0'], 0.75)
    P25 = np.nanquantile(results[station][time_step]["values"]['<0'], 0.25)
    criteria = P95 + 8.9 * (P75-P25)
    time_step = str(time_step)
    plt.plot(time_step, 7.5, color='orange', marker='_', markersize=20)
    plt.plot(time_step, criteria, color='red', marker='_', markersize=20)

ax = plt.gca()
# ax.set_xscale('log')
ax.set_yscale('log')
plt.ylabel("Magnitude of wind speed decreases [$m\:s^{-1}$]")
plt.xlabel("$\Delta_{t}$ [hour]")
plt.tight_layout()
