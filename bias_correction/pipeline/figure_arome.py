import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from bias_correction.config.config_double_v1 import config
from bias_correction.train.metrics import bias
from bias_correction.train.model import CustomModel
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.utils_bc.print_functions import print_headline
from bias_correction.train.visu import save_figure


def add_topo_carac_time_series(time_series: pd.DataFrame,
                               stations: pd.DataFrame,
                               topo_carac: str ="tpi_500"
                               ) -> pd.DataFrame:
    time_series.loc[:, topo_carac] = np.nan
    for station in time_series["name"].unique():
        value_topo_carac = stations.loc[stations["name"] == station, topo_carac + "_NN_0"].values[0]
        time_series.loc[time_series["name"] == station, topo_carac] = value_topo_carac
    return time_series


def classify_topo_carac(stations: pd.DataFrame,
                        df: pd.DataFrame,
                        topo_carac: list = ['mu', 'curvature', 'tpi_500', 'tpi_2000', 'laplacian', 'alti']
                        ) -> pd.DataFrame:
    for carac in topo_carac:

        if not hasattr(df, f"class_{carac}"):
            df[f"class_{carac}"] = np.nan

        carac_nn = carac if carac not in ["alti", "country"] else carac

        if carac not in ["alti", "country"]:
            q25 = np.nanquantile(stations[carac_nn+"_NN_0"].values, 0.25)
            q50 = np.nanquantile(stations[carac_nn+"_NN_0"].values, 0.5)
            q75 = np.nanquantile(stations[carac_nn+"_NN_0"].values, 0.75)
        else:
            q25 = np.nanquantile(stations[carac_nn].values, 0.25)
            q50 = np.nanquantile(stations[carac_nn].values, 0.5)
            q75 = np.nanquantile(stations[carac_nn].values, 0.75)

        filter_1 = (df[carac_nn] <= q25)
        filter_2 = (df[carac_nn] > q25) & (df[carac_nn] <= q50)
        filter_3 = (df[carac_nn] > q50) & (df[carac_nn] <= q75)
        filter_4 = (df[carac_nn] > q75)

        df.loc[filter_1, f"class_{carac}"] = "0 $x \leq q_{25}$"
        df.loc[filter_2, f"class_{carac}"] = "1 $q_{25}<x \leq q_{50}$"
        df.loc[filter_3, f"class_{carac}"] = "2 $q_{50}<x \leq q_{75}$"
        df.loc[filter_4, f"class_{carac}"] = "3 $q_{75}<x$"

        print(f"Quantiles {carac}: ", q25, q50, q75)

    return df

print_headline("Restore experience", "")
exp, config = ExperienceManager.from_previous_experience(config["restore_experience"])
cm = CustomModel.from_previous_experience(exp, config, "last")

arome = pd.read_pickle(config["time_series"])
stations = pd.read_pickle(config["stations"])

arome = arome[["name", "alti", "vw10m(m/s)", "Wind"]].dropna()
arome["bias"] = bias(arome["vw10m(m/s)"].values, arome["Wind"].values)
arome = add_topo_carac_time_series(arome, stations, "tpi_500")
arome = classify_topo_carac(stations, arome, ["tpi_500"])
arome = classify_topo_carac(stations, arome, ["alti"])

nb_stations = arome[["class_alti", "class_tpi_500", "bias", "name"]].groupby(["class_alti", "class_tpi_500"]).nunique().reset_index().pivot(index='class_alti', columns='class_tpi_500', values='name')
df_pivot = arome[["class_alti", "class_tpi_500", "bias"]].groupby(["class_alti", "class_tpi_500"]).mean().reset_index().pivot(index='class_alti', columns='class_tpi_500', values='bias')
vcenter = 0
vmin, vmax = df_pivot.min().min(), df_pivot.max().max()
#normalize = mcolors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
normalize = mcolors.CenteredNorm(vcenter=0)
palette = sns.diverging_palette(220, 20, as_cmap=True)

plt.figure()
fontsize=10
sns.heatmap(df_pivot,
            annot=nb_stations,
            vmin=vmin,
            vmax=vmax,
            center=0,
            linewidths=.5,
            cmap=palette)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=fontsize)
plt.xlabel("TPI category of the observation station")
plt.ylabel("Elevation category of the observation station")
plt.axis("equal")
plt.tight_layout()
save_figure("AROME_heatmap/AROME_heatmap_tpi_alti", exp=exp, svg=True)