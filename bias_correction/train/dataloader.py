import numpy as np
import pandas as pd
import tensorflow as tf
from copy import copy
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple, Union, Any

from bias_correction.train.metrics import get_metric


class TopoGenerator:

    def __init__(self, dict_topos, names):

        try:
            self.names = names.values
        except AttributeError:
            self.names = names

        self.dict_topos1 = dict_topos

    def __call__(self):
        for name in self.names:
            yield self.dict_topos1[name]["data"]


class MeanGenerator:

    def __init__(self, mean, length):
        self.mean = mean
        self.length = length

    def __call__(self):
        for i in range(self.length):
            yield self.mean


class Batcher:

    def __init__(self, config):
        self.config = config

    def _get_prefetch(self):
        if self.config.get("prefetch") == "auto":
            return tf.data.AUTOTUNE
        else:
            return self.config["prefetch"]

    def batch_train(self, dataset):
        return dataset \
            .batch(batch_size=self.config["global_batch_size"]) \
            .cache() \
            .prefetch(self._get_prefetch())

    def batch_test(self, dataset):
        raise NotImplementedError("Test data are not batched")

    def batch_val(self, dataset):
        return dataset.batch(batch_size=self.config["global_batch_size"])


class Splitter:

    def __init__(self, config):
        self.config = config

    def _split_by_time(self,
                       time_series: pd.DataFrame,
                       mode: Union[str, None] = None,
                       **kwargs: Any
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        assert mode is not None, "mode must be specified"

        time_series_train = time_series[time_series.index < self.config[f"date_split_train_{mode}"]]
        time_series_test = time_series[time_series.index >= self.config[f"date_split_train_{mode}"]]

        return time_series_train, time_series_test

    def _split_by_space(self,
                        time_series: pd.DataFrame,
                        mode: Union[str, None] = None,
                        **kwargs: Any
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        assert mode is not None, "mode must be specified"

        time_series_train = time_series[time_series["name"].isin(self.config[f"stations_train"])]
        time_series_test = time_series[time_series["name"].isin(self.config[f"stations_{mode}"])]

        return time_series_train, time_series_test

    def _split_random(self,
                      time_series: pd.DataFrame,
                      mode: Union[str, None] = None,
                      **kwargs: Any
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        assert mode is not None, "mode must be specified"

        str_test_size = f"random_split_test_size_{mode}"
        str_random_state = f"random_split_state_{mode}"

        if self.config["quick_test"] and mode == "test":
            test_size = 0.8
        elif self.config["quick_test"] and mode == "val":
            test_size = 0.01
        else:
            test_size = self.config[str_test_size]

        return train_test_split(time_series, test_size=test_size, random_state=self.config[str_random_state])

    def _split_time_and_space(self,
                              time_series: pd.DataFrame,
                              mode: Union[str, None] = None,
                              **kwargs: Any
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        assert mode is not None, "mode must be specified"

        time_series_train, time_series_test = self._split_by_space(time_series, mode)
        time_series_train, _ = self._split_by_time(time_series_train, mode)
        _, time_series_test = self._split_by_time(time_series_test, mode)

        return time_series_train, time_series_test

    def _split_by_country(self,
                          time_series: pd.DataFrame,
                          stations: Union[pd.DataFrame, None] = None,
                          **kwargs: Any
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        assert stations is not None, "stations pd.DataFrame must be specified"

        countries_to_reject = self.config["country_to_reject_during_training"]
        names_country_to_reject = stations["name"][stations["country"].isin(countries_to_reject)].values
        time_series_other_countries = time_series[time_series["name"].isin(names_country_to_reject)]
        time_series = time_series[~time_series["name"].isin(names_country_to_reject)]
        return time_series, time_series_other_countries

    def split_wrapper(self,
                      time_series: pd.DataFrame,
                      mode: str = "test",
                      stations: Union[pd.DataFrame, None] = None,
                      split_strategy: Union[str, None] = None,
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        split_strategy = self.config[f"split_strategy_{mode}"] if split_strategy is None else split_strategy

        strategies = {"time": self._split_by_time,
                      "space": self._split_by_space,
                      "time_and_space": self._split_time_and_space,
                      "random": self._split_random,
                      "country": self._split_by_country}

        time_series_train, time_series_test = strategies[split_strategy](time_series, mode=mode, stations=stations)

        return time_series_train, time_series_test

    def _check_split_strategy_is_correct(self) -> None:
        not_implemented_strategies_test = ["time", "space", "time", "random", "random", "random", "time", "space"]
        not_implemented_strategies_val = ["time", "time", "space", "space", "time", "time_and_space",
                                          "time_and_space", "time_and_space"]

        implemented_strategies_test = ["space", "random", "time", "space", "time_and_space", "time_and_space"]
        implemented_strategies_val = ["space", "random", "random", "random", "time_and_space", "random"]

        strat_t = self.config["split_strategy_test"]
        strat_v = self.config["split_strategy_val"]

        if (strat_t, strat_v) in zip(implemented_strategies_test, implemented_strategies_val):
            print(f"\nSplit strategy is implemented: test={strat_t}, val={strat_v}")
        elif (strat_t, strat_v) in zip(not_implemented_strategies_test, not_implemented_strategies_val):
            raise NotImplementedError(f"Split strategy test={strat_t} and val={strat_v} is not implemented")
        else:
            raise NotImplementedError("Split strategy not referenced")

    def split_train_test_val(self,
                             time_series: pd.DataFrame,
                             split_strategy: Union[str, None] = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        self._check_split_strategy_is_correct()

        strat_t = self.config["split_strategy_test"]
        strat_v = self.config["split_strategy_val"]

        time_series_train, time_series_test = self.split_wrapper(time_series,
                                                                 mode="test",
                                                                 split_strategy=split_strategy)

        if "time_and_space" == strat_t and "time_and_space" == strat_v:
            ts = time_series
        else:
            ts = time_series_train

        time_series_train, time_series_val = self.split_wrapper(ts,
                                                                mode="val",
                                                                split_strategy=split_strategy)
        return time_series_train, time_series_test, time_series_val


class Loader:

    def __init__(self, config):
        self.config = config

    def load_dict_topo(self):
        with open(self.config["topos_near_station"], 'rb') as f:
            dict_topos = pickle.load(f)

        y_l = 140 - 70
        y_r = 140 + 70
        x_l = 140 - 70
        x_r = 140 + 70
        for station in dict_topos:
            dict_topos[station]["data"] = np.reshape(dict_topos[station]["data"][y_l:y_r, x_l:x_r], (140, 140, 1))

        return dict_topos

    def open_time_series_pkl(self):
        return pd.read_pickle(self.config["time_series"])

    def open_stations_pkl(self):
        return pd.read_pickle(self.config["stations"])


class CustomDataHandler:

    def __init__(self, config, load_dict_topo=True):

        self.config = config
        self.variables_needed = copy(['name'] + self.config["input_variables"] + self.config["labels"])

        self.batcher = Batcher(config)
        self.splitter = Splitter(config)
        self.loader = Loader(config)

        if load_dict_topo:
            self.dict_topos = self.loader.load_dict_topo()

        # Attributes defined later
        self.inputs_train = None
        self.inputs_test = None
        self.inputs_val = None
        self.inputs_other_countries = None
        self.length_train = None
        self.length_test = None
        self.length_val = None
        self.length_other_countries = None
        self.labels_train = None
        self.labels_test = None
        self.labels_val = None
        self.labels_other_countries = None
        self.names_train = None
        self.names_test = None
        self.names_val = None
        self.names_other_countries = None
        self.mean_standardize = None
        self.std_standardize = None
        self.is_prepared = None
        self.predicted_train = None
        self.predicted_test = None
        self.predicted_val = None
        self.predicted_other_countries = None

    def _select_all_variables_needed(self, df, variables_needed=None):
        variables_needed = self.variables_needed if variables_needed is None else variables_needed
        return df[variables_needed]

    def add_topo_carac_time_series(self, time_series, stations):
        for topo_carac in ["tpi_500", "curvature", "laplacian", "mu"]:
            if topo_carac in self.config["input_variables"]:
                time_series.loc[:, topo_carac] = np.nan
                for station in time_series["name"].unique():
                    value_topo_carac = stations.loc[stations["name"] == station, topo_carac + "_NN_0"].values[0]
                    time_series.loc[time_series["name"] == station, topo_carac] = value_topo_carac
        return time_series

    def reject_stations(self, time_series, stations):
        """Reject stations from inputs files as defined by the user"""
        time_series = time_series[~time_series["name"].isin(self.config["stations_to_reject"])]
        stations = stations[~stations["name"].isin(self.config["stations_to_reject"])]

        return time_series, stations

    def reject_country(self, time_series, stations):
        """Reject stations from inputs files as defined by the user"""
        countries_to_reject = self.config["country_to_reject_during_training"]
        names_country_to_reject = stations["name"][stations["country"].isin(countries_to_reject)].values
        time_series = time_series[~time_series["name"].isin(names_country_to_reject)]
        stations = stations[~stations["name"].isin(names_country_to_reject)]
        return time_series, stations

    def _get_stations_test_and_val(self):
        if "space" in self.config["split_strategy_test"] and "space" in self.config["split_strategy_val"]:
            return self.config["stations_test"] + self.config["stations_val"]
        elif "space" in self.config["split_strategy_test"]:
            return self.config["stations_test"]
        elif "space" in self.config["split_strategy_val"]:
            return self.config["stations_test"]
        else:
            return []

    def _get_train_stations(self, df):
        assert "name" in df, "DataFrame must contain a name column"
        all_stations = df["name"].unique()
        stations_test_val = self._get_stations_test_and_val()
        return [s for s in all_stations if s not in set(stations_test_val)]

    def unbalance_training_dataset(self):
        if self.config["current_variable"] == "T2m":
            raise NotImplementedError("Unbalanced dataset is not implemented for temperature")

        bool_train_labels = self.labels_train["vw10m(m/s)"] >= self.config["unbalanced_threshold"]
        bool_train_labels = bool_train_labels.values

        pos_features = self.inputs_train[bool_train_labels].values
        neg_features = self.inputs_train[~bool_train_labels].values

        pos_labels = self.labels_train[bool_train_labels].values
        neg_labels = self.labels_train[~bool_train_labels].values

        pos_names = self.names_train[bool_train_labels].values
        neg_names = self.names_train[~bool_train_labels].values

        ids = np.arange(len(neg_features))
        choices = np.random.choice(ids, len(pos_features))

        res_neg_features = neg_features[choices]
        res_neg_labels = neg_labels[choices]
        res_neg_names = neg_names[choices]

        assert len(res_neg_features) == len(pos_features)
        assert len(res_neg_labels) == len(pos_labels)
        assert len(res_neg_names) == len(pos_names)

        self.inputs_train = np.concatenate([res_neg_features, pos_features], axis=0)
        self.labels_train = np.concatenate([res_neg_labels, pos_labels], axis=0)
        self.names_train = np.concatenate([res_neg_names, pos_names], axis=0)
        self.length_train = len(self.inputs_train)

    @staticmethod
    def _try_random_choice(list_stations, station, patience=10, stations_to_exclude=[]):
        i = 0
        while i < patience:
            station_name = np.random.choice(station["name"].values)
            _is_aiguille_du_midi = station_name == "AGUIL. DU MIDI"
            station_already_selected = (station_name in list_stations) or (
                    station_name in stations_to_exclude) or _is_aiguille_du_midi
            print(station_name)
            print(station_name in list_stations)
            print(station_name in stations_to_exclude)
            print(_is_aiguille_du_midi)
            if station_already_selected:
                print(station_name, patience)
                i += 1
            else:
                list_stations.append(station_name)
                i = patience + 1
        return list_stations

    @staticmethod
    def add_nwp_stats_to_stations(stations, time_series, metrics=["rmse"]):

        for metric in metrics:
            stations[metric] = np.nan
            for station in time_series["name"].unique():
                filter_station = time_series["name"] == station
                time_series_station = time_series.loc[filter_station, ["Wind", "vw10m(m/s)"]].dropna()
                metric_func = get_metric(metric)
                try:
                    mean_metric = metric_func(time_series_station["vw10m(m/s)"].values,
                                              time_series_station["Wind"].values)
                    stations.loc[stations["name"] == station, [metric]] = mean_metric
                except ValueError:
                    print(station)

        return stations

    def add_mode_to_df(self, df):

        assert self.is_prepared

        df["mode"] = np.nan

        filter_test = df["name"].isin(self.get_names("test", unique=True))
        filter_val = df["name"].isin(self.get_names("val", unique=True))
        filter_train = df["name"].isin(self.get_names("train", unique=True))
        filter_other = df["name"].isin(self.get_names("other_countries", unique=True))
        filter_custom = df["name"].isin(self.get_names("custom", unique=True))
        filter_rejected = ~(filter_test | filter_val | filter_train | filter_other | filter_custom)

        df.loc[filter_test, "mode"] = "Test"
        df.loc[filter_val, "mode"] = "Validation"
        df.loc[filter_train, "mode"] = "Training"
        df.loc[filter_other, "mode"] = "other_countries"
        df.loc[filter_rejected, "mode"] = "rejected"
        df.loc[filter_custom, "mode"] = "custom"

        return df

    @staticmethod
    def add_country_to_time_series(time_series, stations):
        time_series["country"] = np.nan
        for station in time_series["name"].unique():
            filter_s = stations["name"] == station
            filter_ts = time_series["name"] == station
            time_series.loc[filter_ts, "country"] = stations.loc[filter_s, "country"].values[0]
        return time_series

    @staticmethod
    def add_elevation_category_to_df(df, list_min=[0, 1000, 2000, 3000], list_max=[1000, 2000, 3000, 5000]):
        df["cat_zs"] = np.nan
        for z_min, z_max in zip(list_min, list_max):
            filter_alti = (z_min <= df["alti"]) & (df["alti"] < z_max)
            df.loc[filter_alti, ["cat_zs"]] = f"{int(z_min)}m $\leq$ Station elevation $<$ {int(z_max)}m"
        return df

    def _select_randomly_test_val_stations(self, time_series, stations, mode, stations_to_exclude=[]):
        metric = self.config["metric_split"]

        stations = self.add_nwp_stats_to_stations(stations, time_series, [metric])

        if mode == "test":
            list_parameters = self.config["parameters_split_test"]
            list_stations = ["Col du Lac Blanc"]
        else:
            list_parameters = self.config["parameters_split_val"]
            list_stations = []

        for parameter in list_parameters:
            print(f"Parameter: {parameter}")
            q33 = np.quantile(stations[parameter].values, 0.33)
            q66 = np.quantile(stations[parameter].values, 0.66)

            small_values = stations[stations[parameter].values <= q33]
            medium_values = stations[(q33 <= stations[parameter].values) & (stations[parameter].values < q66)]
            large_values = stations[q66 <= stations[parameter].values]

            for index, stratified_stations in enumerate([small_values, medium_values, large_values]):
                q33_strat = np.nanquantile(stratified_stations[metric].values, 0.33)
                q66_strat = np.nanquantile(stratified_stations[metric].values, 0.66)

                first_q = stratified_stations[metric] < q33_strat
                second_q = (q33_strat <= stratified_stations[metric]) & (stratified_stations[metric] < q66_strat)
                third_q = q66_strat <= stratified_stations[metric]

                strat_0 = stratified_stations[first_q]
                strat_1 = stratified_stations[second_q]
                strat_2 = stratified_stations[third_q]

                for idx, station in enumerate([strat_0, strat_1, strat_2]):
                    list_stations = self._try_random_choice(list_stations,
                                                            station,
                                                            patience=10,
                                                            stations_to_exclude=stations_to_exclude)

        return list_stations

    def define_test_and_val_stations(self, time_series, stations):
        i = 0

        time_series, stations = self.reject_country(time_series, stations)

        self.config["stations_test"] = []
        self.config["stations_val"] = []
        col_du_lac_blanc_not_in_test = "Col du Lac Blanc" not in self.config["stations_test"]
        aiguille_du_midi_in_test = "AGUIL. DU MIDI" in self.config["stations_test"]
        aiguille_du_midi_in_val = "AGUIL. DU MIDI" in self.config["stations_val"]
        aiguille_du_midi_not_in_train = aiguille_du_midi_in_test or aiguille_du_midi_in_val
        while col_du_lac_blanc_not_in_test or aiguille_du_midi_not_in_train:
            i += 1
            self.config["stations_test"] = self._select_randomly_test_val_stations(time_series,
                                                                                   stations,
                                                                                   mode="test")
            self.config["stations_val"] = self._select_randomly_test_val_stations(time_series,
                                                                                  stations,
                                                                                  mode="val",
                                                                                  stations_to_exclude=self.config[
                                                                                      "stations_test"])

            col_du_lac_blanc_not_in_test = "Col du Lac Blanc" not in self.config["stations_test"]
            aiguille_du_midi_in_test = "AGUIL. DU MIDI" in self.config["stations_test"]
            aiguille_du_midi_in_val = "AGUIL. DU MIDI" in self.config["stations_val"]
            aiguille_du_midi_not_in_train = aiguille_du_midi_in_test | aiguille_du_midi_in_val

    def _apply_quick_test(self, time_series):
        return time_series[time_series["name"].isin(self.config["quick_test_stations"])]

    def shuffle_eventually(self, time_series, _shuffle=None):
        _shuffle = self.config["shuffle"] if _shuffle is None else _shuffle
        if _shuffle:
            return shuffle(time_series)
        else:
            return time_series

    def prepare_train_test_data(self, _shuffle=True, variables_needed=None):

        # Pre-processing time_series
        time_series = self.loader.open_time_series_pkl()
        stations = self.loader.open_stations_pkl()

        # Reject stations
        time_series, stations = self.reject_stations(time_series, stations)

        # Add topo characteristics
        time_series = self.add_topo_carac_time_series(time_series, stations)

        # Add country
        time_series = self.add_country_to_time_series(time_series, stations)

        # Select variables
        time_series = self._select_all_variables_needed(time_series, variables_needed)

        # Define test/train/val stations if random
        if self.config["stations_test"] == "random" and self.config["stations_val"] == "random":
            self.define_test_and_val_stations(time_series, stations)

        # Dropna
        time_series = time_series.dropna()

        # Quick test
        if self.config["quick_test"]:
            time_series = self._apply_quick_test(time_series)

        # Shuffle
        time_series = self.shuffle_eventually(time_series)

        # Split time_series with countries
        if self.config["country_to_reject_during_training"]:
            time_series, time_series_other_countries = self.splitter.split_wrapper(time_series,
                                                                                   stations=stations,
                                                                                   split_strategy="country")

        # Stations train
        self.config[f"stations_train"] = self._get_train_stations(time_series)

        # train/test split
        split_strategy = "random" if self.config[f"quick_test"] else None
        time_series_train, time_series_test, time_series_val = self.splitter.split_train_test_val(time_series,
                                                                                                  split_strategy=split_strategy)
        # Other countries
        if self.config["country_to_reject_during_training"]:
            _, time_series_other_countries = self.splitter.split_wrapper(time_series_other_countries,
                                                                         mode="test",
                                                                         split_strategy="time")

        # Input variables
        self.inputs_train = time_series_train[self.config["input_variables"]]
        self.inputs_test = time_series_test[self.config["input_variables"]]
        self.inputs_val = time_series_val[self.config["input_variables"]]
        if self.config["country_to_reject_during_training"]:
            self.inputs_other_countries = time_series_other_countries[self.config["input_variables"]]

        # Length
        self.length_train = len(self.inputs_train)
        self.length_test = len(self.inputs_test)
        self.length_val = len(self.inputs_val)
        if self.config["country_to_reject_during_training"]:
            self.length_other_countries = len(self.inputs_other_countries)

        # labels
        self.labels_train = time_series_train[self.config["labels"]]
        self.labels_test = time_series_test[self.config["labels"]]
        self.labels_val = time_series_val[self.config["labels"]]
        if self.config["country_to_reject_during_training"]:
            self.labels_other_countries = time_series_other_countries[self.config["labels"]]

        # names
        self.names_train = time_series_train["name"]
        self.names_test = time_series_test["name"]
        self.names_val = time_series_val["name"]
        if self.config["country_to_reject_during_training"]:
            self.names_other_countries = time_series_other_countries["name"]

        if self.config["standardize"]:
            self.mean_standardize = self.inputs_train.mean()
            self.std_standardize = self.inputs_train.std()

        if self.config["unbalanced_dataset"]:
            self.unbalance_training_dataset()

        self._set_is_prepared()

    def get_inputs(self, mode):
        return getattr(self, f"inputs_{mode}")

    def get_length(self, mode):
        return getattr(self, f"length_{mode}")

    def get_labels(self, mode):
        return getattr(self, f"labels_{mode}")

    def get_names(self, mode, unique=False):
        names = getattr(self, f"names_{mode}")

        if unique:
            return list(names.unique())
        else:
            return names

    def get_mean(self):
        return self.mean_standardize

    def get_std(self):
        return self.std_standardize

    def get_tf_topos(self, mode, names=None):

        if names is None:
            names = self.get_names(mode)

        topos_generator = TopoGenerator(self.dict_topos, names)

        return tf.data.Dataset.from_generator(topos_generator, output_types=tf.float32, output_shapes=(140, 140, 1))

    def get_tf_mean_std(self, mode):
        length = self.get_length(mode)
        mean = self.get_mean()
        std = self.get_std()

        mean = tf.data.Dataset.from_generator(MeanGenerator(mean,
                                                            length),
                                              output_types=tf.float32,
                                              output_shapes=(self.config["nb_input_variables"],))
        std = tf.data.Dataset.from_generator(MeanGenerator(std,
                                                           length),
                                             output_types=tf.float32,
                                             output_shapes=(self.config["nb_input_variables"],))
        return mean, std

    def get_tf_zipped_inputs(self, mode="test", inputs=None, names=None):

        if inputs is None:
            inputs = self.get_inputs(mode)

        if hasattr(inputs, "values"):
            inputs = inputs.values

        inputs = tf.data.Dataset.from_tensor_slices(inputs)

        if self.config["standardize"]:
            mean, std = self.get_tf_mean_std(mode)
            return tf.data.Dataset.zip((self.get_tf_topos(mode=mode, names=names), inputs, mean, std))
        else:
            return tf.data.Dataset.zip((self.get_tf_topos(mode=mode, names=names), inputs))

    def get_tf_zipped_inputs_labels(self, mode):
        labels = self.get_labels(mode)

        if hasattr(labels, "values"):
            labels = labels.values

        labels = tf.data.Dataset.from_tensor_slices(labels)
        inputs = self.get_tf_zipped_inputs(mode=mode)

        return tf.data.Dataset.zip((inputs, labels))

    def get_time_series(self,
                        prepared=False,
                        mode=True):
        time_series = self.loader.open_time_series_pkl()
        if prepared:

            assert self.is_prepared

            stations = self.get_stations()

            # Reject stations
            time_series, stations = self.reject_stations(time_series, stations)

            # Add topo characteristics
            time_series = self.add_topo_carac_time_series(time_series, stations)

            # Add country
            time_series = self.add_country_to_time_series(time_series, stations)

            # Add mode
            if mode:
                time_series = self.add_mode_to_df(time_series)

            # Select variables
            time_series = self._select_all_variables_needed(time_series)

            time_series = time_series.dropna()

            return time_series
        else:
            return time_series

    def get_stations(self,
                     add_mode: Optional[bool] = False):
        stations = self.loader.open_stations_pkl()
        if add_mode:
            stations = self.add_mode_to_df(stations)
        return stations

    def get_predictions(self,
                        mode: str):
        try:
            return getattr(self, f"predicted_{mode}")
        except AttributeError:
            try:
                return getattr(self, f"predicted{mode}")
            except AttributeError:
                raise NotImplementedError("We only support modes train/test/val/other_countries/devine")

    def get_topos(self,
                  mode=None,
                  names=None):
        if names is None:
            names = self.get_names(mode)

        if hasattr(names, "values"):
            names = names.values

        topos_generator = TopoGenerator(self.dict_topos, names)

        return topos_generator()

    def get_batched_inputs_labels(self,
                                  mode: str):
        dataset = self.get_tf_zipped_inputs_labels(mode)

        batch_func = {"train": self.batcher.batch_train,
                      "test": self.batcher.batch_test,
                      "val": self.batcher.batch_val}

        return batch_func[mode](dataset)

    def _nn_output2df(self, result, mode, name_uv="UV_nn"):

        df = pd.DataFrame()
        df["name"] = self.get_names(mode)
        if "component" in self.config["type_of_output"]:

            df[name_uv] = np.sqrt(result[0] ** 2 + result[1] ** 2)

        else:

            df[name_uv] = np.squeeze(result)

        return df[["name", name_uv]]

    def _set_predictions(self, results, mode="test", str_model="_nn"):
        filter_int = isinstance(results, tuple) and len(results) > 1
        has_intermediate_outputs = filter_int and self.config["get_intermediate_output"]
        if has_intermediate_outputs and not (str_model == "_int"):
            results = results[0]
            str_model = str_model
            mode_str = mode
        elif has_intermediate_outputs and (str_model == "_int"):
            results = results[1][:, 0]
            str_model = "_int"
            mode_str = "int"
        else:
            str_model = str_model
            mode_str = mode

        name_uv = f"{self.config['current_variable']}{str_model}"
        df = self._nn_output2df(results, mode, name_uv=name_uv)
        setattr(self, f"predicted_{mode_str}", df)

    def set_predictions(self, results, mode="test", str_model="_nn"):
        self._set_predictions(results, mode=mode, str_model=str_model)
        if self.config["get_intermediate_output"]:
            self._set_predictions(results, mode=mode, str_model="_int")

    def _set_is_prepared(self):
        self.is_prepared = True

    def add_model(self, model, mode="test"):
        path_to_files = {"_D": self.config["path_to_devine"] + f"devine_2022_08_04_v4_{mode}.pkl",
                         "_A": self.config["path_to_analysis"] + "time_series_bc_a.pkl"}

        predictions = pd.read_pickle(path_to_files[model])

        if model == "_A":
            predictions = predictions.rename(columns={"Wind": "UV_A"})

        setattr(self, f"predicted{model}", predictions)
