import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import date
import os
import json
from typing import Union, MutableSequence, Tuple

from bias_correction.utils_bc.network import detect_network
from bias_correction.train.eval import CustomEvaluation
from bias_correction.train.model import CustomModel
from bias_correction.train.dataloader import CustomDataHandler


def _is_full_path(path_to_previous_exp: str) -> bool:
    if len(path_to_previous_exp.split("/")) == 1:
        full_path = False
    elif len(path_to_previous_exp.split("/")) == 2 and path_to_previous_exp.split("/")[-1] == '':
        full_path = False
    else:
        full_path = True

    return full_path


def _get_path_with_root(path_to_previous_exp: str, network: str) -> str:
    pt = {"local": "/home/letoumelinl/bias_correction/Data/3_Predictions/Experiences/" + path_to_previous_exp,
          "labia": "//scratch/mrmn/letoumelinl/bias_correction/Data/3_Predictions/Experiences/" + path_to_previous_exp}
    return pt[network]


def _get_full_path_to_previous_exp(path_to_previous_exp: str) -> str:
    network = detect_network()
    if _is_full_path(path_to_previous_exp):
        return path_to_previous_exp
    else:
        return _get_path_with_root(path_to_previous_exp, network)


def labia2local(config: dict) -> dict:
    for key in config:
        if isinstance(config[key], str):
            if "//scratch/mrmn" in config[key]:
                config[key] = config[key].replace("//scratch/mrmn", "//home")
            if "//home/mrmn" in config[key]:
                config[key] = config[key].replace("//home/mrmn", "//home")
    return config


class FolderShouldNotExistError(Exception):
    pass


class AllExperiences:

    def __init__(self,
                 config: dict,
                 override: bool = False,
                 create: bool = True
                 ) -> None:

        self.config = config
        self.path_experiences = config["path_experiences"]

        if create:
            self.create_csv_file_if_doesnt_exists("experiences",
                                                  ["exp", "finished", "details"],
                                                  override=override)
            self.create_csv_file_if_doesnt_exists("metrics",
                                                  ["exp", "finished", "details", "MAE_nn",
                                                   "MAE_a", "RMSE_nn", "RMSE_a", "MB_n", "MB_a"],
                                                  override=override)
            self.create_csv_file_if_doesnt_exists("hyperparameters",
                                                  ["exp", "finished", "hyperparameter1", "hyperparameter2", "details"],
                                                  override=override)

    def check_csv_file_exists(self, name: str) -> bool:
        return f"{name}.csv" in os.listdir(self.path_experiences)

    def create_csv_file_if_doesnt_exists(self,
                                         name: str,
                                         columns: list[str],
                                         override: bool = False
                                         ) -> None:
        df = pd.DataFrame(columns=columns)
        df.index.name = "index"
        file_doesnt_exists = not self.check_csv_file_exists(name)
        if file_doesnt_exists or override:
            df.to_csv(self.path_experiences + f"{name}.csv", index=False)


class ExperienceManager(AllExperiences):

    def __init__(self,
                 config: dict,
                 override: bool = False,
                 restore_old_experience: bool = False,
                 create: bool = True
                 ) -> None:

        super().__init__(config, override=override, create=create)
        self.list_physical_devices()

        if not restore_old_experience:

            # Current and previous experiences
            self.current_date = self._get_current_date_str()
            self.other_experiences_created_today = self._get_experiences_created_today()
            self.current_id = self._get_current_id()
            self.name_current_experience = self._get_name_current_experience()
            self.is_finished = 0

            # Paths
            path_to_current_experience = self._get_path_to_current_experience()
            self.dict_paths = {"path_to_current_experience": path_to_current_experience,
                               "path_to_logs": path_to_current_experience + "logs/",
                               "path_to_best_model": path_to_current_experience + "best_model/",
                               "path_to_last_model": path_to_current_experience + "last_model/",
                               "path_to_tensorboard_logs": path_to_current_experience + "tensorboard_logs/",
                               "path_to_figures": path_to_current_experience + "figures/",
                               "path_to_feature_importance": path_to_current_experience + "feature_importance/",
                               "path_to_predictions": path_to_current_experience + "predictions/"}

            # Attributes and create folders
            for key in self.dict_paths:
                setattr(self, key, self.dict_paths[key])
                self.create_folder_if_doesnt_exist(self.dict_paths[key])

            # Update csv files
            for name in ["experiences", "metrics", "hyperparameters"]:
                self._update_experience_to_csv_file(name)
            self.save_config_json()

    def detect_variable(self) -> str:
        if "temperature" in self.config["global_architecture"]:
            return "T2m"
        else:
            return "UV"

    @staticmethod
    def list_physical_devices() -> None:
        gpus = tf.config.list_physical_devices('GPU')
        cpus = tf.config.list_physical_devices('CPU')
        print("\nPhysical devices available:")
        for device in cpus + gpus:
            print("Name:", device.name, "  Type:", device.device_type)

    @staticmethod
    def _get_current_date_str() -> str:
        today = date.today()
        return f"{today.year}_{today.month}_{today.day}"

    def _get_experiences_created_today(self) -> list:
        other_experiences_created_today = []

        for file in os.listdir(self.path_experiences):
            if self.current_date in file:
                other_experiences_created_today.append(file)

        return other_experiences_created_today

    def _get_current_id(self) -> str:
        if self.other_experiences_created_today:
            ids = [int(file.split("v")[1]) for file in self.other_experiences_created_today]
            current_id = np.max(ids).astype(np.int) + 1
        else:
            current_id = 0
        return current_id

    def _get_name_current_experience(self) -> str:
        return self.current_date + f"_{self.config['network']}_v{self.current_id}/"

    def _get_path_to_current_experience(self) -> str:
        return self.path_experiences + self.name_current_experience

    @staticmethod
    def create_folder_if_doesnt_exist(path: str,
                                      _raise: bool = True,
                                      verbose: bool = False
                                      ) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        elif _raise:
            raise FolderShouldNotExistError(path)
        else:
            if verbose:
                print(f"{path} already exists")
            pass

    def _update_experience_to_csv_file(self, name: str
                                       ) -> None:
        df = pd.read_csv(self.path_experiences + f"{name}.csv")

        keys = ["exp", "finished", "details"]
        values = [self.name_current_experience, self.is_finished, self.config["details"]]
        dict_to_append = {key: value for key, value in zip(keys, values)}
        df = df.append(dict_to_append, ignore_index=True)
        df.to_csv(self.path_experiences + f"{name}.csv", index=False)

    def _update_finished_csv_file(self) -> None:
        for name in ["experiences", "metrics", "hyperparameters"]:
            df = pd.read_csv(self.path_experiences + f"{name}.csv")
            filter_exp = df["exp"] == self.name_current_experience
            df.loc[filter_exp, "finished"] = 1
            df.to_csv(self.path_experiences + f"{name}.csv", index=False)
            print(f"Save info about experience in: " + self.path_experiences + f"{name}.csv")

    def _update_single_metrics_csv(self,
                                   metric_value: float,
                                   metric_name: str,
                                   precision: int = 3
                                   ) -> None:
        df = pd.read_csv(self.path_experiences + "metrics.csv")
        filter_exp = df["exp"] == self.name_current_experience
        df.loc[filter_exp, metric_name] = np.round(metric_value, precision)
        df.to_csv(self.path_experiences + "metrics.csv", index=False, float_format=f"%.{precision}f")
        print("Updated: " + self.path_experiences + "metrics.csv")

    def _update_metrics_csv(self,
                            list_metric_values: list,
                            metric_name: Union[str, None] = None,
                            precision: int = 3
                            ) -> None:
        for metric_value, model in zip(list_metric_values, ["_a", "_nn", "_int"]):
            self._update_single_metrics_csv(metric_value, metric_name + model, precision=precision)

    def _update_csv_files_with_results(self,
                                       c_eval: CustomEvaluation
                                       ) -> None:

        assert hasattr(c_eval, "df_results")

        mae = c_eval.df2mae()
        rmse = c_eval.df2rmse()
        bias = c_eval.df2mbe()

        self._update_metrics_csv(mae, metric_name="MAE")
        self._update_metrics_csv(rmse, metric_name="RMSE")
        self._update_metrics_csv(bias, metric_name="MB")

    def finished(self) -> None:
        self.is_finished = 1

    def save_model(self,
                   custom_model: CustomModel
                   ) -> None:
        tf.keras.models.save_model(custom_model.model, self.path_to_last_model)

    def save_config_json(self) -> None:
        with open(self.path_to_current_experience + 'config.json', 'w') as fp:
            json.dump(self.config, fp, sort_keys=True, indent=4)

    def save_norm_param(self,
                        mean: MutableSequence[float],
                        std: MutableSequence[float]
                        ) -> None:
        np.save(self.path_to_current_experience + "mean.npy", mean)
        np.save(self.path_to_current_experience + "std.npy", std)

    def save_experience_json(self) -> None:

        if hasattr(self, "is_finished"):
            if self.is_finished is None:
                self.is_finished = 0
        else:
            self.is_finished = 0

        dict_exp = self.__dict__

        # Remove dict, list... etc that are can not be (easily) transformed into a json file
        for key in ["dict_paths", "other_experiences_created_today", "config"]:
            if key in dict_exp:
                dict_exp.pop(key)

        # Numbers are converted to str
        for key in dict_exp:
            if not isinstance(dict_exp[key], str):
                dict_exp[key] = str(dict_exp[key])

        # Paths
        with open(self.path_to_current_experience + 'exp.json', 'w') as fp:
            json.dump(dict_exp, fp, sort_keys=True, indent=4)

    def save_all(self,
                 data: CustomDataHandler,
                 custom_model: CustomModel
                 ) -> None:

        self.save_model(custom_model)
        self.save_config_json()
        if self.config["standardize"]:
            self.save_norm_param(data.mean_standardize, data.std_standardize)
        self.save_experience_json()

    def save_results(self,
                     c_eval: CustomEvaluation
                     ) -> None:
        self.finished()
        self._update_finished_csv_file()
        self._update_csv_files_with_results(c_eval)

    @classmethod
    def from_previous_experience(cls, path_to_previous_exp):

        path_to_previous_exp = _get_full_path_to_previous_exp(path_to_previous_exp)

        # Load old config csv file
        with open(path_to_previous_exp + "/config.json", 'r') as fp:
            config = json.load(fp)

        # Detect current network (might be different from previous experience network)
        config["network"] = detect_network()

        # Update data paths if the network has change
        if config["network"] == "local":
            config = labia2local(config)

        inst = cls(config, override=False, restore_old_experience=True, create=False)

        # Load old experience csv file
        with open(path_to_previous_exp + "/exp.json", 'r') as fp:
            dict_exp = json.load(fp)

        # Update experience paths if the network has changed
        if config["network"] == "local":
            dict_exp = labia2local(dict_exp)

        # Add key of previous experience
        for key in dict_exp:
            setattr(inst, key, dict_exp[key])

        config["restore_experience"] = True

        inst.config = config

        return inst, config
