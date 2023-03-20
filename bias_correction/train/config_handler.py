from typing import Tuple, Dict, MutableSequence, Union
from dataclasses import dataclass, field


@dataclass
class DataFitDirection:
    batch_size: int
    epochs: int
    learning_rate: int
    loss: str = "cosine_distance"
    labels: MutableSequence[str] = field(default_factory=lambda: ['winddir(deg)'])
    type_of_output: str = "output_direction"
    remove_null_speeds: bool = True
    csv_logger: str = "CSVLogger_dir"
    # We don't fit a model with two outputs because it cause error in the loss function
    get_intermediate_output: bool = False
    current_variable: str = "UV_DIR"
    name: str = "fit_direction"


@dataclass
class DataPredictDirection:
    labels: MutableSequence[str] = field(default_factory=lambda: ['winddir(deg)'])
    type_of_output: str = "output_direction"
    remove_null_speeds: bool = True
    current_variable: str = "UV_DIR"
    name: str = "predict_direction"


@dataclass
class DataFitSpeed:
    loss: str
    batch_size: int
    epochs: int
    learning_rate: int
    labels: MutableSequence[str] = field(default_factory=lambda: ["vw10m(m/s)"])
    type_of_output: str = "output_speed"
    remove_null_speeds: bool = False
    csv_logger: str = "CSVLogger"
    # We don't fit a model with two outputs because it cause error in the loss function
    get_intermediate_output: bool = False
    current_variable: str = "UV"
    name: str = "fit_speed"


@dataclass
class DataPredictSpeed:
    labels: MutableSequence[str] = field(default_factory=lambda: ["vw10m(m/s)"])
    type_of_output: str = "output_speed"
    remove_null_speeds: bool = False
    current_variable: str = "UV"
    name: str = "fit_speed"


@dataclass
class DataPersist:
    get_intermediate_output: bool
    quick_test: bool
    quick_test_stations: MutableSequence[str]
    initial_loss: str


@dataclass
class DataFitSpeedDir:
    batch_size: int
    epochs: int
    learning_rate: int
    loss: str = "mixed"
    labels: MutableSequence[str] = field(default_factory=lambda: ["vw10m(m/s)", "winddir(deg)"])
    type_of_output: str = "output_speed_and_dir"
    remove_null_speeds: bool = False
    csv_logger: str = "CSVLogger"
    # We don't fit a model with two outputs because it cause error in the loss function
    get_intermediate_output: bool = False
    current_variable: str = "UV"
    name: str = "fit_speed_and_dir"


class PersistentConfig:

    def __init__(self, config: Dict) -> None:

        self.config = config

        self.data_persist = DataPersist(config["get_intermediate_output"],
                                        config["quick_test"],
                                        config["quick_test_stations"],
                                        config["loss"])

        self.data_fit_direction = DataFitDirection(config["batch_size_dir"],
                                                   config["epochs_dir"],
                                                   config["learning_rate_dir"])

        self.data_fit_speed = DataFitSpeed(config["loss"],
                                           config["batch_size_speed"],
                                           config["epochs_speed"],
                                           config["learning_rate_speed"])

        self.data_fit_speed_and_dir = DataFitSpeedDir(config["batch_size"],
                                                      config["epochs"],
                                                      config["learning_rate"])
        batch_size: int
        epochs: int
        learning_rate: int

        self.data_predict_speed = DataPredictSpeed()
        self.data_predict_direction = DataPredictDirection()

    def restore_persistent_data(self, keys: Tuple[str, ...], config: Dict) -> Dict:
        """Restore data from old config in new config"""
        for key in keys:
            config[key] = getattr(self.data_persist, key)
        return config

    @staticmethod
    def modify_config_for_fit(config: Dict,
                              fit_data: Union[DataFitDirection, DataFitSpeed]
                              ) -> Dict:
        """Adapts the configuration in order to fit model on direction or speed"""
        config["labels"] = fit_data.labels
        config["type_of_output"] = fit_data.type_of_output
        config["loss"] = fit_data.loss
        config["remove_null_speeds"] = fit_data.remove_null_speeds
        config["current_variable"] = fit_data.current_variable
        config["get_intermediate_output"] = fit_data.get_intermediate_output
        config["epochs"] = fit_data.epochs
        config["learning_rate"] = fit_data.learning_rate
        config["batch_size"] = fit_data.batch_size
        return config

    @staticmethod
    def modify_config_for_predict(config: Dict,
                                  predict_data: Union[DataPredictSpeed, DataPredictDirection]
                                  ) -> Dict:
        """Adapts the configuration in order to fit model on direction or speed"""
        config["labels"] = predict_data.labels
        config["type_of_output"] = predict_data.type_of_output
        config["remove_null_speeds"] = predict_data.remove_null_speeds
        config["current_variable"] = predict_data.current_variable
        return config

    @staticmethod
    def modify_csvlogger_in_callbacks(config: Dict,
                                      fit_data: Union[DataFitDirection, DataFitSpeed],
                                      not_fit_data: Union[DataFitDirection, DataFitSpeed]
                                      ) -> Dict:
        """Ensures speed CSVLogger is not in callbacks during fit on direction and vice-versa."""
        logger_not_wanted_in_callbacks = not_fit_data.csv_logger in config["callbacks"]
        if logger_not_wanted_in_callbacks:
            config["callbacks"].remove(not_fit_data.csv_logger)

        logger_wanted_in_data = fit_data.csv_logger in config["callbacks"]
        if not logger_wanted_in_data:
            config["callbacks"].append(fit_data.csv_logger)

        return config

    def config_fit_dir(self, config: Dict) -> Dict:
        """Adapts the configuration in order to fit model on direction"""
        print(self.data_fit_direction)
        config = self.modify_config_for_fit(config, self.data_fit_direction)
        return self.modify_csvlogger_in_callbacks(config, self.data_fit_direction, self.data_fit_speed)

    def config_fit_speed_and_dir(self, config: Dict) -> Dict:
        """Adapts the configuration in order to fit model on direction"""
        print(self.data_fit_speed_and_dir)
        config = self.modify_config_for_fit(config, self.data_fit_speed_and_dir)
        return self.modify_csvlogger_in_callbacks(config, self.data_fit_direction, self.data_fit_speed)

    def config_fit_speed(self, config: Dict) -> Dict:
        """Adapts the configuration in order to fit model on direction"""
        print(self.data_fit_speed)
        config = self.modify_config_for_fit(config, self.data_fit_speed)
        return self.modify_csvlogger_in_callbacks(config, self.data_fit_speed, self.data_fit_direction)

    def config_predict_dir(self, config: Dict) -> Dict:
        """Adapts the configuration in order to predict model on direction at the center of the map"""
        print(self.data_predict_direction)
        return self.modify_config_for_predict(config, self.data_predict_direction)

    def config_predict_speed(self, config: Dict) -> Dict:
        """Adapts the configuration in order to predict model on speed at the center of the map"""
        print(self.data_predict_speed)
        return self.modify_config_for_predict(config, self.data_predict_speed)

    def config_predict_parser(self,
                              name: str,
                              config: Dict) -> Dict:
        """Parse argument to detect which config needs to be created"""
        if "direction" in name or "dir" in name:
            return self.config_predict_dir(config)
        elif "speed" in name:
            return self.config_predict_speed(config)
        else:
            raise NotImplementedError("Predict on speed or direction only.")
