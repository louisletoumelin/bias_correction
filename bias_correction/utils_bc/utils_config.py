import numpy as np


def assert_input_for_skip_connection(config):
    if config["global_architecture"] != "temperature":
        if config["final_skip_connection"]:
            for variable in ["Wind", "Wind_DIR"]:
                if variable not in config["wind_nwp_variables"]:
                    config["wind_nwp_variables"].append(variable)

            assert "Wind" in config["wind_nwp_variables"]
            assert "Wind_DIR" in config["wind_nwp_variables"]
    else:
        if config["final_skip_connection"]:
            for variable in ["Tair"]:
                if variable not in config["wind_temp_variables"]:
                    config["wind_temp_variables"].append(variable)

            assert "Tair" in config["wind_nwp_variables"]
    return config


def sort_input_variables(config):
    if "temperature" not in config["global_architecture"]:
        for variable in config["wind_nwp_variables"]:
            if variable in config["input_variables"]:
                config["input_variables"].remove(variable)
            config["input_variables"].append(variable)

        assert config["input_variables"][-2] == config["wind_nwp_variables"][-2]
        assert config["input_variables"][-1] == config["wind_nwp_variables"][-1]
    else:
        for variable in config["wind_temp_variables"]:
            if variable in config["input_variables"]:
                config["input_variables"].remove(variable)
            config["input_variables"].append(variable)

        assert config["input_variables"][-1] == config["wind_temp_variables"][-1]
    return config


def adapt_distribution_strategy_to_available_devices(config):
    import tensorflow as tf

    no_gpu_available = not tf.config.list_physical_devices('GPU')
    gpu_distribution_strategy_is_specified = config["distribution_strategy"] is not None

    if gpu_distribution_strategy_is_specified and no_gpu_available:
        config["distribution_strategy"] = None

    return config


def init_learning_rate_adapted(config):
    config["learning_rate_adapted"] = False
    return config


def detect_variable(config):
    if "temperature" in config["global_architecture"]:
        config["current_variable"] = "T2m"
    else:
        if config["type_of_output"] != "output_direction":
            config["current_variable"] = "UV"
        else:
            config["current_variable"] = "UV_DIR"
    return config


def define_input_variables(config):
    config["input_variables"] = np.unique(list(config["input_speed"]) + list(config["input_dir"])).tolist()
    return config


def get_idx_speed_and_dir_variables(config):
    input_var = list(config["input_variables"])
    config["idx_speed_var"] = np.sort([input_var.index(i) for i in config["input_speed"]])
    config["idx_dir_var"] = np.sort([input_var.index(i) for i in config["input_dir"]])
    return config
