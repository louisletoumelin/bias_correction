import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, \
    load_model
from tensorflow.keras import backend as K

try:
    import horovod.tensorflow as hvd

    _horovod = True
except ModuleNotFoundError:
    _horovod = False

import os
from functools import partial
from typing import Callable, Union

from bias_correction.train.layers import RotationLayer, \
    CropTopography, \
    SelectCenter, \
    Normalization, \
    ActivationArctan, \
    Components2Speed, \
    Components2Direction, \
    SpeedDirection2Components, \
    Components2Alpha, \
    Alpha2Direction, \
    NormalizationInputs, \
    SimpleScaling
from bias_correction.train.optimizer import load_optimizer
from bias_correction.train.initializers import load_initializer
from bias_correction.train.loss import load_loss
from bias_correction.train.callbacks import load_callback_with_custom_model
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.unet import create_unet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class StrategyInitializer:
    _horovod = _horovod

    def __init__(self, config):
        self.config = config

        # Define later
        self.strategy = None
        self._horovod = None
        self.init_strategy()

    def init_strategy(self):
        if self.config["distribution_strategy"] == "Horovod" and self._horovod:
            self.init_horovod()
        else:
            self._horovod = False

        if self.config["distribution_strategy"] == "MirroredStrategy":
            self.strategy = self.init_mirrored_strategy()
            nb_replicas = self.strategy.num_replicas_in_sync
            # Adapt batch size according to the number of devices
            self.config["global_batch_size"] = self.config["batch_size"] * nb_replicas
        else:
            # Do not modify batch size
            self.config["global_batch_size"] = self.config["batch_size"]

        if self.config["distribution_strategy"] is None and self.config["network"] == "labia":
            print("\ntf.config.experimental.set_memory_growth called\n")
            physical_devices = tf.config.list_physical_devices('GPU')
            for gpu_instance in physical_devices:
                tf.config.experimental.set_memory_growth(gpu_instance, True)

    @staticmethod
    def init_mirrored_strategy():
        """http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-hvd-tf-multi.html"""
        # --- create the distribution strategy before calling any other tensorflow op
        cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver()
        implementation = tf.distribute.experimental.CommunicationImplementation.NCCL
        communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)
        strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver,
                                                             communication_options=communication_options)

        # get task id from cluster resolver
        task_info = cluster_resolver.get_task_info()
        task_id = task_info[1]
        print(f"task_info: {task_info}")
        print(f"task_id: {task_id}")
        # ---

        # --- get total number of GPUs
        n_workers = int(os.environ['SLURM_NTASKS'])  # get number of workers
        devices = tf.config.experimental.list_physical_devices('GPU')  # get list of devices visible per worker
        n_gpus_per_worker = len(devices)  # get number of devices per worker
        n_gpus = n_workers * n_gpus_per_worker  # get total number of GPUs
        print("ngpus: ", n_gpus)
        # ---
        return strategy

    def adapt_learning_rate_if_horovod(self):
        self.config["learning_rate"] = self.config["learning_rate"] * hvd.size()
        self.config["learning_rate_adapted"] = True

    def init_horovod(self):
        """https://github.com/horovod/horovod/blob/master/examples/keras/keras_mnist_advanced.py"""
        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        hvd.init()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        self.adapt_learning_rate_if_horovod()

        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"


class DevineBuilder(StrategyInitializer):

    def __init__(self,
                 config: dict
                 ) -> None:
        super().__init__(config)

        # Get norm
        self.mean_norm_cnn, self.std_norm_cnn = self.load_norm_unet(config["unet_path"])

    @staticmethod
    def load_classic_unet(model_path: str):
        def root_mse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_true - y_pred)))

        dependencies = {'root_mse': root_mse}

        return load_model(model_path,
                          custom_objects=dependencies,
                          options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))

    @staticmethod
    def load_norm_unet(model_path):
        """Load normalization parameters: mean and std"""

        dict_norm = pd.read_csv(model_path + "dict_norm.csv")
        mean = dict_norm["0"].iloc[0]
        std = dict_norm["0"].iloc[1]

        return mean, std

    @staticmethod
    def disable_training(model):
        model.trainable = False
        return model

    @staticmethod
    def load_custom_unet(input_shape, model_path):
        unet = create_unet(input_shape)
        unet.load_weights(model_path)
        return unet

    def devine(self,
               topos,
               x,
               inputs,
               use_crop=True):

        #  x[:, 0] is nwp wind speed.
        #  x[:, 1] is wind direction.
        y = RotationLayer(clockwise=False, unit_input="degree", fill_value=-9999)(topos, x[:, 1])

        if self.config["custom_unet"]:
            length_y = self.config["custom_input_shape"][0]
            length_x = self.config["custom_input_shape"][1]
            min_length = np.min([self.config["custom_input_shape"][0], self.config["custom_input_shape"][1]])
            y_diff = np.intp((min_length/np.sqrt(2))) // 2
            x_diff = np.intp((min_length/np.sqrt(2))) // 2
        else:
            length_y = 140
            length_x = 140
            y_diff = 79 // 2
            x_diff = 69 // 2

        if use_crop:
            print("debug")
            print(length_x)
            print(length_y)
            print(y_diff)
            print(x_diff)
            y = CropTopography(initial_length_x=length_x,
                               initial_length_y=length_y,
                               y_offset=y_diff,
                               x_offset=x_diff)(y)

        y = Normalization(self.mean_norm_cnn, self.std_norm_cnn)(y)

        if self.config["custom_unet"]:
            print("debug shape custom devine")
            print((y_diff*2+1, y_diff*2+1, 1))
            unet = self.load_custom_unet((y_diff*2+1, y_diff*2+1, 1), self.config["unet_path"])
        else:
            unet = self.load_classic_unet(self.config["unet_path"])

        if self.config["disable_training_cnn"]:
            unet = self.disable_training(unet)

        y = unet(y)

        if self.config["type_of_output"] == "map_u_v_w":
            w = y[:, :, :, 2]
            if len(w.shape) == 3:
                w = tf.expand_dims(w, -1)
            w = RotationLayer(clockwise=True,
                              unit_input="degree",
                              fill_value=np.nan)(w, x[:, 1])
            w = SimpleScaling()(w, x[:, 0])

        if self.config["type_of_output"] in ["output_components", "map", "map_components", "map_u_v_w"]:
            # Direction
            alpha_or_direction = Components2Alpha()(y)
            alpha_or_direction = Alpha2Direction("degree", "radian")(x[:, 1], alpha_or_direction)
            alpha_or_direction = RotationLayer(clockwise=True,
                                               unit_input="degree",
                                               fill_value=np.nan)(alpha_or_direction, x[:, 1])

        # Speed
        y = Components2Speed()(y)
        y = RotationLayer(clockwise=True, unit_input="degree", fill_value=np.nan)(y, x[:, 1])
        y = ActivationArctan(alpha=38.2)(y, x[:, 0])

        if self.config["type_of_output"] == "output_components":
            x, y = SpeedDirection2Components("degree")(y, alpha_or_direction)
            x = SelectCenter(79, 69)(x)
            y = SelectCenter(79, 69)(y)
            bc_model = Model(inputs=inputs, outputs=(x, y), name="bias_correction")

        elif self.config["type_of_output"] == "output_speed":
            y = SelectCenter(79, 69)(y)
            bc_model = Model(inputs=inputs, outputs=(y), name="bias_correction")

        elif self.config["type_of_output"] == "map_components":
            x, y = SpeedDirection2Components("degree")(y, alpha_or_direction)
            bc_model = Model(inputs=inputs, outputs=(x, y), name="bias_correction")

        elif self.config["type_of_output"] == "map_u_v_w":
            x, y = SpeedDirection2Components("degree")(y, alpha_or_direction)
            bc_model = Model(inputs=inputs, outputs=(x, y, w), name="bias_correction")

        elif self.config["type_of_output"] == "map":
            bc_model = Model(inputs=inputs, outputs=(y, alpha_or_direction), name="bias_correction")

        return bc_model


class CNNInput(StrategyInitializer):

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def tf_input_cnn(topos, kernel_size=(2, 2), filters=[32, 16, 8, 4], activation="relu", pool_size=(3, 3)):

        topos_norm = NormalizationInputs()(topos, 1258.0, 791.0)
        last_index = len(filters) - 1

        for idx, filter in enumerate(filters):

            conv_layer = Conv2D(filters=filter, kernel_size=kernel_size, activation=activation)

            if idx == 0:
                y = conv_layer(topos_norm)
            else:
                y = conv_layer(y)

            if idx < last_index:
                y = MaxPooling2D(pool_size=pool_size)(y)

        y = Flatten()(y)
        return y


class ArtificialNeuralNetwork(StrategyInitializer):

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def _dense_network(nwp_input,
                       nb_outputs,
                       nb_units=None,
                       activation_dense=None,
                       initializer=None,
                       batch_normalization=None,
                       dropout_rate=None,
                       use_bias=None):

        for index, nb_unit in enumerate(nb_units):

            dense_layer = Dense(nb_unit,
                                activation=activation_dense,
                                kernel_initializer=initializer,
                                name=f"D{index}",
                                use_bias=use_bias)

            # First layer using standardized inputs
            if index == 0:
                x = dense_layer(nwp_input)
            else:
                x = dense_layer(x)

            if batch_normalization:
                x = BatchNormalization()(x)

            if dropout_rate:
                x = Dropout(dropout_rate)(x)

        x = Dense(nb_outputs,
                  activation="linear",
                  kernel_initializer=initializer,
                  name=f"D_output",
                  use_bias=use_bias)(x)

        return x

    @staticmethod
    def _dense_network_with_skip_connections(nwp_input,
                                             nb_outputs,
                                             nb_units=None,
                                             activation_dense=None,
                                             initializer=None,
                                             batch_normalization=None,
                                             dropout_rate=None,
                                             use_bias=None):
        for index in range(nb_units):

            dense_unit = Dense(nb_units,
                               activation=activation_dense,
                               kernel_initializer=initializer,
                               name=f"D{index}",
                               use_bias=use_bias)

            # First layer using standardized inputs
            if index == 0:
                y = dense_unit(nwp_input) + nwp_input
            else:
                y = dense_unit(y) + y

            if batch_normalization:
                y = BatchNormalization()(y)

            if dropout_rate:
                y = Dropout(dropout_rate)(y)

        y = Dense(nb_outputs,
                  activation="linear",
                  kernel_initializer=initializer,
                  name=f"D_output",
                  use_bias=use_bias)(y)

        return y

    def get_func_dense_network(self, config: dict) -> Callable:
        kwargs_dense = {
            "nb_units": config["nb_units"],
            "activation_dense": config["activation_dense"],
            "initializer": config["initializer"],
            "batch_normalization": config["batch_normalization"],
            "dropout_rate": config["dropout_rate"],
            "use_bias": config["use_bias"]}

        if config["dense_with_skip_connection"]:

            if kwargs_dense["input_cnn"]:
                raise NotImplementedError("'input_cnn' option not implemented for dense network with skip connections")

            assert len(set(config["nb_units"])) == 1, "Skip connections requires units of the same size."

            return partial(self._dense_network_with_skip_connections, **kwargs_dense)
        else:
            return partial(self._dense_network, **kwargs_dense)


class CustomModel(StrategyInitializer):
    _horovod = _horovod

    def __init__(self, experience, config):
        super().__init__(config)

        if hasattr(self, "is_finished"):
            del self.is_finished

        # Get initializer
        self.initializer = self.get_initializer()
        self.exp = experience
        self.ann = ArtificialNeuralNetwork(config)
        self.cnn_input = CNNInput(config)
        self.devine_builder = DevineBuilder(config)

        # Defined later
        self.model = None
        self.model_version = None
        self.model_is_built = None
        self.model_is_compiled = None

    def get_optimizer(self):
        optimizer = load_optimizer(self.config["optimizer"],
                                   self.config["learning_rate"],
                                   *self.config["args_optimizer"],
                                   **self.config["kwargs_optimizer"])

        if self.config["distribution_strategy"] == "Horovod" and _horovod:
            return hvd.DistributedOptimizer(optimizer)
        else:
            return optimizer

    def get_loss(self):
        name_loss = self.config["loss"]
        return load_loss(name_loss,
                         *self.config["args_loss"][name_loss],
                         **self.config["kwargs_loss"][name_loss])

    def get_initializer(self):
        return load_initializer(self.config["initializer"],
                                *self.config["args_initializer"],
                                **self.config["kwargs_initializer"])

    def get_callbacks(self, data_loader=None, mode_callback=None):
        return load_callback_with_custom_model(self, data_loader=data_loader, mode_callback=mode_callback)

    def get_dense_network(self):
        return self.ann.get_func_dense_network(self.config)

    def cnn_and_concatenate(self, topos, inputs_nwp):
        if self.config["standardize"]:
            # inputs_cnn must be after cnn outputs to be sure speed and direction are latest in the list
            return tf.keras.layers.Concatenate(axis=1)([self.cnn_input.tf_input_cnn(topos), inputs_nwp])

    def add_intermediate_output(self):
        assert self.model_is_built
        inputs = self.model.input
        intermediate_outputs = self.model.get_layer("Add_dense_output").output
        self.model = Model(inputs=inputs, outputs=(self.model.outputs, intermediate_outputs))
        self.model.compile(loss=self.get_loss(), optimizer=self.get_optimizer())

    def _build_model_architecture(self,
                                  nb_input_variables: int,
                                  nb_outputs_dense_network: int,
                                  nb_var_for_skip_connection: int,
                                  use_input_cnn: bool,
                                  use_final_skip_connection: bool,
                                  use_devine: bool,
                                  use_standardize: bool,
                                  use_final_relu: bool,
                                  name: Union[str, None],
                                  input_shape_topo: tuple[int, int, int] = (140, 140, 1),
                                  ):

        # Inputs
        topos = Input(shape=input_shape_topo, name="input_topos")
        nwp_variables = Input(shape=(nb_input_variables,), name="input_nwp")

        # Standardize inputs
        if use_standardize:
            mean_norm = Input(shape=(nb_input_variables,), name="mean_norm")
            std_norm = Input(shape=(nb_input_variables,), name="std_norm")
            nwp_variables_norm = NormalizationInputs()(nwp_variables, mean_norm, std_norm)
            inputs = (topos, nwp_variables, mean_norm, std_norm)
        else:
            inputs = (topos, nwp_variables)

        # Input CNN
        if use_input_cnn and use_standardize:
            nwp_variables_norm = self.cnn_and_concatenate(topos, nwp_variables_norm)
        elif use_input_cnn and not use_standardize:
            nwp_variables = self.cnn_and_concatenate(topos, nwp_variables)

        # Dense network
        dense_network = self.get_dense_network()
        if use_standardize:
            x = dense_network(nwp_variables_norm, nb_outputs=nb_outputs_dense_network)
        else:
            x = dense_network(nwp_variables, nb_outputs=nb_outputs_dense_network)

        # Final skip connection
        if use_final_skip_connection:
            x = Add(name="Add_dense_output")([x, nwp_variables[:, -nb_var_for_skip_connection:]])

        if use_final_relu:
            x = tf.keras.activations.relu(x)

        if use_devine:
            bc_model = self.devine_builder.devine(topos, x, inputs)
        else:
            bc_model = Model(inputs=inputs, outputs=(x[:, 0]), name=name)

        print(bc_model.summary())
        self.model = bc_model

    def _build_dense_temperature(self):
        self._build_model_architecture(
            nb_input_variables=self.config["nb_input_variables"],
            nb_outputs_dense_network=1,
            nb_var_for_skip_connection=1,
            use_input_cnn=self.config["input_cnn"],
            use_final_skip_connection=self.config["final_skip_connection"],
            use_devine=False,
            use_standardize=self.config["standardize"],
            use_final_relu=False,
            name="bias_correction_temperature",
            input_shape_topo=(140, 140, 1),
        )

    def _build_dense_only(self):
        self._build_model_architecture(
            nb_input_variables=self.config["nb_input_variables"],
            nb_outputs_dense_network=2,
            nb_var_for_skip_connection=2,
            use_input_cnn=self.config["input_cnn"],
            use_final_skip_connection=self.config["final_skip_connection"],
            use_devine=False,
            use_standardize=self.config["standardize"],
            use_final_relu=True,
            name="bias_correction",
            input_shape_topo=(140, 140, 1),
        )

    def _build_devine_only(self):

        if self.config["custom_unet"]:
            input_shape = self.config["custom_input_shape"]
        else:
            input_shape = (140, 140, 1)

        # Inputs
        topos = Input(shape=input_shape, name="input_topos")
        x = Input(shape=(2,), name="input_wind_field")
        inputs = (topos, x)
        bc_model = self.devine_builder.devine(topos, x, inputs)

        print(bc_model.summary())
        self.model = bc_model

    def _build_ann_v0(self):
        self._build_model_architecture(
            nb_input_variables=self.config["nb_input_variables"],
            nb_outputs_dense_network=2,
            nb_var_for_skip_connection=2,
            use_input_cnn=self.config["input_cnn"],
            use_final_skip_connection=self.config["final_skip_connection"],
            use_devine=True,
            use_standardize=self.config["standardize"],
            use_final_relu=True,
            name=None,
            input_shape_topo=(140, 140, 1),
        )

    def _build_model(self):
        """Supported architectures: ann_v0, dense_only, dense_temperature, devine_only"""
        model_architecture = self.config["global_architecture"]
        methods_build = {"ann_v0": self._build_ann_v0,
                         "dense_only": self._build_dense_only,
                         "dense_temperature": self._build_dense_temperature,
                         "devine_only": self._build_devine_only}
        methods_build[model_architecture]()
        self.model_is_built = True

    def _build_compiled_model(self):
        self._build_model()
        self.model.compile(loss=self.get_loss(), optimizer=self.get_optimizer())
        self.model_is_compiled = True

    def _build_mirrored_strategy(self):

        # Get number of devices
        nb_replicas = self.strategy.num_replicas_in_sync
        print('\nMirroredStrategy: number of devices: {}'.format(nb_replicas))

        with self.strategy.scope():
            self._build_compiled_model()
            if self.config["get_intermediate_output"]:
                self.add_intermediate_output()

    def _build_classic_strategy(self):
        if self.config["distribution_strategy"] is None:
            print('\nNot distributed: number of devices: 1')

        self._build_compiled_model()

        if self.config["get_intermediate_output"]:
            self.add_intermediate_output()

    def build_model_with_strategy(self):
        if self.config["distribution_strategy"] == "MirroredStrategy":
            self._build_mirrored_strategy()
        else:
            self._build_classic_strategy()

    def select_model(self, force_build=False, model_version="last"):

        has_model_version = {"ann_v0": True,
                             "dense_only": True,
                             "dense_temperature": True,
                             "devine_only": False}

        if force_build:
            self.build_model_with_strategy()

        assert self.model_is_built

        model_architecture = self.config["global_architecture"]
        if has_model_version[model_architecture]:
            self.select_model_version(model_version, force_build)

    def select_model_version(self, model_version, build=False):

        if model_version == "last":
            # For previous experiences, last model is not built, for current experience, last model is built
            if build:
                self.model.load_weights(self.exp.path_to_last_model)
                self.model_version = "last"
                print("last model weights loaded")
            else:
                print("last model is already built and weights are loaded")

        elif "best":
            self.model.load_weights(self.exp.path_to_best_model)
            self.model_version = "best"
            print("best model weights loaded")

    def predict_with_batch(self, inputs_test, model_version="last", force_build=False):

        if model_version:
            self.select_model(force_build=force_build, model_version=model_version)

        for index, i in enumerate(inputs_test):
            results_test = self.model.predict(i)
            print("WARNING: multi batch prediction not supported")

        return results_test

    def fit_with_strategy(self, dataset, validation_data=None, dataloader=None, mode_callback=None):

        if not self.model_is_built and not self.model_is_compiled:
            self.build_model_with_strategy()

        results = self.model.fit(dataset,
                                 validation_data=validation_data,
                                 epochs=self.config["epochs"],
                                 callbacks=self.get_callbacks(dataloader, mode_callback))

        self._set_model_version_after_training()

        return results

    def _set_model_version_after_training(self):
        has_earlystopping = "EarlyStopping" in self.config["callbacks"]
        restore_best_weights = self.config["kwargs_callbacks"]["EarlyStopping"]["restore_best_weights"] is True
        if has_earlystopping and restore_best_weights:
            self.model_version = "best"
        else:
            self.model_version = "last"

    @classmethod
    def from_previous_experience(cls,
                                 exp: ExperienceManager,
                                 config: dict,
                                 model_str: str):

        inst = cls(exp, config)

        inst.select_model(model_version=model_str, force_build=True)

        return inst


"""
    def _build_dense_temperature(self):

        # Inputs
        topos = Input(shape=(140, 140, 1), name="input_topos")
        nwp_variables = Input(shape=(self.config["nb_input_variables"],), name="input_nwp")

        # Standardize inputs
        if self.config["standardize"]:
            mean_norm = Input(shape=(self.config["nb_input_variables"],), name="mean_norm")
            std_norm = Input(shape=(self.config["nb_input_variables"],), name="std_norm")
            nwp_variables_norm = NormalizationInputs()(nwp_variables, mean_norm, std_norm)
            inputs = (topos, nwp_variables, mean_norm, std_norm)
        else:
            inputs = (topos, nwp_variables)

        # Input CNN
        if self.config["input_cnn"] and self.config["standardize"]:
            nwp_variables_norm = self.cnn_and_concatenate(topos, nwp_variables_norm)
        elif self.config["input_cnn"] and not self.config["standardize"]:
            nwp_variables = self.cnn_and_concatenate(topos, nwp_variables)

        # Dense network
        dense_network = self.get_dense_network()
        if self.config["standardize"]:
            x = dense_network(nwp_variables_norm, nb_outputs=1)
        else:
            x = dense_network(nwp_variables, nb_outputs=1)

        # Final skip connection
        if self.config["final_skip_connection"]:
            x = Add(name="Add_dense_output")([x, nwp_variables[:, -1:]])

        bc_model = Model(inputs=inputs, outputs=(x), name="bias_correction_temperature")

        print(bc_model.summary())
        self.model = bc_model

    def _build_dense_only(self):

        # Inputs
        topos = Input(shape=(140, 140, 1), name="input_topos")
        nwp_variables = Input(shape=(self.config["nb_input_variables"],), name="input_nwp")

        # Standardize inputs
        if self.config["standardize"]:
            mean_norm = Input(shape=(self.config["nb_input_variables"],), name="mean_norm")
            std_norm = Input(shape=(self.config["nb_input_variables"],), name="std_norm")
            nwp_variables_norm = NormalizationInputs()(nwp_variables, mean_norm, std_norm)
            inputs = (topos, nwp_variables, mean_norm, std_norm)
        else:
            inputs = (topos, nwp_variables)

        # Input CNN
        if self.config["input_cnn"] and self.config["standardize"]:
            nwp_variables_norm = self.cnn_and_concatenate(topos, nwp_variables_norm)
        elif self.config["input_cnn"] and not self.config["standardize"]:
            nwp_variables = self.cnn_and_concatenate(topos, nwp_variables)

        # Dense network
        dense_network = self.get_dense_network()
        if self.config["standardize"]:
            x = dense_network(nwp_variables_norm)
        else:
            x = dense_network(nwp_variables)

        # Final skip connection
        if self.config["final_skip_connection"]:
            x = Add(name="Add_dense_output")([x, nwp_variables[:, -2:]])
            x = tf.keras.activations.relu(x)
        else:
            x = tf.keras.activations.relu(x)

        bc_model = Model(inputs=inputs, outputs=(x[:, 0]), name="bias_correction")

        print(bc_model.summary())
        self.model = bc_model

    def _build_devine_only(self):
        # Inputs
        topos = Input(shape=(140, 140, 1), name="input_topos")
        wind_field = Input(shape=(2,), name="input_wind_field")
        inputs = (topos, wind_field)

        bc_model = self.devine(topos, wind_field, inputs)

        print(bc_model.summary())
        self.model = bc_model

    def _build_ann_v0(self):

        # Inputs
        topos = Input(shape=(140, 140, 1), name="input_topos")
        nwp_variables = Input(shape=(self.config["nb_input_variables"],), name="input_nwp")

        # Standardize inputs
        if self.config["standardize"]:
            mean_norm = Input(shape=(self.config["nb_input_variables"],), name="mean_norm")
            std_norm = Input(shape=(self.config["nb_input_variables"],), name="std_norm")
            nwp_variables_norm = NormalizationInputs()(nwp_variables, mean_norm, std_norm)
            inputs = (topos, nwp_variables, mean_norm, std_norm)
        else:
            inputs = (topos, nwp_variables)

        # Input CNN
        if self.config["input_cnn"] and self.config["standardize"]:
            nwp_variables_norm = self.cnn_and_concatenate(topos, nwp_variables_norm)
        elif self.config["input_cnn"] and not self.config["standardize"]:
            nwp_variables = self.cnn_and_concatenate(topos, nwp_variables)

        # Dense network
        dense_network = self.get_dense_network()

        if self.config["standardize"]:
            x = dense_network(nwp_variables_norm)
        else:
            x = dense_network(nwp_variables)

        # Final skip connection
        if self.config["final_skip_connection"]:
            x = Add(name="Add_dense_output")([x, nwp_variables[:, -2:]])
            x = tf.keras.activations.relu(x)
        else:
            x = tf.keras.activations.relu(x)

        bc_model = self.devine(topos, x, inputs)

        print(bc_model.summary())
        self.model = bc_model
        """
