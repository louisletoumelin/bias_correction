import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, \
    load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    NormalizationInputs
from bias_correction.train.optimizer import load_optimizer
from bias_correction.train.initializers import load_initializer
from bias_correction.train.loss import load_loss
from bias_correction.train.callbacks import FeatureImportanceCallback

try:
    import horovod.tensorflow as hvd

    _horovod = True
except ModuleNotFoundError:
    _horovod = False


class Initializer:
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


class CNNDevine(Initializer):

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def load_cnn(model_path):
        def root_mse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_true - y_pred)))

        dependencies = {'root_mse': root_mse}

        return load_model(model_path,
                          custom_objects=dependencies,
                          options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))

    @staticmethod
    def load_norm_cnn(model_path):
        """Load normalization parameters: mean and std"""

        model_path = model_path
        dict_norm = pd.read_csv(model_path + "dict_norm.csv")
        mean = dict_norm["0"].iloc[0]
        std = dict_norm["0"].iloc[1]

        return mean, std

    @staticmethod
    def disable_training(model):
        model.trainable = False
        return model

    def devine(self, topos, x, inputs):
        #  x[:, 0] is nwp wind speed.
        #  x[:, 1] is wind direction.
        y = RotationLayer(clockwise=False, unit_input="degree", fill_value=-1)(topos, x[:, 1])
        y = CropTopography(initial_length=140, y_offset=79 // 2, x_offset=69 // 2)(y)
        y = Normalization(self.mean_norm_cnn, self.std_norm_cnn)(y)
        y = self.cnn_devine(y)

        if self.config["type_of_output"] in ["output_components", "map", "map_components"]:
            # Direction
            alpha_or_direction = Components2Alpha()(y)
            alpha_or_direction = Alpha2Direction("degree", "radian")(x[:, 1], alpha_or_direction)
            alpha_or_direction = RotationLayer(clockwise=True,
                                               unit_input="degree",
                                               fill_value=-1)(alpha_or_direction, x[:, 1])

        # Speed
        y = Components2Speed()(y)
        y = RotationLayer(clockwise=True, unit_input="degree", fill_value=-1)(y, x[:, 1])
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

        elif self.config["type_of_output"] == "map":
            bc_model = Model(inputs=inputs, outputs=(y, alpha_or_direction), name="bias_correction")

        return bc_model


class CustomModel(CNNDevine):
    _horovod = _horovod

    def __init__(self, experience, config):
        super().__init__(config)
        self.__dict__ = experience.__dict__

        if hasattr(self, "is_finished"):
            del self.is_finished

        # Get initializer
        self.initializer = self.get_initializer()

        self.exp = experience
        self.model_is_built = None
        self.model_is_compiled = None

        # Load cnn
        self.cnn_devine = self.load_cnn(config["model_path"])

        # Freeze CNN weights
        if config["disable_training_cnn"]:
            self.cnn_devine = self.disable_training(self.cnn_devine)

        # Get norm
        self.mean_norm_cnn, self.std_norm_cnn = self.load_norm_cnn(config["model_path"])

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
        loss_func = load_loss(self.config["loss"],
                              *self.config["args_loss"],
                              **self.config["kwargs_loss"])

        return loss_func

    def get_initializer(self):
        return load_initializer(self.config["initializer"],
                                *self.config["args_initializer"],
                                **self.config["kwargs_initializer"])

    def get_callbacks(self, data_loader=None, mode_callback=None):
        callbacks = []

        if self.config["distribution_strategy"] == "Horovod" and _horovod:
            callbacks = self._append_hvd_callbacks(callbacks)

            if hvd.rank() == 0:
                callbacks = self._append_regular_callbacks(callbacks)
        else:
            callbacks = self._append_regular_callbacks(callbacks, data_loader=data_loader, mode_callback=mode_callback)

        return callbacks

    def get_prefetch(self):
        if self.config["prefetch"] == "auto":
            return tf.data.AUTOTUNE
        else:
            return self.config["prefetch"]

    @staticmethod
    def _append_hvd_callbacks(callbacks):

        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        callbacks.append(hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0))

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        callbacks.append(hvd.keras.callbacks.MetricAverageCallback())

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        # str_callback = "HVDLearningRateWarmupCallback"
        # callbacks.append(hvd.keras.callbacks.LearningRateWarmupCallback(initial_lr=self.config["learning_rate"],
        #                                                                **self.config["kwargs_callbacks"][str_callback]))

        return callbacks

    def _append_regular_callbacks(self, callbacks, data_loader=None, mode_callback=None):

        if "TensorBoard" in self.config["callbacks"]:
            callbacks.append(TensorBoard(log_dir=self.path_to_tensorboard_logs,
                                         **self.config["kwargs_callbacks"]["TensorBoard"]))

        if "ReduceLROnPlateau" in self.config["callbacks"]:
            callbacks.append(ReduceLROnPlateau(**self.config["kwargs_callbacks"]["ReduceLROnPlateau"]))

        if "EarlyStopping" in self.config["callbacks"]:
            callbacks.append(EarlyStopping(**self.config["kwargs_callbacks"]["EarlyStopping"]))

        if "CSVLogger" in self.config["callbacks"]:
            callbacks.append(CSVLogger(self.path_to_logs + "tf_logs.csv"))

        if "ModelCheckpoint" in self.config["callbacks"]:
            callbacks.append(ModelCheckpoint(filepath=self.path_to_best_model,
                                             **self.config["kwargs_callbacks"]["ModelCheckpoint"]))

        if "FeatureImportanceCallback" in self.config["callbacks"]:
            callbacks.append(FeatureImportanceCallback(data_loader, self, self.exp, mode_callback))

        return callbacks

    @staticmethod
    def _input_cnn(topos):
        topos_norm = NormalizationInputs()(topos, 1258.0, 791.0)
        y = Conv2D(filters=32,
                   kernel_size=(2, 2),
                   activation='relu')(topos_norm)
        y = MaxPooling2D((3, 3))(y)
        y = Conv2D(filters=16,
                   kernel_size=(2, 2),
                   activation='relu')(y)
        y = MaxPooling2D((3, 3))(y)
        y = Conv2D(filters=8,
                   kernel_size=(2, 2),
                   activation='relu')(y)
        y = MaxPooling2D((3, 3))(y)
        y = Conv2D(filters=4,
                   kernel_size=(2, 2),
                   activation='relu')(y)

        y = Flatten()(y)
        return y

    def dense_network(self, nwp_input, nb_outputs=2):
        for index, nb_unit in enumerate(self.config["nb_units"]):

            # First layer using standardized inputs
            if index == 0:
                x = Dense(nb_unit, activation=self.config["activation_dense"], kernel_initializer=self.initializer,
                          name=f"D{index}", use_bias=self.config["use_bias"])(nwp_input)

            else:
                x = Dense(nb_unit, activation=self.config["activation_dense"], kernel_initializer=self.initializer,
                          name=f"D{index}", use_bias=self.config["use_bias"])(x)

            if self.config["batch_normalization"]:
                x = BatchNormalization()(x)

            if self.config["dropout_rate"]:
                x = Dropout(self.config["dropout_rate"])(x)

        x = Dense(nb_outputs, activation="linear", kernel_initializer=self.initializer, name=f"D_output",
                  use_bias=self.config["use_bias"])(x)

        return x

    def dense_network_with_skip_connections(self, nwp_input, nb_outputs=2):

        if self.config["input_cnn"]:
            raise NotImplementedError("'input_cnn' option not implemented for dense network with skip connections")

        for index in range(self.config["nb_layers_skip_connection"]):

            # First layer using standardized inputs
            if index == 0:
                x = Dense(self.config["nb_input_variables"], activation=self.config["activation_dense"],
                          kernel_initializer=self.initializer,
                          name=f"D{index}")(nwp_input)
                x = x + nwp_input
                if self.config["batch_normalization"]:
                    x = BatchNormalization()(x)

                if self.config["dropout_rate"]:
                    x = Dropout(self.config["dropout_rate"])(x)

            else:
                y = Dense(self.config["nb_input_variables"], activation=self.config["activation_dense"],
                          kernel_initializer=self.initializer,
                          name=f"D{index}")(x)
                y = y + x

                if self.config["batch_normalization"]:
                    y = BatchNormalization()(y)

                if self.config["dropout_rate"]:
                    y = Dropout(self.config["dropout_rate"])(y)

        y = Dense(nb_outputs, activation="linear", kernel_initializer=self.initializer, name=f"D_output")(y)

        return y

    def select_dense_network(self):
        if self.config["dense_with_skip_connection"]:
            return self.dense_network_with_skip_connections
        else:
            return self.dense_network

    def cnn_and_concatenate(self, topos, inputs_nwp):
        y = self._input_cnn(topos)

        if self.config["standardize"]:
            # inputs_cnn must be after y to be sure speed and direction are latest in the list
            return tf.keras.layers.Concatenate(axis=1)([y, inputs_nwp])

    def add_intermediate_output(self):
        XX = self.model.input
        YY = self.model.get_layer("Add_dense_output").output
        self.model = Model(inputs=XX, outputs=(self.model.outputs, YY))

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
        dense_network = self.select_dense_network()

        if self.config["standardize"]:
            x = dense_network(nwp_variables_norm, nb_outputs=1)
        else:
            x = dense_network(nwp_variables, nb_outputs=1)

        # Skip connection
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
        dense_network = self.select_dense_network()

        if self.config["standardize"]:
            x = dense_network(nwp_variables_norm)
        else:
            x = dense_network(nwp_variables)

        # Skip connection
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
        dense_network = self.select_dense_network()

        if self.config["standardize"]:
            x = dense_network(nwp_variables_norm)
        else:
            x = dense_network(nwp_variables)

        # Skip connection
        if self.config["final_skip_connection"]:
            x = Add(name="Add_dense_output")([x, nwp_variables[:, -2:]])
            x = tf.keras.activations.relu(x)
        else:
            x = tf.keras.activations.relu(x)

        bc_model = self.devine(topos, x, inputs)

        print(bc_model.summary())
        self.model = bc_model

    def build_model(self):
        """Supported architectures: ann_v0, dense_only, dense_temperature, devine_only"""
        model_architecture = self.config["global_architecture"]
        method_build = getattr(self, f"_build_{model_architecture}")
        method_build()
        self.model_is_built = True

    def build_compiled_model(self):
        self.build_model()
        self.model.compile(loss=self.get_loss(), optimizer=self.get_optimizer())
        self.model_is_built = True
        self.model_is_compiled = True

    def _build_mirrored_strategy(self):

        # Get number of devices
        nb_replicas = self.strategy.num_replicas_in_sync
        print('\nMirroredStrategy: number of devices: {}'.format(nb_replicas))

        # Adapt batch size according to the number of devices
        self.config["global_batch_size"] = self.config["batch_size"] * nb_replicas

        with self.strategy.scope():
            self.build_compiled_model()
            # todo change here
            if self.config["get_intermediate_output"]:
                self.add_intermediate_output()
                self.model.compile(loss=self.get_loss(), optimizer=self.get_optimizer())

    def _build_classic_strategy(self):
        if self.config["distribution_strategy"] is None:
            print('\nNot distributed: number of devices: 1')

        # Do not modify batch size
        self.config["global_batch_size"] = self.config["batch_size"]

        self.build_compiled_model()
        if self.config["get_intermediate_output"]:
            self.add_intermediate_output()
            self.model.compile(loss=self.get_loss(), optimizer=self.get_optimizer())

    def build_model_with_strategy(self):
        if self.config["distribution_strategy"] == "MirroredStrategy":
            self._build_mirrored_strategy()
        else:
            self._build_classic_strategy()

    def select_model_version(self, model_str, build=False):

        if build:
            self.build_model_with_strategy()

        if model_str == "last":
            if build:
                self.model.load_weights(self.path_to_last_model)
            pass

        elif "best":
            self.model.load_weights(self.path_to_best_model)

    def predict_with_batch(self, inputs_test, model_str="last"):

        if model_str:
            self.select_model_version(model_str)

        if not self.model_is_built:
            self.build_model()

        for index, i in enumerate(inputs_test):
            results_test = self.model.predict(i)

        return results_test

    def prepare_train_dataset(self, dataset):
        dataset = dataset.batch(batch_size=self.config["global_batch_size"]) \
            .cache() \
            .prefetch(self.get_prefetch())
        return dataset

    def prepare_val_dataset(self, dataset):
        dataset = dataset.batch(batch_size=self.config["global_batch_size"])
        return dataset

    def fit_with_strategy(self, dataset, validation_data=None, dataloader=None, mode_callback=None):

        if not self.model_is_built and not self.model_is_compiled:
            self.build_model_with_strategy()

        dataset = self.prepare_train_dataset(dataset)

        validation_data = self.prepare_val_dataset(validation_data)

        results = self.model.fit(dataset,
                                 validation_data=validation_data,
                                 epochs=self.config["epochs"],
                                 callbacks=self.get_callbacks(dataloader, mode_callback))

        return results

    @classmethod
    def from_previous_experience(cls, exp, config, model_str):

        inst = cls(exp, config)

        inst.select_model_version(model_str, build=True)

        return inst
