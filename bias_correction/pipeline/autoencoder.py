import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
import matplotlib.pyplot as plt

import sys

sys.path.append("/home/letoumelinl/bias_correction/src/")
sys.path.append("//home/mrmn/letoumelinl/bias_correction/src/")
from bias_correction.config._config import config

use_custom_generator = False
batch_size = 2 ** 10
launch_training = False
n = 2


class RandomTopoGenerator:

    def __init__(self,
                 path=config['path_root'] + "Data/1_Raw//DEM/DEM_FRANCE_L93_30m_bilinear.nc",
                 min_y=17_000,
                 max_y=26_000,
                 min_x=26_000,
                 max_x=32_000,
                 nb_topos=100
                 ) -> None:
        self.topos = np.expand_dims(xr.open_dataset(path).isel(band=0).__xarray_dataarray_variable__.values, axis=-1)
        print("debug")
        print(self.topos.shape)
        self.min_y = min_y
        self.max_y = max_y
        self.min_x = min_x
        self.max_x = max_x
        self.nb_topos = nb_topos

    def __call__(self):
        for i in range(self.nb_topos):
            x = np.random.randint(self.min_x, self.max_x - 41)
            y = np.random.randint(self.min_y, self.max_y - 41)
            yield self.topos[y:y + 41, x:x + 41, :]


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 path=config['path_root'] + "Data/1_Raw//DEM/DEM_FRANCE_L93_30m_bilinear.nc",
                 min_y=17_000,
                 max_y=26_000,
                 min_x=26_000,
                 max_x=32_000,
                 nb_topos=10_000,
                 batch_size=1000,
                 ) -> None:
        try:
            self.topos = np.expand_dims(xr.open_dataset(path).isel(band=0).__xarray_dataarray_variable__.values,
                                        axis=-1)
        except (ValueError, AttributeError):
            self.topos = np.expand_dims(xr.open_dataset(path).alti.values, axis=-1)

        print("debug")
        print(self.topos.shape)
        self.min_y = min_y
        self.max_y = max_y
        self.min_x = min_x
        self.max_x = max_x
        self.nb_topos = nb_topos
        self.batch_size = batch_size

    def __len__(self):
        return np.intp(self.nb_topos // self.batch_size)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 41, 41, 1))

        for i in range(self.batch_size):
            x = np.random.randint(self.min_x, self.max_x - 41)
            y = np.random.randint(self.min_y, self.max_y - 41)
            X[i, :, :, :] = self.topos[y:y + 41, x:x + 41, :]

        return X, X


input_img = Input(shape=(41, 41, 1))

x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(input_img)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)

x = layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same', name="encoded")(x)

x = layers.UpSampling2D((2, 2))(encoded)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.Cropping2D(cropping=((0, 1), (0, 1)))(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Cropping2D(cropping=((0, 1), (0, 1)))(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Cropping2D(cropping=((0, 1), (0, 1)))(x)
x = layers.Conv2D(8 * n, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Cropping2D(cropping=((0, 1), (0, 1)))(x)
decoded = layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

if use_custom_generator:
    topos_generator = RandomTopoGenerator(nb_topos=1_000)
    x_train = tf.data.Dataset.zip((tf.data.Dataset.from_generator(topos_generator,
                                                                  output_types=tf.float32,
                                                                  output_shapes=(41, 41, 1)),
                                   tf.data.Dataset.from_generator(topos_generator,
                                                                  output_types=tf.float32,
                                                                  output_shapes=(41, 41, 1))))
else:
    x_train = DataGenerator(nb_topos=batch_size * 100,
                            batch_size=batch_size)
    x_val = DataGenerator(path=config['path_root'] + "Data/1_Raw/DEM/DEM_ALPES_L93_30m.nc",
                          min_y=12_200,
                          max_y=13_200,
                          min_x=6_600,
                          max_x=7_600,
                          nb_topos=1000,
                          batch_size=1000)
if launch_training:
    if use_custom_generator:
        autoencoder.fit(x_train,
                        epochs=25,
                        batch_size=1)
    else:
        autoencoder.fit(x_train,
                        epochs=25,
                        validation_data=x_val)

    tf.keras.models.save_model(autoencoder, config["path_root"] + "Data/3_Predictions/autoencoder/")
    print(config["path_root"] + "Data/3_Predictions/autoencoder/")

if not launch_training:
    x_test = DataGenerator(nb_topos=1,
                           batch_size=1)
    model = tf.keras.models.load_model(config["path_root"] + "Data/3_Predictions/autoencoder/")
    x_test_0 = x_test.__getitem__(1)[0]

    plt.figure()
    plt.imshow(x_test_0[0, :, :, 0])
    plt.colorbar()
    plt.title("Ground truth")
    x_pred_0 = model.predict(x_test_0)

    plt.figure()
    plt.imshow(x_pred_0[0, :, :, 0])
    plt.colorbar()
    plt.title("Prediction")

    plt.figure()
    plt.imshow(x_test_0[0, :, :, 0] - x_pred_0[0, :, :, 0], cmap="coolwarm")
    plt.colorbar()
    plt.title("Difference")

    encoder = Model(model.input, model.get_layer("encoded").output)
    conv_id = 1
    layer_conv = Model(model.input, model.layers[conv_id].output)

    for i in range(layer_conv.predict(x_test_0).shape[-1]):
        plt.figure()
        plt.imshow(layer_conv.predict(x_test_0)[0, 2:-2, 2:-2, i])
        plt.colorbar()
        plt.title(f"Convolution {conv_id} id = {i}")
