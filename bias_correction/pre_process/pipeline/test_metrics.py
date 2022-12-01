import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf

path = "/home/letoumelinl/himalaya/"
filename = "large_dem.nc"

test = xr.open_dataset(path+filename).isel(band=0).band_data.values

test = test[1839:2243, 2783:3129]


def terrain_slope_azimuth_map(mnt, dx, verbose=True):
    """
    3 pi /2 - tan-1[ (dz/dy) / (dz/dx) ]
    """
    gradient_y, gradient_x = np.gradient(mnt, dx)
    arctan_value = np.where(gradient_x != 0,
                            np.arctan(gradient_y / gradient_x),
                            np.where(gradient_y > 0,
                                     np.pi / 2,
                                     np.where(gradient_y < 0,
                                              gradient_y,
                                              -np.pi / 2)))

    print("____Library: numpy") if verbose else None

    return 3 * np.pi / 2 - arctan_value


def mu_average_tensorflow(mu: np.ndarray,
                          x_win: float = 69//2,
                          y_win: float = 79//2
                          ) -> np.ndarray:
    """Compute mean slope with tensorflow."""

    # reshape for tensorflow
    mu = mu.reshape((1, mu.shape[0], mu.shape[1], 1)).astype(np.float32)

    # filter
    x_length = x_win * 2 + 1
    y_length = y_win * 2 + 1
    filter_mean = np.ones((1, y_length, x_length, 1), dtype=np.float32) / (x_length * y_length)
    filter_mean = filter_mean.reshape((y_length, x_length, 1, 1))

    # convolution
    return tf.nn.convolution(mu, filter_mean, strides=[1, 1, 1, 1], padding="SAME").numpy()[0, :, :, 0]


def plot_fig(data, cmap="viridis"):
    plt.figure()
    plt.imshow(data, cmap=cmap)
    plt.colorbar()


test_avg_small = mu_average_tensorflow(test, 3, 3)
test_avg_large = mu_average_tensorflow(test, 10, 10)

diff = test_avg_small - test_avg_large
