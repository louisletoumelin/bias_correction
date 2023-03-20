import numpy as np
import matplotlib.pyplot as plt
import pytest
import tensorflow as tf

import random

from bias_correction.train.model import RotationLayer
from bias_correction.tests.fixtures import image_cross


@pytest.fixture
def rotation_layer():
    return RotationLayer(False, "degree")


def test_type_RotationLayer(rotation_layer):
    assert isinstance(rotation_layer, RotationLayer)


def figure_rotationlayer(rotation_layer):
    for wind_direction in [0, 90, 180, 270, 360, 45]:

        plt.figure()
        plt.subplot(121)
        plt.imshow(image_cross[0, :, :, 0])
        plt.title("Wind direction = 0")
        plt.subplot(122)
        result = rotation_layer(image_cross, np.array([wind_direction]))[0, :, :, 0]
        plt.imshow(result)
        plt.title(f"Wind direction = {wind_direction}")
        plt.colorbar()


def test_shape_input_equals_shape_output(rotation_layer):
    inputs = tf.convert_to_tensor(image_cross)
    outputs = rotation_layer(inputs, random.randint(0, 360))

    assert inputs.shape == outputs.shape





