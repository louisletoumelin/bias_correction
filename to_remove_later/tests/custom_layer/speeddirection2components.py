import numpy as np
import tensorflow as tf
import pytest

from bias_correction.train.model import SpeedDirection2Components


@pytest.fixture
def speeddirection2components():
    return SpeedDirection2Components(unit_input="degree")


@pytest.fixture
def speed():
    inputs = np.ones((1, 10, 20, 1))
    return tf.convert_to_tensor(inputs)


@pytest.fixture
def direction():
    inputs = np.full((1, 10, 20, 1), 270)
    return tf.convert_to_tensor(inputs)


@pytest.fixture()
def outputs():
    return speeddirection2components(speed, direction)


def test_shape_inputs_equals_shape_outputs():
    assert speed.shape == outputs[0].shape
    assert speed.shape == outputs[1].shape


def test_result_consistent_with_numpy():
    outputs = outputs.numpy()
    inputs = inputs.numpy()
    expected_U = np.ones(inputs.shape)
    expected_V = np.zeros(inputs.shape) 
    np.testing.assert_allclose(outputs, expected_U, atol=0.005)
    np.testing.assert_allclose(outputs, expected_V, atol=0.005)





