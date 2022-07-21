import numpy as np
import tensorflow as tf
import pytest

from bias_correction.train.model import Components2Direction


@pytest.fixture
def components2direction():
    return Components2Direction()


@pytest.fixture
def inputs():
    inputs = np.ones((1, 10, 20, 3))
    inputs[:, :, :, 0] = 1
    inputs[:, :, :, 1] = 0
    return tf.convert_to_tensor(inputs)


@pytest.fixture()
def outputs():
    return components2direction(inputs)


def test_shape_inputs_equals_shape_outputs():
    assert inputs.shape == outputs.shape


def test_result_consistent_with_numpy():
    outputs = outputs.numpy()
    inputs = inputs.numpy()
    expected_result = np.mod(180+np.rad2deg(np.arctan2(inputs[:, :, :, 0], inputs[:, :, :, 1])), 360)
    np.testing.assert_allclose(outputs, expected_result, atol=0.005)


def test_result():
    outputs = outputs.numpy()
    expected_result = np.full_like(outputs, 270)
    np.testing.assert_allclose(outputs, expected_result, atol=0.005)





