import numpy as np
import tensorflow as tf
import pytest

from bias_correction.train.model import Components2Speed


@pytest.fixture
def components2speed():
    return Components2Speed()


@pytest.fixture
def inputs():
    inputs = np.ones((1, 10, 20, 3))
    inputs[:, :, :, 1] = 3
    return tf.convert_to_tensor(inputs)


@pytest.fixture()
def outputs():
    return components2speed(inputs)


def test_shape_inputs_equals_shape_outputs():
    assert inputs.shape == outputs.shape


def test_result():
    expected_result = np.expand_dims(np.sqrt(inputs[:, :, :, 0] ** 2 + inputs[:, :, :, 1] ** 2), axis=-1)
    np.testing.assert_allclose(outputs, expected_result, atol=0.005)

