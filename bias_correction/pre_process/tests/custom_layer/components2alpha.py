import numpy as np
import tensorflow as tf
import pytest

from bias_correction.train.model import Components2Alpha


@pytest.fixture
def components2alpha():
    return Components2Alpha(unit_input="degree")


@pytest.fixture
def inputs():
    inputs = np.ones((1, 10, 20, 3))
    return tf.convert_to_tensor(inputs)


@pytest.fixture
def inputs1():
    inputs1 = np.ones((1, 10, 20, 3))
    inputs1[:, :, :, 0] = 0
    return tf.convert_to_tensor(inputs1)


@pytest.fixture()
def outputs():
    return components2alpha(inputs)


@pytest.fixture()
def outputs1():
    return components2alpha(inputs1)


def test_shape_inputs_equals_shape_outputs():
    assert len(inputs.shape) == len(outputs.shape)


def test_result_consistent_with_numpy():
    outputs = outputs.numpy()
    expected_outputs = np.full((1, 10, 20, 1), np.pi/4)
    np.testing.assert_allclose(outputs, expected_outputs, atol=0.005)

    expected_outputs = np.full((1, 10, 20, 1), np.pi/2)
    np.testing.assert_allclose(outputs, expected_outputs, atol=0.005)
