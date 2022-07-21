import numpy as np
import tensorflow as tf
import pytest

from bias_correction.train.model import Alpha2Direction


@pytest.fixture
def alpha2direction():
    return Alpha2Direction(unit_direction="degree",
                           unit_alpha="radian")


@pytest.fixture
def alpha():
    alpha = np.full((1, 10, 20, 1), np.pi/2, dtype=np.float32)
    return tf.convert_to_tensor(alpha)


@pytest.fixture
def direction():
    direction = np.full((1, 1), 270, dtype=np.float32)
    return tf.convert_to_tensor(direction)


@pytest.fixture()
def outputs():
    return alpha2direction(direction, alpha)


def test_shape_inputs_equals_shape_outputs():
    assert len(alpha.shape) == len(outputs.shape)


def test_result_consistent_with_numpy():
    outputs = outputs.numpy()
    expected_outputs = np.full((1, 10, 20, 1), 180)
    np.testing.assert_allclose(outputs, expected_outputs, atol=0.005)
