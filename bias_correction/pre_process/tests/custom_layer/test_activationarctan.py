import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pytest

from bias_correction.train.model import ActivationArctan


@pytest.fixture
def inputs():
    inputs = np.ones((1, 79, 69, 3))
    return tf.convert_to_tensor(inputs)


@pytest.fixture
def outputs(inputs):
    return ActivationArctan(38.5)(inputs, np.array([10.], dtype=np.float32))


def test_shape_inputs_equals_shape_outputs(inputs, outputs):
    assert inputs.shape == outputs.shape


def test_result(inputs, outputs):
    expected_result = 38.2 * np.arctan(3.33 / 38.2)
    np.testing.assert_allclose(np.full_like(inputs, expected_result), outputs.numpy(), atol=0.005)
