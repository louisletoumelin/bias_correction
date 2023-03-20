import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pytest




@pytest.fixture
def output_normalizationcnn(image_cross):
    return normalizationcnn(image_cross)


def figure_normalizationcnn(image_cross, croptopography):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image_cross[0, :, :, 0])
    plt.title("Before normalization")
    plt.colorbar()

    plt.subplot(122)
    result = output_normalizationcnn[0, :, :, 0]
    plt.imshow(result)
    plt.title(f"After normalization")
    plt.colorbar()


def test_output_is_tensor():
    assert "tensorflow" in str(type(output_normalizationcnn))


def test_shape_inputs_equals_shape_outputs(image_cross):
    assert tf.convert_to_tensor(image_cross).shape == output_normalizationcnn.shape


def test_norm_arrays(image_cross):
    np.testing.assert_array_almost_equal((image_cross-0.5)/2, output_normalizationcnn.numpy())
