import numpy as np
import tensorflow as tf
import pytest


@pytest.fixture(scope="module")
def results():

    from bias_correction.train.model import build_model
    from bias_correction.config.config_train import config

    model = build_model(config)
    model.summary()
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    nb_features = 500
    nb_epochs = 5
    topos = 2000*np.ones((nb_features, 140, 140, 1))
    nwp_variables = 3*np.ones((nb_features, 47))
    target = 10 * np.ones((nb_features, 2))

    id_weights_0 = 3 # Dense layer
    id_weights = 11 # DEVINE CNN

    w0_0 = np.copy(model.layers[id_weights_0].weights[0])
    w0 = np.copy(model.layers[id_weights].weights[0])
    result = model.fit((topos, nwp_variables), target, epochs=nb_epochs, batch_size=1)
    w1_0 = np.copy(model.layers[id_weights_0].weights[0])
    w1 = np.copy(model.layers[id_weights].weights[0])

    return result, w0_0, w0, w1_0, w1


def test_training_loss_decreases(results):
    result = results[0]
    assert result.history["loss"][-1] < 0.5*result.history["loss"][0]


def test_DEVINE_weights_are_not_updated(results):
    w0 = results[2]
    w1 = results[4]
    np.testing.assert_allclose(w1-w0, np.zeros_like(w1))


def test_weights_dense_layer_are_updated(results):
    w0_0 = results[1]
    w1_0 = results[3]
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, w1_0 - w0_0, np.zeros_like(w1_0))
