import tensorflow as tf

dict_activations = {"relu": "relu",
		    "selu": "selu",
                    "LeakyRelu": tf.keras.layers.LeakyReLU,
                    "gelu": tf.keras.activations.gelu
                    }


def load_activation(name_activation):
    if name_activation == "LeakyRelu":
        return dict_activations[name_activation]()
    else:
        return dict_activations[name_activation]

