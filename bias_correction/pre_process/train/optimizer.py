import tensorflow as tf

dict_optimizers = {"RMSprop": tf.keras.optimizers.RMSprop,
                   "Adadelta": tf.keras.optimizers.Adadelta,
                   "Adagrad": tf.keras.optimizers.Adagrad,
                   "Adam": tf.keras.optimizers.Adam,
                   "Adamax": tf.keras.optimizers.Adamax,
                   "Ftrl": tf.keras.optimizers.Ftrl,
                   "Nadam": tf.keras.optimizers.Nadam,
                   "SGD": tf.keras.optimizers.SGD}


def load_optimizer(name_optimizer, learning_rate, *args, **kwargs):
    return dict_optimizers[name_optimizer](learning_rate, *args, **kwargs)
