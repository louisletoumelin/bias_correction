import tensorflow as tf


def load_optimizer(name_optimizer, learning_rate, *args, **kwargs):

    if name_optimizer == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate, *args, **kwargs)

    elif name_optimizer == "Adadelta":
        return tf.keras.optimizers.Adadelta(learning_rate, *args, **kwargs)

    elif name_optimizer == "Adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate, *args, **kwargs)

    elif name_optimizer == "Adam":
        return tf.keras.optimizers.Adam(learning_rate, *args, **kwargs)

    elif name_optimizer == "Adamax":
        return tf.keras.optimizers.Adamax(learning_rate, *args, **kwargs)

    elif name_optimizer == "Ftrl":
        return tf.keras.optimizers.Ftrl(learning_rate, *args, **kwargs)

    elif name_optimizer == "Nadam":
        return tf.keras.optimizers.Nadam(learning_rate, *args, **kwargs)

    elif name_optimizer == "SGD":
        return tf.keras.optimizers.SGD(learning_rate, *args, **kwargs)

    else:
        raise NotImplementedError(f"Optimizer {name_optimizer} not implemented")
