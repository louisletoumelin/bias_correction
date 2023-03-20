import tensorflow as tf


def load_initializer(name_initializer, *args, **kwargs):

    if name_initializer == "GlorotNormal":
        return tf.keras.initializers.GlorotNormal(*args, **kwargs)

    elif name_initializer == "GlorotUniform":
        return tf.keras.initializers.GlorotUniform(*args, **kwargs)

    elif name_initializer == "HeNormal":
        return tf.keras.initializers.HeNormal(*args, **kwargs)

    elif name_initializer == "HeUniform":
        return tf.keras.initializers.HeUniform(*args, **kwargs)

    elif name_initializer == "Identity":
        return tf.keras.initializers.Identity(*args, **kwargs)

    elif name_initializer == "LecunNormal":
        return tf.keras.initializers.LecunNormal(*args, **kwargs)

    elif name_initializer == "LecunUniform":
        return tf.keras.initializers.LecunUniform(*args, **kwargs)

    elif name_initializer == "Constant":
        if "seed" in kwargs:
            kwargs.pop("seed")
        return tf.keras.initializers.Constant(*args, **kwargs)

    elif name_initializer == "Orthogonal":
        return tf.keras.initializers.Orthogonal(*args, **kwargs)

    elif name_initializer == "RandomNormal":
        return tf.keras.initializers.RandomNormal(*args, **kwargs)

    elif name_initializer == "RandomUniform":
        return tf.keras.initializers.RandomUniform(*args, **kwargs)

    elif name_initializer == "TruncatedNormal":
        return tf.keras.initializers.TruncatedNormal(*args, **kwargs)

    elif name_initializer is None:
        return None

    else:
        raise NotImplementedError(f"Initializer {name_initializer} not implemented")

