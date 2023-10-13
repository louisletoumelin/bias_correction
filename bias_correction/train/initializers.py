import tensorflow as tf


dict_initializer = {"GlorotNormal": tf.keras.initializers.GlorotNormal,
                    "GlorotUniform": tf.keras.initializers.GlorotUniform,
                    "HeNormal": tf.keras.initializers.HeNormal,
                    "HeUniform": tf.keras.initializers.HeUniform,
                    "Identity": tf.keras.initializers.Identity,
                    "LecunNormal": tf.keras.initializers.LecunNormal,
                    "LecunUniform": tf.keras.initializers.LecunUniform,
                    "Constant": tf.keras.initializers.Constant,
                    "Orthogonal": tf.keras.initializers.Orthogonal,
                    "RandomNormal": tf.keras.initializers.RandomNormal,
                    "RandomUniform": tf.keras.initializers.RandomUniform,
                    "TruncatedNormal": tf.keras.initializers.TruncatedNormal,
                    None: None
                    }


def load_initializer(name_initializer, *args, **kwargs):

    if name_initializer == "Constant":
        if "seed" in kwargs:
            kwargs.pop("seed")
    if isinstance(args, tuple):
        args = []
    return dict_initializer[name_initializer](*args, **kwargs)

