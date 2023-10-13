import tensorflow as tf

from typing import Union, Callable

mse = tf.keras.losses.MeanSquaredError()


class PenalizedMSE(tf.keras.losses.Loss):

    def __init__(self, penalty=10, speed_threshold=5):
        super().__init__()
        self.penalty = tf.convert_to_tensor(penalty, dtype=tf.float32)
        self.speed_threshold = tf.convert_to_tensor(speed_threshold, dtype=tf.float32)

    def call(self, y_true, y_pred):
        result = tf.where(tf.greater(y_true, self.speed_threshold),
                          mse(y_true, y_pred) * self.penalty,
                          mse(y_true, y_pred))

        return result


class MSESmall(tf.keras.losses.Loss):

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        result = mse(y_true, y_pred) / 100
        return result


class Pinball(tf.keras.losses.Loss):

    def __init__(self, tho=0.85):
        super().__init__()
        self.tho = tho

    def call(self, y_true, y_pred):
        result = tf.where(tf.greater(y_pred, y_true),
                          (1 - self.tho) * mse(y_true, y_pred),
                          self.tho * mse(y_true, y_pred))

        return result


class PinballWeight(tf.keras.losses.Loss):

    def __init__(self, tho=0.95):
        super().__init__()
        self.tho = tho

    def call(self, y_true, y_pred):
        epsilon = 0.001
        alpha = (epsilon + y_true) / (epsilon + y_pred)
        result = tf.where(tf.greater(y_pred, y_true),
                          alpha * (1 - self.tho) * mse(y_true, y_pred),
                          alpha * self.tho * mse(y_true, y_pred))

        return result


class PinballProportional(tf.keras.losses.Loss):

    def __init__(self, tho=0.6):
        super().__init__()
        self.tho = tho

    def call(self, y_true, y_pred):
        result = y_true * tf.where(tf.greater(y_pred, y_true),
                                   (1 - self.tho) * mse(y_true, y_pred),
                                   self.tho * mse(y_true, y_pred))

        return result


class MSEProportionalInput(tf.keras.losses.Loss):

    def __init__(self, penalty=1):
        super().__init__()
        self.penalty = tf.convert_to_tensor(penalty, dtype=tf.float32)

    def call(self, y_true, y_pred):
        result = mse(y_true, y_pred) * y_true

        return result


class MSEpower(tf.keras.losses.Loss):

    def __init__(self, penalty=1, power=2):
        super().__init__()
        self.penalty = tf.convert_to_tensor(penalty, dtype=tf.float32)
        self.power = tf.convert_to_tensor(power, dtype=tf.float32)

    def call(self, y_true, y_pred):
        result = mse(y_true, y_pred) * y_true ** self.power

        return result


class CosineDistance(tf.keras.losses.Loss):

    def __init__(self, power=1):
        super().__init__()
        self.power = power

    @staticmethod
    def tf_deg2rad(angle):
        """
        Converts angles in degrees to radians

        Note: pi/180 = 0.01745329
        """

        return angle * tf.convert_to_tensor(0.01745329)

    def call(self, y_true, y_pred):
        return (1 - tf.math.cos(self.tf_deg2rad(y_true) - self.tf_deg2rad(y_pred))) ** self.power


class MixedLoss(tf.keras.losses.Loss):

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tho = 0.6

    @staticmethod
    def tf_deg2rad(angle):
        """
        Converts angles in degrees to radians

        Note: pi/180 = 0.01745329
        """

        return angle * tf.convert_to_tensor(0.01745329)

    def call(self, y_true, y_pred):
        speed_true = y_true[0]
        speed_pred = y_pred[0]
        dir_true = y_true[1]
        dir_pred = y_pred[1]

        loss_speed = y_true * tf.where(tf.greater(speed_pred, speed_true),
                                       (1 - self.tho) * mse(speed_true, speed_pred),
                                       self.tho * mse(speed_true, speed_pred))

        # 10 is to express loss_dir with the same order of magnitude as loss_speed
        loss_dir = 10 * (1 - tf.math.cos(self.tf_deg2rad(dir_true) - self.tf_deg2rad(dir_pred)))

        return self.alpha * loss_speed + self.beta * loss_dir


dict_loss = {"mse": "mse",
             "mae": "mean_absolute_error",
             "penalized_mse": PenalizedMSE,
             "mse_proportional": MSEProportionalInput,
             "mse_power": MSEpower,
             "pinball": Pinball,
             "pinball_proportional": PinballProportional,
             "pinball_weight": PinballWeight,
             "cosine_distance": CosineDistance,
             "mixed": [PinballProportional(), CosineDistance()],
             "msesmall": MSESmall}


def load_loss(name_loss: str, *args, **kwargs) -> Union[str, Callable]:
    if isinstance(dict_loss[name_loss], str):
        return name_loss
    elif isinstance(dict_loss[name_loss], list):
        return dict_loss[name_loss]
    else:
        return dict_loss[name_loss](*args, **kwargs)
