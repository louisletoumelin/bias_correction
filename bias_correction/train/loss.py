import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()


class PenalizedMSE(tf.keras.losses.Loss):

    def __init__(self, penalty, speed_threshold):
        super().__init__()
        self.penalty = tf.convert_to_tensor(penalty, dtype=tf.float32)
        self.speed_threshold = tf.convert_to_tensor(speed_threshold, dtype=tf.float32)

    def call(self, y_true, y_pred):

        result = tf.where(tf.greater(y_true, self.speed_threshold),
                          mse(y_true, y_pred) * self.penalty,
                          mse(y_true, y_pred))

        return result


class Pinball(tf.keras.losses.Loss):

    def __init__(self, tho):
        super().__init__()
        self.tho = tho

    def call(self, y_true, y_pred):

        result = tf.where(tf.greater(y_pred, y_true),
                          (1-self.tho)*mse(y_true, y_pred),
                          self.tho*mse(y_true, y_pred))

        return result


class PinballWeight(tf.keras.losses.Loss):

    def __init__(self, tho):
        super().__init__()
        self.tho = tho

    def call(self, y_true, y_pred):
        epsilon = 0.001
        alpha = (epsilon + y_true) / (epsilon + y_pred)
        result = tf.where(tf.greater(y_pred, y_true),
                          alpha*(1-self.tho)*mse(y_true, y_pred),
                          alpha*self.tho*mse(y_true, y_pred))

        return result


class PinballProportional(tf.keras.losses.Loss):

    def __init__(self, tho):
        super().__init__()
        self.tho = tho

    def call(self, y_true, y_pred):

        result = y_true * tf.where(tf.greater(y_pred, y_true),
                                   (1-self.tho)*mse(y_true, y_pred),
                                   self.tho*mse(y_true, y_pred))

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

        result = mse(y_true, y_pred) * y_true**self.power

        return result


def load_loss(name_loss, *args, **kwargs):

    if name_loss == "mse":
        return name_loss

    elif name_loss == "penalized_mse":

        penalty = kwargs["penalized_mse"].get("penalty")
        speed_threshold = kwargs["penalized_mse"].get("speed_threshold")
        assert penalty is not None, "Penalty keyword argument (int) must be provided for penalized loss"
        assert speed_threshold is not None, "Speed threshold (int) must be provided for penalized loss"

        return PenalizedMSE(penalty, speed_threshold)

    elif name_loss == "mse_proportional":

        penalty = kwargs["mse_proportional"].get("penalty", 1)

        return MSEProportionalInput(penalty)

    elif name_loss == "mse_power":

        penalty = kwargs["mse_proportional"].get("penalty", 1)
        power = kwargs["mse_proportional"].get("power", 2)
        assert penalty is not None, "Penalty keyword argument (int) must be provided for penalized loss"
        assert power is not None, "Speed threshold (int) must be provided for penalized loss"

        return MSEpower(penalty, power)

    elif name_loss == "pinball":

        tho = kwargs["pinball"].get("tho", 0.75)
        assert tho is not None, "Penalty keyword argument (int) must be provided for penalized loss"

        return Pinball(tho)

    elif name_loss == "pinball_proportional":

        tho = kwargs["pinball_proportional"].get("tho", 0.75)
        assert tho is not None, "Penalty keyword argument (int) must be provided for penalized loss"

        return PinballProportional(tho)

    elif name_loss == "pinball_weight":

        tho = kwargs["pinball_weight"].get("tho", 0.75)
        assert tho is not None, "Penalty keyword argument (int) must be provided for penalized loss"

        return PinballWeight(tho) #PinballProportional before

    else:
        raise NotImplementedError(f"Loss {name_loss} not implemented")


"""
def custom_loss(y_true, y_pred):
  penalty = 20

  # actual = 0.1 and pred = -0.05 should be penalized a lot more than actual = 0.1 and pred = 0.05
  loss = tf.cond(tf.logical_and(tf.greater(y_true, 0.0), tf.less(y_pred, 0.0)),
                   lambda: mse(y_true, y_pred) * penalty,
                   lambda: mse(y_true, y_pred) * penalty / 4)
  
  #actual = 0.1 and pred = 0.15 slightly more penalty than actual = 0.1 and pred = 0.05
  loss = tf.cond(tf.greater(y_pred, y_true),
                   lambda: loss * penalty / 2,
                   lambda: loss * penalty / 3)
  return loss 
"""
