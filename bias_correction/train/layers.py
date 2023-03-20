import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa


class RotationLayer(Layer):
    """
    Custom layer for rotations of topographic and wind maps

    Custom layers:
    https://www.tensorflow.org/tutorials/customization/custom_layers

    __init__, build and call must be implemented
    """
    def __init__(self,
                 clockwise,
                 unit_input,
                 interpolation="nearest",
                 fill_mode="constant",
                 fill_value=-1):

        super(RotationLayer, self).__init__()
        self.clockwise = clockwise
        self.unit_input = unit_input

        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value

    def build(self, input_shape):
        super(RotationLayer, self).build(input_shape)

    @staticmethod
    def tf_deg2rad(angle):
        """
        Converts angles in degrees to radians

        Note: pi/180 = 0.01745329
        """

        return angle * tf.convert_to_tensor(0.01745329)

    def call(self, inputs, wind_dir):
        # Convert to degrees
        if self.unit_input == "degree":
            wind_dir = self.tf_deg2rad(wind_dir)

        # Select direction of the rotation
        if self.clockwise:
            angles = -np.pi / 2 - wind_dir
        else:
            angles = np.pi / 2 + wind_dir

        result = tfa.image.rotate(inputs,
                                  angles,
                                  interpolation=self.interpolation,
                                  fill_mode=self.fill_mode,
                                  fill_value=self.fill_value)
        tf.convert_to_tensor(result)
        result = tf.keras.backend.reshape(result, tf.shape(inputs))
        return result


class CropTopography(Layer):
    def __init__(self,
                 initial_length_x=140,
                 initial_length_y=140,
                 y_offset=39,
                 x_offset=34,
                 ):

        super(CropTopography, self).__init__()
        self.y_offset_left = tf.constant(initial_length_y//2 - y_offset)
        self.y_offset_right = tf.constant(initial_length_y//2 + y_offset + 1)
        self.x_offset_left = tf.constant(initial_length_x//2 - x_offset)
        self.x_offset_right = tf.constant(initial_length_x//2 + x_offset + 1)

    def build(self, input_shape):
        super(CropTopography, self).build(input_shape)

    def call(self, topos):
        return topos[:, self.y_offset_left:self.y_offset_right, self.x_offset_left:self.x_offset_right, :]


class SelectCenter(Layer):
    def __init__(self,
                 len_y=79,
                 len_x=69,
                 ):

        super(SelectCenter, self).__init__()
        self.len_y = len_y
        self.len_x = len_x

    def build(self, input_shape):
        super(SelectCenter, self).build(input_shape)

    def call(self, inputs):
        if len(inputs.shape) == 3:
            return inputs[:, self.len_y//2, self.len_x//2]
        elif len(inputs.shape) == 4:
            return inputs[:, self.len_y//2, self.len_x//2, :]


class MeanTopo(Layer):
    def __init__(self):
        super(MeanTopo, self).__init__()

    def build(self, input_shape):
        super(MeanTopo, self).build(input_shape)

    def call(self, inputs):
        return tf.convert_to_tensor(tf.math.reduce_mean(inputs, axis=[-2, -3]), tf.float32)


class NormalizationInputs(Layer):
    """
    Normalization of inputs before calling the CNN
    """
    def __init__(self):

        super(NormalizationInputs, self).__init__()

    def build(self, input_shape):
        super(NormalizationInputs, self).build(input_shape)

    def call(self, inputs, mean, std):
        num = tf.convert_to_tensor(inputs - mean, dtype=tf.float32)
        den = tf.convert_to_tensor(std + tf.keras.backend.epsilon(), dtype=tf.float32)

        return num / den


class Normalization(Layer):
    """
    Normalization of inputs before calling the CNN
    """
    def __init__(self, std):

        super(Normalization, self).__init__()
        self.std = tf.convert_to_tensor(std, tf.float32)

    def build(self, input_shape):
        super(Normalization, self).build(input_shape)

    def call(self, inputs):
        mean = tf.expand_dims(tf.expand_dims(tf.math.reduce_mean(inputs, axis=[-2, -3]), axis=-1), axis=-1)
        return (inputs-mean) / self.std


class ActivationArctan(Layer):
    """
    Normalization of inputs before calling the CNN
    """
    def __init__(self,
                 alpha):
        super(ActivationArctan, self).__init__()
        self.alpha = tf.convert_to_tensor(alpha)

    def build(self, input_shape):
        super(ActivationArctan, self).build(input_shape)

    @staticmethod
    def reshape_as_inputs(inputs, outputs):
        if outputs.shape[-1] is None:
            outputs = tf.keras.backend.reshape(outputs, tf.shape(inputs))
        return outputs

    def call(self, output_cnn, wind_nwp):
        wind_nwp = tf.expand_dims(tf.expand_dims(tf.expand_dims(wind_nwp, axis=-1), axis=-1), axis=-1)
        scaled_wind = wind_nwp * output_cnn / tf.convert_to_tensor(3.)  # 3 = ARPS initialization speed
        return self.reshape_as_inputs(output_cnn, self.alpha * tf.math.atan(scaled_wind/self.alpha))


class SimpleScaling(Layer):
    """
    Normalization of inputs before calling the CNN
    """
    def __init__(self):
        super(SimpleScaling, self).__init__()

    def build(self, input_shape):
        super(SimpleScaling, self).build(input_shape)

    @staticmethod
    def reshape_as_inputs(inputs, outputs):
        if outputs.shape[-1] is None:
            outputs = tf.keras.backend.reshape(outputs, tf.shape(inputs))
        return outputs

    def call(self, output_cnn, wind_nwp):
        wind_nwp = tf.expand_dims(tf.expand_dims(tf.expand_dims(wind_nwp, axis=-1), axis=-1), axis=-1)
        scaled_wind = wind_nwp * output_cnn / tf.convert_to_tensor(3.)  # 3 = ARPS initialization speed
        result = self.reshape_as_inputs(output_cnn, scaled_wind)
        return result


class Components2Speed(Layer):
    """
    Normalization of inputs before calling the CNN
    """
    def __init__(self):
        super(Components2Speed, self).__init__()

    def build(self, input_shape):
        super(Components2Speed, self).build(input_shape)

    def call(self, inputs):

        UV = tf.sqrt(inputs[:, :, :, 0]**2+inputs[:, :, :, 1]**2)

        if len(UV.shape) == 3:
            UV = tf.expand_dims(UV, axis=-1)

        return UV


class Components2Direction(Layer):
    """
    Normalization of inputs before calling the CNN

    Unit output in degree
    """
    def __init__(self):
        super(Components2Direction, self).__init__()

    def build(self, input_shape):
        super(Components2Direction, self).build(input_shape)

    @staticmethod
    def tf_rad2deg(inputs):
        """Convert input in radian to degrees"""
        return tf.convert_to_tensor(57.2957795, dtype=tf.float32) * inputs

    def call(self, inputs):
        outputs = tf.math.atan2(inputs[:, :, :, 0], inputs[:, :, :, 1])
        outputs = self.tf_rad2deg(outputs)
        constant_0 = tf.convert_to_tensor(180, dtype=tf.float32)
        constant_1 = tf.convert_to_tensor(360, dtype=tf.float32)
        outputs = tf.math.mod(constant_0 + outputs, constant_1)
        if len(outputs.shape) == 3:
            outputs = tf.expand_dims(outputs, -1)
        return outputs


class SpeedDirection2Components(Layer):
    """
    Normalization of inputs before calling the CNN

    Unit output in degree
    """

    def __init__(self, unit_input):
        self.unit_input = unit_input
        super(SpeedDirection2Components, self).__init__()

    def build(self, input_shape):
        super(SpeedDirection2Components, self).build(input_shape)

    @staticmethod
    def reshape_as_inputs(inputs, outputs):
        if outputs.shape[-1] is None:
            outputs = tf.keras.backend.reshape(outputs, tf.shape(inputs))
        return outputs

    @staticmethod
    def tf_deg2rad(angle):
        """
        Converts angles in degrees to radians

        Note: pi/180 = 0.01745329
        """

        return angle * tf.convert_to_tensor(0.01745329)

    def call(self, speed, direction):

        if self.unit_input == "degree":
            direction = self.tf_deg2rad(direction)

        U_zonal = - tf.math.sin(direction) * speed
        V_meridional = - tf.math.cos(direction) * speed

        U_zonal = self.reshape_as_inputs(speed, U_zonal)
        V_meridional = self.reshape_as_inputs(speed, V_meridional)

        if len(U_zonal.shape) == 3:
            U_zonal = tf.expand_dims(U_zonal, -1)
        if len(U_zonal.shape) == 3:
            V_meridional = tf.expand_dims(V_meridional, -1)
        return U_zonal, V_meridional


class Components2Alpha(Layer):

    def __init__(self):
        self.unit_output = "radian"
        super(Components2Alpha, self).__init__()

    def build(self, input_shape):
        super(Components2Alpha, self).build(input_shape)

    @staticmethod
    def tf_deg2rad(angle):
        """
        Converts angles in degrees to radians

        Note: pi/180 = 0.01745329
        """

        return angle * tf.convert_to_tensor(0.01745329)

    def get_unit_output(self):
        return self.unit_output

    def reshape_output(self, output):
        if len(output.shape) == 3:
            return tf.expand_dims(output, -1)

    def call(self, inputs):
        result = tf.where(inputs[:, :, :, 0] == 0.,
                          tf.where(inputs[:, :, :, 1] == 0.,
                                   0.,
                                   tf.sign(inputs[:, :, :, 1]) * tf.cast(3.14159 / 2., dtype=tf.float32)),
                          tf.math.atan(inputs[:, :, :, 1] / inputs[:, :, :, 0]))
        return self.reshape_output(result)


class Alpha2Direction(Layer):
    def __init__(self, unit_direction, unit_alpha):
        self.unit_direction = unit_direction
        self.unit_alpha = unit_alpha
        self.unit_output = "degree"
        super(Alpha2Direction, self).__init__()

    def build(self, input_shape):
        super(Alpha2Direction, self).build(input_shape)

    @staticmethod
    def tf_deg2rad(angle):
        """
        Converts angles in degrees to radians

        Note: pi/180 = 0.01745329
        """

        return angle * tf.convert_to_tensor(0.01745329)

    @staticmethod
    def tf_rad2deg(inputs):
        """Convert input in radian to degrees"""
        return tf.convert_to_tensor(57.2957795) * inputs

    def get_unit_output(self):
        return self.unit_output

    @staticmethod
    def reshape_as_inputs(inputs, outputs):
        if outputs.shape[-1] is None:
            outputs = tf.keras.backend.reshape(outputs, tf.shape(inputs))
        return outputs

    def call(self, direction, alpha):
        if self.unit_direction == "radian":
            direction = self.tf_rad2deg(direction)

        if self.unit_alpha == "radian":
            alpha = self.tf_rad2deg(alpha)
        direction = tf.expand_dims(tf.expand_dims(tf.expand_dims(direction, axis=-1), axis=-1), axis=-1)
        outputs = tf.math.mod(direction - alpha, 360)
        result = self.reshape_as_inputs(alpha, outputs)
        return result
