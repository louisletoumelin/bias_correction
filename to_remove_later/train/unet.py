from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, UpSampling2D, MaxPooling2D, concatenate, Cropping2D

prm = {
    'kernel_size': (3, 3),
    'padding': 'same',
    'nb_filters': 32,
    'initializer': None,
    'up_conv': (2, 2),
    'activation': 'relu',
    'activation_regression': 'linear',
    'pool_size': (2, 2),
    'nb_channels_output': 3,
}


def create_unet(input_shape):

    inputs = Input(input_shape)
    zero_padding = ZeroPadding2D(padding=((0, 1), (0, 1)), input_shape=input_shape)(inputs)

    '''
    1st conv/pool
    '''
    conv1 = Conv2D(prm["nb_filters"],
                   prm['kernel_size'],
                   activation=prm["activation"],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv1_0')(zero_padding)
    conv1 = Conv2D(prm['nb_filters'],
                   prm['kernel_size'],
                   activation=prm['activation'],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv1')(conv1)
    pool1 = MaxPooling2D(pool_size=prm['pool_size'],
                         name='pool1')(conv1)
    '''
    2nd conv/pool
    '''
    conv2 = Conv2D(2 * prm['nb_filters'],
                   prm['kernel_size'],
                   activation=prm['activation'],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv2_0')(pool1)
    conv2 = Conv2D(2 * prm['nb_filters'],
                   prm['kernel_size'],
                   activation=prm['activation'],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=prm['pool_size'],
                         name='pool2')(conv2)

    '''
    3rd conv/pool
    '''
    conv3 = Conv2D(4 * prm['nb_filters'],
                   prm['kernel_size'],
                   activation=prm['activation'],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv3_0')(pool2)
    conv3 = Conv2D(4 * prm['nb_filters'],
                   prm['kernel_size'],
                   activation=prm['activation'],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=prm['pool_size'],
                         name='pool3')(conv3)

    '''
    4th conv/pool/up
    '''
    conv4 = Conv2D(8 * prm['nb_filters'],
                   prm['kernel_size'],
                   activation=prm['activation'],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv4_0')(pool3)
    conv4 = Conv2D(8 * prm['nb_filters'],
                   prm['kernel_size'],
                   activation=prm['activation'],
                   padding=prm['padding'],
                   kernel_initializer=prm['initializer'],
                   name='conv4')(conv4)
    up4 = UpSampling2D(size=prm['up_conv'],
                       name='up4_0')(conv4)
    up4 = Conv2D(4 * prm['nb_filters'],
                 prm['up_conv'],
                 activation=prm['activation'],
                 padding=prm['padding'],
                 kernel_initializer=prm['initializer'],
                 name='up4')(up4)
    up4 = ZeroPadding2D(padding=((0, 0), (0, 0)))(up4)

    '''
    3rd up
    '''
    merge3 = concatenate([conv3, up4],
                         axis=3,
                         name='concat_3')
    conv3_up = Conv2D(4 * prm['nb_filters'],
                      prm['kernel_size'],
                      activation=prm['activation'],
                      padding=prm['padding'],
                      kernel_initializer=prm['initializer'],
                      name='conv3_up_0')(merge3)
    conv3_up = Conv2D(4 * prm['nb_filters'],
                      prm['kernel_size'],
                      activation=prm['activation'],
                      padding=prm['padding'],
                      kernel_initializer=prm['initializer'],
                      name='conv3_up')(conv3_up)
    up3 = UpSampling2D(size=prm['up_conv'],
                       name='up3_0')(conv3_up)
    up3 = Conv2D(2 * prm['nb_filters'],
                 prm['up_conv'],
                 activation=prm['activation'],
                 padding=prm['padding'],
                 kernel_initializer=prm['initializer'],
                 name='up3')(up3)
    #up3 = ZeroPadding2D(padding=((0, 0), (0, 0)))(up3) for test case with Nora
    up3 = ZeroPadding2D(padding=((0, 1), (0, 1)))(up3)
    '''
    2nd up
    '''
    merge2 = concatenate([conv2, up3],
                         axis=3,
                         name='concat_2')
    conv2_up = Conv2D(2 * prm['nb_filters'],
                      prm['kernel_size'],
                      activation=prm['activation'],
                      padding=prm['padding'],
                      kernel_initializer=prm['initializer'],
                      name='conv2_up_0')(merge2)
    conv2_up = Conv2D(2 * prm['nb_filters'],
                      prm['kernel_size'],
                      activation=prm['activation'],
                      padding=prm['padding'],
                      kernel_initializer=prm['initializer'],
                      name='conv2_up')(conv2_up)
    up2 = UpSampling2D(size=prm['up_conv'],
                       name='up2_0')(conv2_up)
    up2 = Conv2D(1 * prm['nb_filters'],
                 prm['up_conv'],
                 activation=prm['activation'],
                 padding=prm['padding'],
                 kernel_initializer=prm['initializer'],
                 name='up2')(up2)

    '''
    1st up
    '''
    merge1 = concatenate([conv1, up2],
                         axis=3,
                         name='concat_1')
    conv1_up = Conv2D(prm['nb_filters'],
                      prm['kernel_size'],
                      activation=prm['activation'],
                      padding=prm['padding'],
                      kernel_initializer=prm['initializer'],
                      name='conv1_up_0')(merge1)
    conv1_up = Conv2D(prm['nb_filters'],
                      prm['kernel_size'],
                      activation=prm['activation'],
                      padding=prm['padding'],
                      kernel_initializer=prm['initializer'],
                      name='conv1_up')(conv1_up)
    conv1_1 = Conv2D(prm['nb_channels_output'],
                     1,
                     activation=prm['activation_regression'],
                     name='conv1_1')(conv1_up)
    up1 = Cropping2D(cropping=((0, 1), (0, 1)))(conv1_1)

    model = Model(inputs=inputs, outputs=up1)

    return model
