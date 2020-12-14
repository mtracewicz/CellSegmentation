import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

N_FILTERS = 48
DROPOUT = 0.5


def contracting_block(input_tensor, n_filters):
    c = conv2d_block(input_tensor, n_filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(c)
    p = tf.keras.layers.Dropout(DROPOUT)(p)
    return (c, p)


def expansive_block(input_tensor, n_filters, concatenated_layer):
    u = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), strides=(
        2, 2), padding='same', data_format="channels_last")(input_tensor)
    u = tf.keras.layers.concatenate([u, concatenated_layer])
    u = tf.keras.layers.Dropout(DROPOUT)(u)
    c = conv2d_block(u, n_filters)
    return c


def conv2d_block(input_tensor, n_filters, kernel_size=3):
    # first layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", data_format="channels_last",
                               padding="same")(input_tensor)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal", data_format="channels_last",
                               padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def get_model(input_size: np.ndarray) -> tf.keras.Model:
    inputs = tf.keras.Input(input_size)
    # contracting path

    l1 = contracting_block(inputs, N_FILTERS)
    l2 = contracting_block(l1[1], N_FILTERS*2)
    l3 = contracting_block(l2[1], N_FILTERS*4)

    l4 = conv2d_block(l3[1], N_FILTERS*8)

    # expansive path
    l5 = expansive_block(l4, N_FILTERS*4, l3[0])
    l6 = expansive_block(l5, N_FILTERS*2, l2[0])
    l7 = expansive_block(l6, N_FILTERS, l1[0])
    outputs = tf.keras.layers.Conv2D(4, 1, activation='sigmoid')(l7)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
