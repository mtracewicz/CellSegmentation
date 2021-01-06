import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

N_FILTERS = 32 
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
    outputs = tf.keras.layers.Conv2D(3, 1, activation='sigmoid')(l7)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (2. * K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef2(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef2(y_true, y_pred)

def custom_metric(y_true, y_pred):
    true_black = tf.where(K.flatten(y_true) == [0,0,0], 1, 0)
    true_red = tf.where(K.flatten(y_true) == [255,0,0], 1, 0)

    good_predictions = tf.where(y_true==y_pred,1,0)

    reds_ratio = K.sum(good_predictions*true_red)/K.sum(true_red)
    blacks_ratio = K.sum(good_predictions*true_black)/K.sum(true_black)
    return ((3*reds_ratio+blacks_ratio)+1)/5.

def custom_loss(y_true, y_pred):
    return 1-custom_metric(y_true, y_pred)

