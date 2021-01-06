import tensorflow as tf
from tensorflow import backend as K


def custom_metric(y_true, y_pred):
    true_black = tf.where(K.flatten(y_true) == [0, 0, 0], 1, 0)
    true_red = tf.where(K.flatten(y_true) == [255, 0, 0], 1, 0)

    good_predictions = tf.where(y_true == y_pred, 1, 0)

    reds_ratio = K.sum(good_predictions*true_red)/K.sum(true_red)
    blacks_ratio = K.sum(good_predictions*true_black)/K.sum(true_black)
    return ((3*reds_ratio+blacks_ratio)+1)/5.


def custom_loss(y_true, y_pred):
    return 1-custom_metric(y_true, y_pred)
