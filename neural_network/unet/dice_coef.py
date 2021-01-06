from tensorflow.keras import backend as K


def custom_metric(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return (2. * K.sum(y_true_f * y_pred_f) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def custom_loss(y_true, y_pred):
    return 1-custom_metric(y_true, y_pred)
