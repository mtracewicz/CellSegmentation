import tensorflow as tf
from keras import backend as K

N_FILTERS=6
DROPOUT=0.5

def contracting_block(input_tensor, n_filters):
    c = conv2d_block(input_tensor, n_filters)
    p = tf.keras.layers.MaxPooling2D((2, 2))(c)
    p = tf.keras.layers.Dropout(DROPOUT)(p)
    return (c,p)

def expansive_block(input_tensor, n_filters, concatenated_layer):
    u = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same') (input_tensor)
    u = tf.keras.layers.concatenate([u, concatenated_layer])
    u = tf.keras.layers.Dropout(DROPOUT)(u)
    c = conv2d_block(u, n_filters)
    return c

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    # first layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def get_model(input_size):
    inputs = tf.keras.Input(input_size)
    # contracting path
    
    l1 = contracting_block(inputs,N_FILTERS)
    l2 = contracting_block(l1[1], N_FILTERS*2)
    l3 = contracting_block(l2[1], N_FILTERS*4)

    l4 = conv2d_block(l3[1], N_FILTERS*8)

    # expansive path
    l5 = expansive_block(l4, N_FILTERS*4, l3[0])
    l6 = expansive_block(l5, N_FILTERS*2, l2[0])
    l7 = expansive_block(l6, N_FILTERS*1, l1[0])
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid') (l7)

    return tf.keras.Model(inputs = inputs, outputs = outputs)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef

def save_model(model, checkpoint_name, to_json = True):
    # Saving compiled model
    tf.keras.models.save_model(
        model,
        f'./checkpoints/{checkpoint_name}/{checkpoint_name}'
    )
    # Saving fo tensorflowjs 
    if to_json:
        with open(f'./checkpoints/{checkpoint_name}/{checkpoint_name}.json', "w") as file:
            file.write(model.to_json())
            file.flush()
