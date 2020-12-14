import os
import sys

import tensorflow as tf
from neural_network.unet.hyperparameters import (BATCH_SIZE, EPOCHS,
                                                 LEARNING_RATE,
                                                 VALIDATION_SPLIT)
from neural_network.unet.image_loader import load_images
from neural_network.unet.model import dice_coef, get_model

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python unet.py inputs_directory truths_directory checkpoint_name")
        exit()

    print('Loading data')
    # Loading data
    x_train = load_images(sys.argv[1])
    y_train = load_images(sys.argv[2],(200,200,4))[:, :, :, 3]

    split = int(x_train.shape[0] * (1-VALIDATION_SPLIT))

    print('Seting up enviorment')

    # Seting tf/keras options
    tf.keras.backend.set_floatx('float64')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print('Creating model')
    # Establish the model's topography
    model = get_model((200, 200, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=[dice_coef])

    print('Model info')
    print(model.summary())

    print('Traing model')
    model.fit(x_train[:split], y_train[:split],
              batch_size=BATCH_SIZE, epochs=EPOCHS)

    print('Model evaluation')
    model.evaluate(x_train[split:], y_train[split:])

    # Save model
    model.save(os.path.join('trained_models', sys.argv[3]))
    print('Finished!')
