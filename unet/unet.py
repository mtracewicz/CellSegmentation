from unet.model import dice_coef, get_model
import os
import sys
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from unet.hyperparameters import (BATCH_SIZE, EPOCHS, LEARNING_RATE,
                                  VALIDATION_SPLIT)


def load_images(dir: str, images_shape: Tuple[int, int, int] = (200, 200, 3)) -> np.array:
    IMAGES_DIRECTORY = os.path.join(os.getcwd(), dir)
    filenames = os.listdir(IMAGES_DIRECTORY)
    number_of_images = len(filenames)
    images = np.zeros((number_of_images)+images_shape)
    for i, filename in enumerate(filenames):
        img = Image.open(os.path.join(IMAGES_DIRECTORY, filename))
        images[i] = np.array(img)
    # Data normalization from [0,255] to [0.0,1.0]
    images = images/255.0
    return images


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python unet.py inputs_directory truths_directory checkpoint_name")
        exit()

    print('Loading data')
    # Loading data
    x_train = load_images(sys.argv[1])
    y_train = load_images(sys.argv[2])[:, :, :, 3]

    split = int(x_train.shape[0] * (1-VALIDATION_SPLIT))

    print('Seting up enviorment')

    # Seting tf/keras options
    tf.keras.backend.set_floatx('float64')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print('Creating model')
    # Establish the model's topography
    model = get_model((200, 200, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy, metrics=[dice_coef])

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
