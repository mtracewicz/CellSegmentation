import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

from hyperparameters import batch_size, epochs, learning_rate, validation_split
from model import get_model, save_model, dice_coef

def load_images(dir: str) -> np.array:
    IMAGES_DIRECTORY= os.path.join(os.getcwd(), dir)
    filenames = os.listdir(IMAGES_DIRECTORY)
    number_of_images = len(filenames)
    images = np.ones((number_of_images,200,200, 3))
    for i, filename in enumerate(filenames):
        img = Image.open(f"{IMAGES_DIRECTORY}/{filename}")
        images[i] = np.array(img)[:,:,:3]
    return images


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python unet.py inputs_directory truths_directory checkpoint_name")
        exit()

    print('Loading data')
    #Loading data
    x_train = load_images(sys.argv[1])
    y_train = load_images(sys.argv[2])[:,:,:,0]

    split = int(x_train.shape[0] * (1-validation_split))

    # Data normalization from [0,255] to [0.0,1.0]
    # and reshaping it for proper dimensions
    x_train= (x_train/255.0)
    y_train= (y_train/255.0)

    print('Seting up enviorment')

    # Seting tf/keras options 
    tf.keras.backend.set_floatx('float64')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print('Creating model')
    # Establish the model's topography
    model = get_model((200, 200, 3))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss="binary_crossentropy", metrics=[dice_coef])

    print('Model info')
    print(model.summary())

    print('Traing model')
    model.fit(x_train[:split], y_train[:split], batch_size = batch_size, epochs = epochs)

    print('Model evaluation')
    model.evaluate(x_train[split:],y_train[split:])

    # Save model
    save_model(model, sys.argv[3])
    print('Finished!')
