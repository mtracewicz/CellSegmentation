import os
import sys

import tensorflow as tf
from neural_network.unet.image_loader import load_images

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: evaluate_trained_model model_name input_directory output_directory")
        exit()

    # Restoring model
    model = tf.keras.models.load_model(
        os.path.join('trained_models', sys.argv[1]))
    # Loading data
    x_test = load_images(sys.argv[1])
    y_test = load_images(sys.argv[2])

    model.evaluate(x_test, y_test)
