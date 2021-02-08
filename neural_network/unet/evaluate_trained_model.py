import os
import sys

import tensorflow as tf
from neural_network.unet.image_loader import load_images

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: evaluate_trained_model model_name predictions truths [custom_functions]")
        exit()

    # Restoring model
    metric_and_loss_file = sys.argv[4] if len(
        sys.argv) >= 5 else 'neural_network.unet.dice_coef'
    exec(f'from {metric_and_loss_file} import custom_metric, custom_loss')

    model = tf.keras.models.load_model(
        os.path.join('trained_models', sys.argv[1]), custom_objects={'custom_metric': custom_metric, 'custom_loss': custom_loss})
    # Loading data
    x_test = load_images(sys.argv[2])
    y_test = load_images(sys.argv[3])

    model.evaluate(x_test, y_test)
