import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from progress.bar import ChargingBar as cb


def make_predictions_for_images_in_directory(model: tf.keras.Model, input_directory: str, output_directory: str = 'out'):
    if not os.path.exists(input_directory):
        raise ValueError('Directory does not exists')
    if not os.path.isdir(input_directory):
        raise NotADirectoryError()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loading data
    images = os.listdir(input_directory)
    with cb('Predicting', max=len(images)) as b:
        for image in images:
            make_prediction_for_image(model, os.path.join(input_directory, image), os.path.join(
                output_directory, f"{os.path.splitext(image)[0]}_prediction.png"))
            b.next()
    print('Finished!')


def make_prediction_for_image(model: tf.keras.Model, input: str, output_name: str = 'prediction.png'):
    if not os.path.exists(os.path.dirname(input)):
        raise ValueError('Parent directory does not exist!')
    if os.path.isdir(input):
        raise IsADirectoryError()
    prediction = make_prediction(model, input)
    save_prediction(prediction, output_name)


def make_prediction(model: tf.keras.Model, input: str) -> np.ndarray:
    if os.path.isdir(input):
        raise IsADirectoryError()

    # Loading data
    image = Image.open(input)
    data = np.array(image, dtype=np.float64)
    # Expanding to add batch dimension
    data = np.expand_dims(data, axis=0)

    # Making a prediction and converting to uint8
    prediction = (model.predict(data)*255)[0]
    return prediction.astype(np.uint8)


def save_prediction(prediction: np.ndarray, output_name: str = 'prediction.png'):
    # Creating and saving an image
    if type(prediction) != np.ndarray:
        raise TypeError('Prediction must be a valid numpy array')
    res = Image.fromarray(prediction)
    res.save(output_name)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: predict.py model file/directory")
        exit()

    metric_and_loss_file = sys.argv[3] if len(
        sys.argv) >= 4 else 'neural_network.unet.dice_coef'
    exec(f'from {metric_and_loss_file} import custom_metric, custom_loss')

    # Restoring model
    model = tf.keras.models.load_model(
        os.path.join('trained_models', sys.argv[1]), custom_objects={'custom_metric': custom_metric, 'custom_loss': custom_loss})

    if os.path.isdir(sys.argv[2]):
        make_predictions_for_images_in_directory(model, sys.argv[2])
    else:
        make_prediction_for_image(model, sys.argv[2])
