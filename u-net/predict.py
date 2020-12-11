import tensorflow as tf
import numpy as np
from PIL import Image

import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usagepredict.py model file")
        exit()

    # Restoring model
    model = tf.keras.models.load_model(
        f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}')

    # Loading data
    image = Image.open(sys.argv[2])
    data = np.array(image, dtype = np.float64)
    # Expanding to add batch dimension
    data = np.expand_dims(data,axis=0)

    # Making a prediction and converting to uint8
    prediction = (model.predict(data)*255)[0]
    prediction = (np.append(prediction,np.zeros((200,200,2)),axis=2)).astype(np.uint8)

    # Creating and saving an image
    res = Image.fromarray(prediction)
    res.save("res.png")
