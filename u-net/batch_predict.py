import os
import tensorflow as tf
import numpy as np
from PIL import Image
from progress.bar import ChargingBar as cb
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usagepredict.py model dir")
        exit()

    # Restoring model
    model = tf.keras.models.load_model(
        f'./checkpoints/{sys.argv[1]}/{sys.argv[1]}')

    # Loading data
    imgs = os.listdir(sys.argv[2])
    l = []
    with cb('Predicting',max=len(imgs)) as b:
        for i, img in enumerate(imgs):
            image = Image.open(os.path.join(sys.argv[2],img))

            data = np.array(image, dtype = np.float64)
            data = np.expand_dims(data,axis=0)
            # Making a prediction and converting to uint8
            prediction = (model.predict(data)*255)[0]
            prediction = (np.append(prediction,np.zeros((200,200,2)),axis=2)).astype(np.uint8)
            
            x= len(np.where(prediction[0,]!=0))
            if x!=0:
                l.append((i,x))

            # Creating and saving an image
            res = Image.fromarray(prediction)
            res.save(os.path.join('_out',f"{i}_res.png"))
            b.next()
    print(f"Most coloured pixels: {max(l)[1]}")
