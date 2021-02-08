import os
from typing import Tuple

import numpy as np
from PIL import Image


def load_images(dir: str, images_shape: Tuple[int, int, int] = (200, 200, 3)) -> np.array:
    IMAGES_DIRECTORY = os.path.join(os.getcwd(), dir)
    filenames = os.listdir(IMAGES_DIRECTORY)
    filenames.sort()
    number_of_images = len(filenames)
    images = np.zeros(tuple([number_of_images])+images_shape)

    for i, filename in enumerate(filenames):
        img = Image.open(os.path.join(IMAGES_DIRECTORY, filename))
        images[i] = np.array(img)

    # Data normalization from [0,255] to [0.0,1.0]
    images = images/255.0
    return images
