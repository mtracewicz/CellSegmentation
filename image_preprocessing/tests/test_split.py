import numpy as np
import pytest
from image_preprocessing.image_preprocessing.split import create_new_image
from PIL import Image


def test_split():
    input = np.ones((100, 100, 3), dtype=np.uint8)
    expected = np.ones((10, 10, 3))
    resault = np.array(create_new_image(input, 5, 5, 10, 10, 5, 5, 10, 10))
    assert(np.array_equal(expected, resault))
