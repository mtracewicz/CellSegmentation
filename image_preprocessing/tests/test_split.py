import numpy as np
import pytest
from image_preprocessing.preprocessing.split import create_new_image
from PIL import Image


def test_split():
    input = np.zeros((100, 100, 3), dtype=np.uint8)
    input[:25,:25] = 1
    input[25:50,25:50] = 2
    expected = np.ones((10, 10, 3))
    expected = input[20:30,20:30]
    resault = np.array(create_new_image(input, 4, 4, 10, 10, 5, 5))
    assert(np.array_equal(expected, resault))
