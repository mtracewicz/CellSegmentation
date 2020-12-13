import numpy as np
import pytest
from image_preprocessing.preprocessing.split import create_new_image, split_image, split_images_in_directory
from PIL import Image


def test_create_new_image():
    input = np.zeros((100, 100, 3), dtype=np.uint8)
    input[:25,:25] = 1
    input[25:50,25:50] = 2
    expected = input[20:30,20:30]
    resault = np.array(create_new_image(input, 4, 4, 10, 10, 5, 5))
    assert(np.array_equal(expected, resault))

def test_to_big_pocket_horizontal():
    input = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as e_info:
        create_new_image(input, 4, 4, 4, 10, 5, 5)

def test_to_big_pocket_vertical():
    input = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as e_info:
        create_new_image(input, 4, 4, 10, 4, 5, 5)

def test_to_big_pocket_both():
    input = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as e_info:
        create_new_image(input, 4, 4, 4, 4, 5, 5)

def test_negative_args():
    input = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError) as e_info:
        create_new_image(input, -4, 4, 4, 4, 5, 5)
        create_new_image(input, 4, -4, 4, 4, 5, 5)
        create_new_image(input, 4, 4, -4, 4, 5, 5)
        create_new_image(input, 4, 4, 4, -4, 5, 5)
        create_new_image(input, 4, 4, 4, 4, -5, 5)
        create_new_image(input, 4, 4, 4, 4, 5, -5)

def test_not_dir(tmpdir):
    f=tmpdir.join('test.txt')
    f.write('test')
    with pytest.raises(NotADirectoryError) as e_info:
        split_images_in_directory('test.txt')

def test_is_dir(tmpdir):
    with pytest.raises(IsADirectoryError) as e_info:
        split_image(tmpdir)

def test_image_type(tmpdir):
    input = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    with pytest.raises(TypeError) as e_info:
        create_new_image(input)