import os
import sys
from typing import Tuple

import numpy as np
from PIL import Image
from progress.bar import ChargingBar as cb


def split_directory(directory_name: str,
                    output_shape: Tuple[int, int] = (200, 200),
                    pocket: Tuple[int, int] = (100, 100),
                    extension: str = '.png'):
    if not os.path.isdir(directory_name):
        raise NotADirectoryError()
    files = os.listdir(directory_name)
    with cb('Splitting', max=len(files)) as bar:
        for file in files:
            file_path = os.path.join(directory_name, file)
            if not os.path.isdir(file_path):
                split(file_path, f'out_{file}',
                    output_shape, pocket, extension)
            bar.next()


def split(filename: str, output_directory: str = 'out',
          output_shape: Tuple[int, int] = (200, 200),
          pocket: Tuple[int, int] = (100, 100),
          extension: str = '.png'):

    if os.path.isdir(filename):
            raise IsADirectoryError()

    image = np.array(Image.open(filename))
    image_width, image_height = image.shape[:2]
    vertical_step, horizontal_step = output_shape
    vertical_pocket, horizontal_pocket = pocket
    extension = extension if extension.startswith(".") else f".{extension}"

    number_of_moves_horizontally = int(
        image_width/(image_width-horizontal_pocket)) - 1
    number_of_moves_vertically = int(
        image_height/(image_height-vertical_pocket)) - 1
    for row in range(number_of_moves_vertically):
        for column in range(number_of_moves_horizontally):
            tmp_img = create_new_image(image,
                                       column,
                                       row,
                                       image_width,
                                       image_height,
                                       horizontal_pocket,
                                       vertical_pocket,
                                       horizontal_step,
                                       vertical_step)
            new_filename = os.path.join(
                output_directory, f'{row}_{column}{extension}')
            tmp_img.save(new_filename)


def create_new_image(image: np.ndarray,
                     column: int,
                     row: int,
                     image_width: int,
                     image_height: int,
                     horizontal_pocket: int,
                     vertical_pocket: int) -> np.ndarray:
    if horizontal_pocket >= image_width or vertical_pocket >= image_height:
        raise ValueError('Pocket must be smaller then image size')
    if column < 0 or row < 0 or image_width <= 0 or image_height <= 0 or horizontal_pocket < 0 or vertical_pocket < 0:
        raise ValueError('Must be positive numbers')
    if type(image) != np.ndarray:
        raise TypeError()
    horizontal_start = column*(image_width - horizontal_pocket)
    vertical_start = row*(image_height - vertical_pocket)
    return Image.fromarray(np.copy(
        image[vertical_start:vertical_start+image_height, horizontal_start:horizontal_start+image_width]))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print('Usage: python split.py input_directory/input_file')
        exit(1)

    if os.path.isdir(sys.argv[1]):
        print('Processing directory')
        split_directory(sys.argv[1])
    else:
        print('Processing file')
        split(sys.argv[1])
