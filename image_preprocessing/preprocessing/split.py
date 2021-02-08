import os
import sys
from typing import Tuple

import numpy as np
from PIL import Image
from progress.bar import ChargingBar as cb


def split_images_in_directory(directory_name: str,
                    output_shape: Tuple[int, int] = (200, 200),
                    pocket: Tuple[int, int] = (100, 100),
                    extension: str = '.png'):
    if not os.path.isdir(directory_name):
        raise NotADirectoryError()
    files = os.listdir(directory_name)
    with cb('Splitting', max=len(files)) as bar:
        for f in files:
            file_path = os.path.join(directory_name, f)
            if not os.path.isdir(file_path):
                dir_name = f'out_{f.split(".")[-2]}'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                split_image(file_path,dir_name,
                    output_shape, pocket, extension)
            bar.next()


def split_image(filename: str, output_directory: str = 'out',
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
        image_width/(horizontal_step-horizontal_pocket)) - 1
    number_of_moves_vertically = int(
        image_height/(vertical_step-vertical_pocket)) - 1
    for row in range(number_of_moves_vertically):
        for column in range(number_of_moves_horizontally):
            tmp_img = create_new_image(image,
                                       column,
                                       row,
                                       horizontal_step,
                                       vertical_step,
                                       horizontal_pocket,
                                       vertical_pocket)
            new_filename = os.path.join(
                output_directory, f'{row}_{column}{extension}')
            tmp_img.save(new_filename)
            print(new_filename)


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
    if len(sys.argv) != 2:
        print('Usage: python split.py input_directory/input_file')
        exit(1)

    if os.path.isdir(sys.argv[1]):
        print('Processing directory')
        split_images_in_directory(sys.argv[1])
    else:
        print('Processing file')
        split_image(sys.argv[1])
