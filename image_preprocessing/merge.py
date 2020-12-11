import os
from typing import Tuple

import numpy as np
from PIL import Image


def parse_name(filename: str) -> Tuple[int, int]:
    if os.path.isdir(filename):
        raise IsADirectoryError()
    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0]
    filenames = filename.split('_')

    if len(filename) < 3:
        raise IndexError('Filename must have format of *_row_column.extension')

    row = int(filenames[-2])
    column = int(filenames[-1])
    return (row, column)


def merge_files(src_directory: str,
                output_shape: Tuple[int, int, int] = (1200, 1600, 3),
                input_shape: Tuple[int, int] = (200, 200),
                pocket: Tuple[int, int] = (200, 200),
                output_filename: str = 'out.png'):

    container = np.zeros(output_shape)
    images = os.listdir(src_directory)
    vertical_step, horizontal_step = input_shape
    vertical_pocket, horizontal_pocket = pocket

    for image in images:
        row, column = parse_name(image)
        start_row = row * (vertical_step - vertical_pocket)
        start_column = column * (horizontal_step - horizontal_pocket)
        end_row = start_row + vertical_step
        end_column = start_column + horizontal_step
        container[start_row:end_row, start_column:end_column] = np.array(
            Image.open(os.path.join(src_directory, image)))

    container = container.astype('uint8')
    img = Image.fromarray(container)
    img.save(output_filename)
