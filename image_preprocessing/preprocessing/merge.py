import os
import sys
from typing import Tuple

import numpy as np
from PIL import Image
from progress.bar import ChargingBar as cb


def merge_directories(directories: list[str],
                      output_shape: Tuple[int, int, int] = (1200, 1600, 3),
                      input_shape: Tuple[int, int] = (200, 200),
                      pocket: Tuple[int, int] = (100, 100),
                      output_filename='out.png'):

    with cb('Merging', max=len(directories)) as bar:
        for directory in directories:
            merge_files(directory, output_shape, input_shape,
                        pocket, f'{directory}_{output_filename}')
            bar.next()


def merge_files(src_directory: str,
                output_shape: Tuple[int, int, int] = (1200, 1600, 3),
                input_shape: Tuple[int, int] = (200, 200),
                pocket: Tuple[int, int] = (100, 100),
                output_filename: str = 'out.png'):

    container = np.zeros(output_shape)
    images = os.listdir(src_directory)
    vertical_step, horizontal_step = input_shape
    vertical_pocket, horizontal_pocket = pocket

    with cb(f'Merging {src_directory}: ', max=len(images)) as bar:
        for image in images:
            row, column = parse_name(image)
            start_row = row * (vertical_step - vertical_pocket)
            start_column = column * (horizontal_step - horizontal_pocket)
            end_row = start_row + vertical_step
            end_column = start_column + horizontal_step
            container[start_row:end_row, start_column:end_column] = np.array(
                Image.open(os.path.join(src_directory, image)))
            bar.next()

    container = container.astype('uint8')
    img = Image.fromarray(container)
    img.save(output_filename)


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


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print('Usage: python merge.py input_directory [input_directory...]')
        exit(1)

    for dir in sys.argv[1:]:
        if not os.path.isdir(dir):
            print(
                'Usage: python merge.py input_directory [input_directory...]')
            exit(1)
    else:
        merge_directories(*sys.argv[1:])
