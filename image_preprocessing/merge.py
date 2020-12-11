import os
import numpy as np
from PIL import Image

class ImageMerger():
    def __init__(self, src_directory, out_directory):
        self._src_directory = src_directory
        self._image_names = os.listdir(src_directory)
        self._image_names.sort()
        self._out_directory = out_directory

    def merge(self, fileprefix):
        tmp = np.zeros((1200,1600,3))
        images = { val for val in self._image_names if val.startswith(fileprefix)}
        for img in images:
            split = img.split('_')
            row = int(split[1])
            start_row = 200*int(row) - 100*int(row)
            column = int(split[2].split('.')[0])
            start_column = 200*int(column) - 100*int(column)
            end_row = start_row+200
            end_column = start_column+200
            tmp[start_row:end_row,start_column:end_column] = np.array(Image.open(f'{self._src_directory}/{img}'))

        tmp = tmp.astype('uint8')
        img = Image.fromarray(tmp)
        img.save(f'{self._out_directory}/{fileprefix}.png')
