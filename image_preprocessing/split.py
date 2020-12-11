import os
import numpy as np
from PIL import Image

class ImageSplitter():
    it = 0

    def __init__(self,src_directory,out_directory):
        self._image_width = 1600
        self._image_height = 1200
        self._new_image_size = 200
        self._pocket = 100
        self._src_directory = src_directory
        self._out_directory = out_directory

    def split_into_images(self, image_name):
        img = np.array(Image.open(f'{self._src_directory}/{image_name}'))
        number_of_moves_horizontally = int(self._image_width/(self._new_image_size-self._pocket)) -1
        number_of_moves_vertically = int(self._image_height/(self._new_image_size-self._pocket)) - 1
        for j in range(number_of_moves_vertically):
            for i in range(number_of_moves_horizontally):
                horizontal_start = i*(self._new_image_size - self._pocket)
                vertical_start = j*(self._new_image_size - self._pocket)
                tmp_img = Image.fromarray(np.copy(img[vertical_start:vertical_start+self._new_image_size,horizontal_start:horizontal_start+self._new_image_size]))
                tmp_img.save(f"{self._out_directory}/{ImageSplitter.it}_{'0' if j < 10 else ''}{j}_{'0' if i < 10 else ''}{i}.png")
        ImageSplitter.it+=1

