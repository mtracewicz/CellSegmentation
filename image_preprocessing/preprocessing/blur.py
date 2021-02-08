import os
import sys

from PIL import Image, ImageFilter, UnidentifiedImageError
from progress.bar import ChargingBar as cb


def blur_dir(dir, parameter=2):
    files = os.listdir(dir)
    with cb('Bluring', max=len(files)) as bar:
        for file in files:
            blur_file(os.path.join(dir, file), parameter)
            bar.next()


def blur_file(filepath, parameter=2):
    try:
        im = Image.open(filepath)
        im = im.filter(ImageFilter.GaussianBlur(parameter))
        im.save(filepath)
    except UnidentifiedImageError:
        print(f'\nError processing file: {filepath}!\n')


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python imagesblur.py directory/file [blur_parameter]')
        exit(1)

    if os.path.isdir(sys.argv[1]):
        print('Processing directory')
        blur_dir(sys.argv[1], sys.argv[2]
                 if len(sys.argv) == 3 else 2)
    else:
        print('Processing file')
        blur_file(sys.argv[1], sys.argv[2]
                  if len(sys.argv) == 3 else 2)
