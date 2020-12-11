import os
import sys
from progress.bar import ChargingBar as cb
from PIL import Image, ImageFilter, UnidentifiedImageError

def blur_file(filepath):
    try:
        im = Image.open(filepath)
        im = im.filter(ImageFilter.GaussianBlur(2))
        im.save(filepath)
    except UnidentifiedImageError:
        print(f'\nError processing file: {filepath}!\n')

def blur_dir(dir):
    files = os.listdir(dir)
    with cb('Bluring', max=len(files)) as bar:
        for file in files:
            blur_file(os.path.join(dir,file))
            bar.next()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python imagesblur.py directory/file')
        exit(1)

    if os.path.isdir(sys.argv[1]):
        print('Processing directory')
        blur_dir(sys.argv[1])
    else:
        print('Processing file')
        blur_file(sys.argv[1])
