import os
import sys
import numpy as np
from progress.bar import ChargingBar as cb
from PIL import Image, UnidentifiedImageError


def remove_alpha_dir(dir, threshold=125):
    files = os.listdir(dir)
    with cb('Removeing alpha', max=len(files)) as bar:
        for file in files:
            remove_alpha_file(os.path.join(dir, file), threshold)
            bar.next()


def remove_alpha_file(filepath, threshold):
    try:
        img = Image.open(filepath)
        r, g, b, a = img.split()
        r = np.array(r)
        a = np.array(a)
        r[a >= threshold] = 255
        r[a < threshold] = 0
        r = Image.fromarray(r)
        img = Image.merge('RGB', (r, g, b))
        img.save(filepath)
    except UnidentifiedImageError:
        print(f'\nError processing file: {filepath}!\n')


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python remove_alpha.py directory/file [threshold]')
        exit(1)

    if os.path.isdir(sys.argv[1]):
        print('Processing directory')
        remove_alpha_dir(sys.argv[1], sys.argv[2]
                         if len(sys.argv) == 3 else 125)
    else:
        print('Processing file')
        remove_alpha_file(sys.argv[1], sys.argv[2]
                          if len(sys.argv) == 3 else 125)
