import os
import sys
from imagessplitter import ImageSplitter
from imagesmerger import ImageMerger
from progress.bar import ChargingBar as cb

def make_sure_directory_exists(dir_names):
    for dir_name in dir_names:
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage python imagesmanipulator.py orginal_images_directory spliced_images_directory output_directory')
        sys.exit(1)

    src_dir = sys.argv[1]
    if not os.path.isdir(src_dir):
        print('Source directory does not exist')
        sys.exit(1)

    tmp_dir = sys.argv[2]
    out_dir = sys.argv[3]
    make_sure_directory_exists([tmp_dir,out_dir])

    splitter = ImageSplitter(src_dir,tmp_dir)
    images = os.listdir(src_dir)
    images.sort()
    with cb("Splitting", max=len(images)) as bar:
        for img in images:
            splitter.split_into_images(img)
            bar.next()

    merger = ImageMerger(tmp_dir,out_dir)
    with cb("Mergeing", max=len(images)) as bar:
        for i in range(len(images)):
            merger.merge(str(i))
            bar.next()
