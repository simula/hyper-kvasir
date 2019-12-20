import glob
import shutil
import os
import argparse

from PIL import Image

parser = argparse.ArgumentParser(description="Generate a annotation file from a class file strucutre.")

parser.add_argument("-d", "--data-dir", type=str)
parser.add_argument("-o", "--output-file", type=str, default="hyper-kvasir-image-annotations-file.csv")

def gather_images(data_dir, output_file):

    with open(output_file, "w") as f:

        file_paths = sorted(list(glob.glob("%s/*/*" % data_dir)), key=lambda x: x.split("/")[-2])

        f.write("file-name;class-name;width;height;kilobytes\n")

        for filepath in file_paths:
            class_name = filepath.split("/")[-2]
            file_name = os.path.basename(filepath)

            image = Image.open(filepath)

            kilobytes = os.path.getsize(filepath) >> 10

            image_width, image_height = image.size          

            f.write("%s;%s;%s;%s;%s\n" % (file_name, class_name, image_width, image_height, kilobytes))

if __name__ == "__main__":

    args = parser.parse_args()

    data_dir = args.data_dir
    output_file = args.output_file

    gather_images(data_dir, output_file)