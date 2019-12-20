import glob
import shutil
import csv
import os
import argparse

parser = argparse.ArgumentParser(description="Split images into folders based on given annotation file.")

parser.add_argument("-s", "--src-dir", type=str)
parser.add_argument("-d", "--dest-dir", type=str)
parser.add_argument("-a", "--annotation-path", type=str)

def split_images(src_dir, dest_dir, annotation_path):

    with open(annotation_path) as f:

        rows = csv.reader(f, delimiter=";")

        for row in rows:

            file_name  = row[0]
            class_name = row[1]
            
            class_path = os.path.join(dest_dir, class_name)

            if not os.path.exists(class_path):
                os.makedirs(class_path)

            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(class_path, file_name)

            if not os.path.exists(src_path):
                continue

            shutil.copy(src_path, dest_path)

if __name__ == "__main__":

    args = parser.parse_args()

    src_dir = args.src_dir
    dest_dir = args.dest_dir
    annotation_path = args.annotation_path

    split_images(src_dir, dest_dir, annotation_path)