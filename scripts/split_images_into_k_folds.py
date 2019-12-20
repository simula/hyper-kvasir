import glob
import shutil
import os
import argparse

parser = argparse.ArgumentParser(description="Split data into k folds.")

file_dir = os.path.dirname(os.path.realpath(__file__))

parser.add_argument("-s", "--src-dir", type=str)
parser.add_argument("-d", "--dest-dir", type=str, default=None)
parser.add_argument("-f", "--number-of-folds", type=int, default=2)
parser.add_argument("-e", "--exclude-classes", nargs='+')

def split_data_into_equal_parts(data, number_of_parts):
    part_length = len(data) // number_of_parts
    parts = []
    for index in range(number_of_parts - 1):
        parts += [ data[ index * part_length : (index + 1) * part_length ] ]
    parts += [ data[ (number_of_parts - 1) * part_length : len(data) ] ]
    return parts

def split_images(src_dir, number_of_folds, dest_dir=None, exclude_classes=[]):

    split_file = open("%s_fold_split.csv" % str(number_of_folds), "w")

    split_file.write("file-name;class-name;split-index\n")

    for class_path in glob.glob("%s/*" % src_dir):

        class_name = os.path.basename(class_path)
        file_paths = list(glob.glob("%s/*" % class_path))

        if class_name in exclude_classes:
            print("Skipping class %s" % class_name)
            continue

        for split_index, split in enumerate(split_data_into_equal_parts(file_paths, number_of_folds)):
            for file_path in split:

                file_name = os.path.basename(file_path)

                split_file.write("%s;%s;%s\n" % (file_name, class_name, str(split_index)))

                if dest_dir is not None:

                    dest_class_path = os.path.join(dest_dir, str(split_index), class_name)

                    if not os.path.exists(dest_class_path):
                        os.makedirs(dest_class_path)
                        
                    shutil.copy(file_path, os.path.join(dest_class_path, file_name))

if __name__ == "__main__":

    args = parser.parse_args()

    src_dir = args.src_dir
    dest_dir = args.dest_dir
    number_of_folds = args.number_of_folds
    exclude_classes = args.exclude_classes

    if not dest_dir is None and os.path.exists(dest_dir):
        raise Exception("%s already exists. Please delete it or choose another destination." % dest_dir)

    split_images(src_dir, number_of_folds, dest_dir, exclude_classes)