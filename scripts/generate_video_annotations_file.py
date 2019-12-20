import glob
import os
import argparse
import csv

import cv2

parser = argparse.ArgumentParser(description="Generate a video annotation file.")

parser.add_argument("-d", "--data-dir", type=str)
parser.add_argument("-o", "--video-annotations", type=str)
parser.add_argument("-o", "--output-file", type=str, default="hyper-kvasir-video-annotations-file.csv")

def gather_images(data_dir, video_annotations, output_file):

    annnotations = {}

    with open(video_annotations) as f:

        reader = csv.reader(f, delimiter=";")

        next(reader)

        for line in reader:
            
            file_name = "%s.avi" %  line[0]
            finding = line[1]

            annnotations[file_name] = finding

    with open(output_file, "w") as f:

        file_paths = sorted(list(glob.glob("%s/*" % data_dir)), key=lambda x: x.split("/")[-2])

        f.write("file-name;main-finding;width;height;number-of-frames;fps;length;kilobytes\n")

        for file_path in file_paths:

            file_name = os.path.basename(file_path)

            video = cv2.VideoCapture(file_path)
            
            number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            video_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if file_name in annnotations:
                finding = annnotations[file_name]
            else:
                finding = "None"

            fps = int(video.get(cv2.CAP_PROP_FPS))

            length = number_of_frames // fps

            kilobytes = os.path.getsize(file_path) >> 10

            f.write("%s;%s;%s;%s;%s;%s;%s;%s\n" % (file_name, finding, video_width, video_height, number_of_frames, fps, length, kilobytes))


if __name__ == "__main__":

    args = parser.parse_args()

    data_dir = args.data_dir
    video_annotations = args.video_annotations
    output_file = args.output_file

    gather_images(data_dir, video_annotations, output_file)