import glob
import os
import argparse
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn import metrics

parser = argparse.ArgumentParser(description="Script to generate baseline metrics.")

np.set_printoptions(linewidth=np.inf)

parser.add_argument("-d", "--data-dir", type=str, required=True)
parser.add_argument("-o", "--output-file", type=str, default="./baseline_metrics.txt")

def calculate_baseline_metrics(data_dir, output_file):

    y_true = []

    labels = sorted(os.listdir(data_dir))
    file_paths = list(glob.glob(os.path.join(data_dir, "*", "*")))

    for file_index, file_path in enumerate(file_paths):
        class_name = file_path.split("/")[-2]
        y_true.append(labels.index(class_name))

    with open(output_file, "w") as f:

        x_data = np.ones(np.array(y_true).shape)

        for strategy in ["stratified", "most_frequent", "prior", "uniform"]:

            dummy = DummyClassifier(strategy=strategy)
            dummy.fit(x_data, y_true)

            y_pred = dummy.predict(x_data)

            f.write("--- %s Macro Averaged Resutls ---\n" % strategy)
            f.write("Precision: %s\n" % metrics.precision_score(y_true, y_pred, average="macro"))
            f.write("Recall: %s\n" % metrics.recall_score(y_true, y_pred, average="macro"))
            f.write("F1-Score: %s\n\n" % metrics.f1_score(y_true, y_pred, average="macro"))

            f.write("--- %s Micro Averaged Resutls ---\n" % strategy)
            f.write("Precision: %s\n" % metrics.precision_score(y_true, y_pred, average="micro"))
            f.write("Recall: %s\n" % metrics.recall_score(y_true, y_pred, average="micro"))
            f.write("F1-Score: %s\n\n" % metrics.f1_score(y_true, y_pred, average="micro"))

            f.write("--- %s Other Resutls ---\n" % strategy)
            f.write("MCC: %s\n" % metrics.matthews_corrcoef(y_true, y_pred))

            f.write("\n\n")
        
if __name__ == "__main__":

    args = parser.parse_args()
    
    data_dir = args.data_dir
    output_file = args.output_file

    calculate_baseline_metrics(data_dir, output_file)