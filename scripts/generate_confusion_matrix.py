import csv
import argparse
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics

parser = argparse.ArgumentParser(description="Generate confusion matrix based on given prediction file.")

np.set_printoptions(linewidth=np.inf)

parser.add_argument("-i", "--input-prediction-file", type=str, required=True)
parser.add_argument("-o", "--output-file", type=str, default="./confusion_matrix.png")

INDEX_TO_LETTER = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
    19: "T", 20: "U", 21: "V", 22: "W"
}

INDEX_TO_LABEL = {
    0: "barretts", 1: "bbps-0-1", 2: "bbps-2-3", 3: "dyed-lifted-polyps",
    4: "dyed-resection-margins", 5: "hemorroids", 6: "ileum", 7: "impacted-stool",
    8: "normal-cecum", 9: "normal-pylorus", 10: "normal-z-line", 11: "oesophagitis-a",
    12: "oesophagitis-b-d", 13: "polyp", 14: "retroflex-rectum", 15: "retroflex-stomach",
    16: "short-segment-barretts", 17: "ulcerative-colities-0-1", 18:"ulcerative-colities-1-2",
    19: "ulcerative-colities-2-3", 20: "ulcerative-colities-grade-1", 21: "ulcerative-colities-grade-2",
    22: "ulcerative-colities-grade-3""
}

LABEL_TO_LETTER = {
    "barretts": "A", "bbps-0-1": "B", "bbps-2-3": "C", "dyed-lifted-polyps": "D",
    "dyed-resection-margins": "E", "hemorroids": "F", "ileum": "G", "impacted-stool": "H",
    "normal-cecum": "I", "normal-pylorus": "J", "normal-z-line": "K", "oesophagitis-a": "L",
    "oesophagitis-b-d": "M", "polyp": "N", "retroflex-rectum": "O", "retroflex-stomach": "P",
    "short-segment-barretts": "Q", "ulcerative-colities-0-1": "R", "ulcerative-colities-1-2": "S",
    "ulcerative-colities-2-3": "T", "ulcerative-colities-grade-1": "U", "ulcerative-colities-grade-2":
    "V", "ulcerative-colities-grade-3": "W"
}

def read_prediction_file(file_path, index_to_label=None):

    y_true = []
    y_pred = []

    with open(file_path) as csv_file:

        reader = csv.reader(csv_file, delimiter=",")

        next(reader)

        for row in reader:

            y_pred_value = row[1]
            y_true_value = row[2]

            if not index_to_label is None:
                y_true_value = LABEL_TO_LETTER[y_true_value]
                y_pred_value = LABEL_TO_LETTER[y_pred_value]

            y_true.append(y_true_value)
            y_pred.append(y_pred_value)

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, filename, labels, ymap=None, figsize=(22, 17)):

    if ymap is not None:
        y_pred = [ ymap[ yi ] for yi in y_pred ]
        y_true = [ ymap[ yi ] for yi in y_true ]
        labels = [ ymap[ yi ] for yi in labels ]
        
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):
        for j in range(ncols):
            c = cm[ i, j ]
            p = cm_perc[ i, j ]
            if i == j:
                s = cm_sum[ i ]
                annot[ i, j ] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[ i, j ] = ""
            else:
                annot[ i, j ] = "%.1f%%\n%d" % (p, c)

    cm = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"

    plt.rc("axes", labelsize=18)
    plt.rc("xtick", labelsize=14) 
    plt.rc("ytick", labelsize=14)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=annot, fmt="", ax=ax, cmap="Purples")

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])

    ax.set_ylim(len(labels) + 0.5, -0.5)

    plt.savefig(filename)

if __name__ == "__main__":

    args = parser.parse_args()
    
    input_prediction_file = args.input_prediction_file
    output_file = args.output_file

    y_true, y_pred = read_prediction_file(input_prediction_file, INDEX_TO_LETTER)

    plot_confusion_matrix(y_true, y_pred, output_file, sorted(list(INDEX_TO_LETTER.values())))