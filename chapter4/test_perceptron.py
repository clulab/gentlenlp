import random
import argparse
import pprint
from pathlib import Path
import numpy as np
from vocabulary import Vocabulary
import imdb

def read_data(filename):
    with np.load(filename) as data:
        X = data['X']
        y = data['y']
    return X, y

def read_model(filename):
    with np.load(filename) as data:
        w = data['w']
        b = data['b']
    return w, b

def binary_classification_report(y_true, y_pred):
    # count true positives, false positives, true negatives, and false negatives
    tp = fp = tn = fn = 0
    for gold, pred in zip(y_true, y_pred):
        if pred == True:
            if gold == True:
                tp += 1
            else:
                fp += 1
        else:
            if gold == False:
                tn += 1
            else:
                fn += 1
    # calculate precision and recall
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    # calculate f1 score
    fscore = 2 * precision * recall / (precision + recall)
    # calculate accuracy
    accuracy  = (tp + tn) / len(y_true)
    # number of positive labels in y_true
    support   = sum(y_true)
    return {
        'precision': precision,
        'recall': recall,
        'f1-score': fscore,
        'support': support,
        'accuracy': accuracy,
    }

if __name__ == '__main__':

    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=Path, help='path to the data file')
    parser.add_argument('model_file', type=Path, help='path to model file')
    args = parser.parse_args()

    print('loading model ...')
    w, b = read_model(args.model_file)
    print('loading data ...')
    X, y_true = read_data(args.data_file)
    print('applying model ...')
    y_pred = np.where(X @ w + b > 0, 1, 0)
    print('classification report:')
    report = binary_classification_report(y_true, y_pred)
    pprint.pprint(report)