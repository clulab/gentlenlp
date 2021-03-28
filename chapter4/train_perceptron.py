import random
import argparse
from pathlib import Path
import numpy as np
from vocabulary import Vocabulary
import imdb

def read_npz(filename):
    with np.load(filename) as data:
        X = data['X']
        y = data['y']
    return X, y

if __name__ == '__main__':

    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=Path, help='path to the data file')
    parser.add_argument('model_file', type=Path, help='path to model file')
    args = parser.parse_args()

    print('loading data ...')
    X, y = read_npz(args.data_file)

    # init model
    n_examples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0
    n_epochs = 10

    print('training ...')
    indices = list(range(n_examples))
    for epoch in range(10):
        print('epoch', epoch+1)
        n_errors = 0
        random.shuffle(indices)
        for i in indices:
            x = X[i]
            y_true = y[i]
            score = x @ w + b
            y_pred = 1 if score > 0 else 0
            if y_true == y_pred:
                continue
            elif y_true == 1 and y_pred == 0:
                w = w + x
                b = b + 1
                n_errors += 1
            elif y_true == 0 and y_pred == 1:
                w = w - x
                b = b - 1
                n_errors += 1
        if n_errors == 0:
            break

    print('saving model ...')
    np.savez_compressed(args.model_file, w=w, b=b)
