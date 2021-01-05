import random
import torch
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import classification_report

def read_vocab(filename):
    with open(filename) as f:
        return f.read().splitlines()

def read_bows(filename, n_features=None, binary_labels=False):
    X, y = load_svmlight_file(filename, n_features=n_features)
    X = X.toarray()
    if binary_labels:
        y = np.where(y > 5, 1, 0)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    return X, y

if __name__ == '__main__':

    print('loading data ...')
    vocab = read_vocab('aclImdb/imdb.vocab')
    X, y = read_bows('aclImdb/train/labeledBow.feat', n_features=len(vocab), binary_labels=True)

    # init model
    n_examples, n_features = X.shape
    w = torch.zeros(n_features, dtype=float)
    b = 0
    n_epochs = 10

    print('training ...')
    indices = list(range(n_examples))
    for epoch in range(n_epochs):
        print('epoch', epoch)
        n_errors = 0
        random.shuffle(indices)
        for i in indices:
            x = X[i]
            y_true = y[i]
            y_pred = 1 if x @ w + b > 0 else 0
            update = y_true - y_pred
            n_errors += abs(update)
            w += update * x
            b += update
        if n_errors == 0:
            break

    print('performance on train')
    X, y_true = read_bows('aclImdb/train/labeledBow.feat', n_features=len(vocab), binary_labels=True)
    y_pred = torch.where(X @ w + b > 0, 1, 0)
    print(classification_report(y_true, y_pred))

    print('performance on test')
    X, y_true = read_bows('aclImdb/test/labeledBow.feat', n_features=len(vocab), binary_labels=True)
    y_pred = torch.where(X @ w + b > 0, 1, 0)
    print(classification_report(y_true, y_pred))