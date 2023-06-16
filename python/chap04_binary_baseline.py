#!/usr/bin/env python
# coding: utf-8

# # A Very Simple Text Classification Baseline
# 

# In[1]:


import random
import numpy as np

# set this variable to a number to be used as the random seed
# or to None if you don't want to set a random seed
seed = 1234

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)


# In[2]:


from glob import glob
from sklearn.feature_extraction.text import CountVectorizer

# load train dataset
pos_files = glob('data/aclImdb/train/pos/*.txt')
neg_files = glob('data/aclImdb/train/neg/*.txt')
cv = CountVectorizer(input='filename')
doc_term_matrix = cv.fit_transform(pos_files + neg_files)
X_train = doc_term_matrix.toarray()
X_train = np.column_stack((X_train, np.ones(X_train.shape[0])))
y_pos = np.ones(len(pos_files))
y_neg = np.zeros(len(neg_files))
y_train = np.concatenate([y_pos, y_neg])


# In[3]:


# load test dataset
pos_files = glob('data/aclImdb/test/pos/*.txt')
neg_files = glob('data/aclImdb/test/neg/*.txt')
doc_term_matrix = cv.transform(pos_files + neg_files)
X_test = doc_term_matrix.toarray()
X_test = np.column_stack((X_test, np.ones(X_test.shape[0])))
y_pos = np.ones(len(pos_files))
y_neg = np.zeros(len(neg_files))
y_test = np.concatenate([y_pos, y_neg])


# Below, we use `DummyClassifier` to generate predictions uniformly at random. 
# See [this page](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) for other possible configurations.

# In[4]:


from sklearn.dummy import DummyClassifier

# apply baseline
dummy = DummyClassifier(strategy='uniform')
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)


# In[5]:


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
    recall = tp / (tp + fn)
    # calculate f1 score
    fscore = 2 * precision * recall / (precision + recall)
    # calculate accuracy
    accuracy = (tp + tn) / len(y_true)
    # number of positive labels in y_true
    support = sum(y_true)
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": fscore,
        "support": support,
        "accuracy": accuracy,
    }


# In[6]:


binary_classification_report(y_test, y_pred)

