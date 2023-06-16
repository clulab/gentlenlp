#!/usr/bin/env python
# coding: utf-8

# # Binary Text Classification with 
# # Logistic Regression Implemented with PyTorch and BCE Loss

# In[1]:


import random
import numpy as np
import torch
from tqdm.notebook import tqdm

# set this variable to a number to be used as the random seed
# or to None if you don't want to set a random seed
seed = 1234

if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# The dataset is divided in two directories called `train` and `test`.
# These directories contain the training and testing splits of the dataset.

# In[2]:


get_ipython().system('ls -lh data/aclImdb/')


# Both the `train` and `test` directories contain two directories called `pos` and `neg` that contain text files with the positive and negative reviews, respectively.

# In[3]:


get_ipython().system('ls -lh data/aclImdb/train/')


# We will now read the filenames of the positive and negative examples.

# In[4]:


from glob import glob

pos_files = glob('data/aclImdb/train/pos/*.txt')
neg_files = glob('data/aclImdb/train/neg/*.txt')

print('number of positive reviews:', len(pos_files))
print('number of negative reviews:', len(neg_files))


# Now, we will use a [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to read the text files, tokenize them, acquire a vocabulary from the training data, and encode it in a document-term matrix in which each row represents a review, and each column represents a term in the vocabulary. Each element $(i,j)$ in the matrix represents the number of times term $j$ appears in example $i$.

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer

# initialize CountVectorizer indicating that we will give it a list of filenames that have to be read
cv = CountVectorizer(input='filename')

# learn vocabulary and return sparse document-term matrix
doc_term_matrix = cv.fit_transform(pos_files + neg_files)
doc_term_matrix


# Note in the message printed above that the matrix is of shape (25000, 74894).
# In other words, it has 1,871,225,000 elements.
# However, only 3,445,861 elements were stored.
# This is because most of the elements in the matrix are zeros.
# The reason is that the reviews are short and most words in the english language don't appear in each review.
# A matrix that only stores non-zero values is called *sparse*.
# 
# Now we will convert it to a dense numpy array:

# In[6]:


X_train = doc_term_matrix.toarray()
X_train.shape


# We will also create a numpy array with the binary labels for the reviews.
# One indicates a positive review and zero a negative review.
# The label `y_train[i]` corresponds to the review encoded in row `i` of the `X_train` matrix.

# In[7]:


# training labels
y_pos = np.ones(len(pos_files))
y_neg = np.zeros(len(neg_files))
y_train = np.concatenate([y_pos, y_neg])
y_train


# Now we will initialize our model, in the form of an array of weights `w` of the same size as the number of features in our dataset (i.e., the number of words in the vocabulary acquired by [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)), and a bias term `b`.
# Both are initialized to zeros.

# In[8]:


n_examples, n_features = X_train.shape


# Now we will use the logistic regression learning algorithm to learn the values of `w` and `b` from our training data.

# In[9]:


import torch
from torch import nn
from torch import optim

lr = 1e-1
n_epochs = 10

model = nn.Linear(n_features, 1)
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

indices = np.arange(n_examples)
for epoch in range(10):
    n_errors = 0
    # randomize training examples
    np.random.shuffle(indices)
    # for each training example
    for i in tqdm(indices, desc=f'epoch {epoch+1}'):
        x = X_train[i]
        y_true = y_train[i]
        # make predictions
        y_pred = model(x)
        # calculate loss
        loss = loss_func(y_pred[0], y_true)
        # calculate gradients through back-propagation
        loss.backward()
        # optimize model parameters
        optimizer.step()
        # ensure gradients are set to zero
        model.zero_grad()


# The next step is evaluating the model on the test dataset.
# Note that this time we use the [`transform()`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.transform) method of the [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), instead of the [`fit_transform()`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer.fit_transform) method that we used above. This is because we want to use the learned vocabulary in the test set, instead of learning a new one.

# In[10]:


pos_files = glob('data/aclImdb/test/pos/*.txt')
neg_files = glob('data/aclImdb/test/neg/*.txt')
doc_term_matrix = cv.transform(pos_files + neg_files)
X_test = doc_term_matrix.toarray()
X_test = torch.tensor(X_test, dtype=torch.float32)
y_pos = np.ones(len(pos_files))
y_neg = np.zeros(len(neg_files))
y_test = np.concatenate([y_pos, y_neg])


# Using the model is easy: multiply the document-term matrix by the learned weights and add the bias.
# We use Python's `@` operator to perform the matrix-vector multiplication.

# In[11]:


y_pred = model(X_test) > 0


# Now we print an evaluation of the prediction results using scikit-learn's [`classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) function.

# In[12]:


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


# In[13]:


binary_classification_report(y_test, y_pred)

