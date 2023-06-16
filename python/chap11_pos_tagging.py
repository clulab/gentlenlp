#!/usr/bin/env python
# coding: utf-8

# # Part-of-speech Tagging Using RNNs

# Some initialization:

# In[4]:


import random
import torch
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

# enable tqdm in pandas
tqdm.pandas()

# set to True to use the gpu (if there is one available)
use_gpu = True

# select device
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
print(f'device: {device.type}')

# random seed
seed = 1234

# set random seed
if seed is not None:
    print(f'random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Next, let's read the words and their POS tags from the CoNLLUP format:

# In[5]:


from conllu import parse_incr

def read_tags(filename):
    data = {'words': [], 'tags': []}
    with open(filename) as f:
        for sent in parse_incr(f):
            words = [tok['form'] for tok in sent]
            tags = [tok['upos'] for tok in sent]
            data['words'].append(words)
            data['tags'].append(tags)
    return pd.DataFrame(data)


# In[6]:


train_df = read_tags('data/UD_Spanish-AnCora/es_ancora-ud-train.conllup')
train_df


# We now load the GloVe embeddings for Spanish, which include a representation for the unknown token:

# In[7]:


from gensim.models import KeyedVectors
glove = KeyedVectors.load_word2vec_format('glove-sbwc.i25.vec')
glove.vectors.shape


# In[8]:


# these embeddings already include <unk>
unk_tok = '<unk>'
unk_id = glove.key_to_index[unk_tok]
unk_tok, unk_id


# In[9]:


# add padding embedding
pad_tok = '<pad>'
pad_emb = np.zeros(300)
glove.add_vector(pad_tok, pad_emb)
pad_tok_id = glove.key_to_index[pad_tok]
pad_tok, pad_tok_id


# Preprocessing: lower case all words, and replace all numbers with '0':

# In[10]:


def preprocess(words):
    result = []
    for w in words:
        w = w.lower()
        if w.isdecimal():
            w = '0'
        result.append(w)
    return result

train_df['words'] = train_df['words'].progress_map(preprocess)
train_df


# Next, construct actual PyTorch `Dataset` and `DataLoader` objects for the train/dev/test partitions:

# In[11]:


def get_ids(tokens, key_to_index, unk_id=None):
    return [key_to_index.get(tok, unk_id) for tok in tokens]

def get_word_ids(tokens):
    return get_ids(tokens, glove.key_to_index, unk_id)

# add new column to the dataframe
train_df['word ids'] = train_df['words'].progress_map(get_word_ids)
train_df


# In[12]:


pad_tag = '<pad>'
index_to_tag = train_df['tags'].explode().unique().tolist() + [pad_tag]
tag_to_index = {t:i for i,t in enumerate(index_to_tag)}
pad_tag_id = tag_to_index[pad_tag]
pad_tag, pad_tag_id


# In[13]:


index_to_tag


# In[14]:


def get_tag_ids(tags):
    return get_ids(tags, tag_to_index)

train_df['tag ids'] = train_df['tags'].progress_map(get_tag_ids)
train_df


# In[15]:


dev_df = read_tags('data/UD_Spanish-AnCora/es_ancora-ud-dev.conllup')
dev_df['words'] = dev_df['words'].progress_map(preprocess)
dev_df['word ids'] = dev_df['words'].progress_map(lambda x: get_ids(x, glove.key_to_index, unk_id))
dev_df['tag ids'] = dev_df['tags'].progress_map(lambda x: get_ids(x, tag_to_index))
dev_df


# In[16]:


from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        x = torch.tensor(self.x[index])
        y = torch.tensor(self.y[index])
        return x, y


# `collate_fn` will be used by `DataLoader` to pad all sentences in the same batch to the same length.

# In[17]:


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # separate xs and ys
    xs, ys = zip(*batch)
    # get lengths
    lengths = [len(x) for x in xs]
    # pad sequences
    x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_tok_id)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=pad_tag_id)
    # return padded
    return x_padded, y_padded, lengths


# Now construct our PyTorch model:

# In[18]:


from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MyModel(nn.Module):
    def __init__(self, vectors, hidden_size, num_layers, bidirectional, dropout, output_size):
        super().__init__()
        # ensure vectors is a tensor
        if not torch.is_tensor(vectors):
            vectors = torch.tensor(vectors)
        # init embedding layer
        self.embedding = nn.Embedding.from_pretrained(embeddings=vectors)
        # init lstm
        self.lstm = nn.LSTM(
            input_size=vectors.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)
        # init dropout
        self.dropout = nn.Dropout(dropout)
        # init classifier
        self.classifier = nn.Linear(
            in_features=hidden_size * 2 if bidirectional else hidden_size,
            out_features=output_size)
        
    def forward(self, x_padded, x_lengths):
        # get embeddings
        output = self.embedding(x_padded)
        output = self.dropout(output)
        # pack data before lstm
        packed = pack_padded_sequence(output, x_lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)
        # unpack data before rest of model
        output, _ = pad_packed_sequence(packed, batch_first=True)
        output = self.dropout(output)
        output = self.classifier(output)
        return output


# In[19]:


from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# hyperparameters
lr = 1e-3
weight_decay = 1e-5
batch_size = 100
shuffle = True
n_epochs = 10
vectors = glove.vectors
hidden_size = 100
num_layers = 2
bidirectional = True
dropout = 0.1
output_size = len(index_to_tag)

# initialize the model, loss function, optimizer, and data-loader
model = MyModel(vectors, hidden_size, num_layers, bidirectional, dropout, output_size).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
train_ds = MyDataset(train_df['word ids'], train_df['tag ids'])
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
dev_ds = MyDataset(dev_df['word ids'], dev_df['tag ids'])
dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

train_loss, train_acc = [], []
dev_loss, dev_acc = [], []


# We are now ready to train!

# In[20]:


# train the model
for epoch in range(n_epochs):
    losses, acc = [], []
    model.train()
    for x_padded, y_padded, lengths in tqdm(train_dl, desc=f'epoch {epoch+1} (train)'):
        # clear gradients
        model.zero_grad()
        # send batch to right device
        x_padded = x_padded.to(device)
        y_padded = y_padded.to(device)
        # predict label scores
        y_pred = model(x_padded, lengths)
        # reshape output
        y_true = torch.flatten(y_padded)
        y_pred = y_pred.view(-1, output_size)
        mask = y_true != pad_tag_id
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        # compute loss
        loss = loss_func(y_pred, y_true)
        # accumulate for plotting
        gold = y_true.detach().cpu().numpy()
        pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        losses.append(loss.detach().cpu().item())
        acc.append(accuracy_score(gold, pred))
        # backpropagate
        loss.backward()
        # optimize model parameters
        optimizer.step()
    train_loss.append(np.mean(losses))
    train_acc.append(np.mean(acc))
    
    model.eval()
    with torch.no_grad():
        losses, acc = [], []
        for x_padded, y_padded, lengths in tqdm(dev_dl, desc=f'epoch {epoch+1} (dev)'):
            x_padded = x_padded.to(device)
            y_padded = y_padded.to(device)
            y_pred = model(x_padded, lengths)
            y_true = torch.flatten(y_padded)
            y_pred = y_pred.view(-1, output_size)
            mask = y_true != pad_tag_id
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            loss = loss_func(y_pred, y_true)
            gold = y_true.cpu().numpy()
            pred = np.argmax(y_pred.cpu().numpy(), axis=1)
            losses.append(loss.cpu().item())
            acc.append(accuracy_score(gold, pred))
        dev_loss.append(np.mean(losses))
        dev_acc.append(np.mean(acc))


# Plot loss and accuracy on dev after each epoch:

# In[21]:


import matplotlib.pyplot as plt

x = np.arange(n_epochs) + 1

plt.plot(x, train_loss)
plt.plot(x, dev_loss)
plt.legend(['train', 'dev'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid(True)


# In[22]:


plt.plot(x, train_acc)
plt.plot(x, dev_acc)
plt.legend(['train', 'dev'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.grid(True)


# In[23]:


test_df = read_tags('data/UD_Spanish-AnCora/es_ancora-ud-test.conllup')
test_df['words'] = test_df['words'].progress_map(preprocess)
test_df['word ids'] = test_df['words'].progress_map(lambda x: get_ids(x, glove.key_to_index, unk_id))
test_df['tag ids'] = test_df['tags'].progress_map(lambda x: get_ids(x, tag_to_index))
test_df


# Now let's evaluate on the test partition:

# In[24]:


from sklearn.metrics import classification_report

model.eval()

test_ds = MyDataset(test_df['word ids'], test_df['tag ids'])
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

all_y_true = []
all_y_pred = []

with torch.no_grad():
    for x_padded, y_padded, lengths in tqdm(test_dl):
        x_padded = x_padded.to(device)
        y_pred = model(x_padded, lengths)
        y_true = torch.flatten(y_padded)
        y_pred = y_pred.view(-1, output_size)
        mask = y_true != pad_tag_id
        y_true = y_true[mask]
        y_pred = torch.argmax(y_pred[mask], dim=1)
        all_y_true.append(y_true.cpu().numpy())
        all_y_pred.append(y_pred.cpu().numpy())

y_true = np.concatenate(all_y_true)
y_pred = np.concatenate(all_y_pred)
target_names = index_to_tag[:-2]
print(classification_report(y_true, y_pred, target_names=target_names))


# Let's generate a confusion matrix for all POS tags in the data:

# In[25]:


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=target_names,
)

fig, ax = plt.subplots(figsize=(10,10))
disp.plot(
    cmap='Blues',
    values_format='.2f',
    colorbar=False,
    ax=ax,
    xticks_rotation=45,
)


# In[ ]:




