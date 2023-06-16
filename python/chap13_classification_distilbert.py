#!/usr/bin/env python
# coding: utf-8

# # Text Classification Using Transformer Networks (DistilBERT)

# Some initialization:

# In[1]:


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


# Read the train/dev/test datasets and create a HuggingFace `Dataset` object:

# In[2]:


def read_data(filename):
    # read csv file
    df = pd.read_csv(filename, header=None)
    # add column names
    df.columns = ['label', 'title', 'description']
    # make labels zero-based
    df['label'] -= 1
    # concatenate title and description, and remove backslashes
    df['text'] = df['title'] + " " + df['description']
    df['text'] = df['text'].str.replace('\\', ' ', regex=False)
    return df


# In[3]:


labels = open('data/ag_news_csv/classes.txt').read().splitlines()
train_df = read_data('data/ag_news_csv/train.csv')
test_df = read_data('data/ag_news_csv/test.csv')
train_df


# In[4]:


from sklearn.model_selection import train_test_split

train_df, eval_df = train_test_split(train_df, train_size=0.9)
train_df.reset_index(inplace=True, drop=True)
eval_df.reset_index(inplace=True, drop=True)

print(f'train rows: {len(train_df.index):,}')
print(f'eval rows: {len(eval_df.index):,}')
print(f'test rows: {len(test_df.index):,}')


# In[5]:


from datasets import Dataset, DatasetDict

ds = DatasetDict()
ds['train'] = Dataset.from_pandas(train_df)
ds['validation'] = Dataset.from_pandas(eval_df)
ds['test'] = Dataset.from_pandas(test_df)
ds


# Tokenize the texts:

# In[6]:


from transformers import AutoTokenizer

transformer_name = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)


# In[7]:


def tokenize(examples):
    return tokenizer(examples['text'], truncation=True)

train_ds = ds['train'].map(tokenize, batched=True, remove_columns=['title', 'description', 'text'])
eval_ds = ds['validation'].map(tokenize, batched=True, remove_columns=['title', 'description', 'text'])
train_ds.to_pandas()


# Create the transformer model:

# In[8]:


from transformers import AutoConfig

config = AutoConfig.from_pretrained(transformer_name, num_labels=len(labels))


# In[9]:


from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification

model = (
    DistilBertForSequenceClassification
    .from_pretrained(transformer_name, config=config)
)


# Create the trainer object and train:

# In[10]:


from transformers import TrainingArguments

num_epochs = 2
batch_size = 24
logging_steps = len(ds['train']) // batch_size
model_name = f'{transformer_name}-sequence-classification'
training_args = TrainingArguments(
    output_dir=model_name,
    log_level='error',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    weight_decay=0.01,
    disable_tqdm=False,
    logging_steps=logging_steps,
)


# In[11]:


from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    y_true = eval_pred.label_ids
    y_pred = np.argmax(eval_pred.predictions, axis=-1)
    return {'accuracy': accuracy_score(y_true, y_pred)}


# In[12]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)


# In[13]:


trainer.train()


# Evaluate on the test partition:

# In[14]:


test_ds = ds['test'].map(tokenize, batched=True, remove_columns=['title', 'description', 'text'])
test_ds.to_pandas()


# In[15]:


output = trainer.predict(test_ds)
output


# In[16]:


from sklearn.metrics import classification_report

y_true = output.label_ids
y_pred = np.argmax(output.predictions, axis=-1)
target_names = labels
print(classification_report(y_true, y_pred, target_names=target_names))


# In[ ]:




