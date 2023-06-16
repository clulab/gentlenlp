#!/usr/bin/env python
# coding: utf-8

# # Part-of-speech Tagging with Transformer Networks

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


# Read the words and POS tags from the Spanish dataset:

# In[2]:


from conllu import parse_incr

def read_tags(filename):
    data = {'words': [], 'tags': []}
    with open(filename) as f:
        for sent in tqdm(parse_incr(f)):
            words = [tok['form'] for tok in sent]
            tags = [tok['upos'] for tok in sent]
            data['words'].append(words)
            data['tags'].append(tags)
    return pd.DataFrame(data)


# In[3]:


train_df = read_tags('data/UD_Spanish-AnCora/es_ancora-ud-train.conllup')
valid_df = read_tags('data/UD_Spanish-AnCora/es_ancora-ud-dev.conllup')
test_df = read_tags('data/UD_Spanish-AnCora/es_ancora-ud-test.conllup')


# In[4]:


tags = train_df['tags'].explode().unique()
index_to_tag = {i:t for i,t in enumerate(tags)}
tag_to_index = {t:i for i,t in enumerate(tags)}


# Create a HuggingFace `DatasetDict` object:

# In[5]:


from datasets import Dataset, DatasetDict

ds = DatasetDict()
ds['train'] = Dataset.from_pandas(train_df)
ds['validation'] = Dataset.from_pandas(valid_df)
ds['test'] = Dataset.from_pandas(test_df)
ds


# In[6]:


ds['train'].to_pandas()


# Now tokenize the texts and assign POS labels to the first token in each word:

# In[7]:


from transformers import AutoTokenizer

transformer_name = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(transformer_name)


# In[8]:


x = ds['train'][0]
tokenized_input = tokenizer(x['words'], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
word_ids = tokenized_input.word_ids()
pd.DataFrame([tokens, word_ids], index=['tokens', 'word ids'])


# In[9]:


# https://arxiv.org/pdf/1810.04805.pdf
# Section 5.3
# We use the representation of the first sub-token as the input to the token-level classifier over the NER label set.

# default value for CrossEntropyLoss ignore_index parameter
ignore_index = -100

def tokenize_and_align_labels(batch):
    labels = []
    # tokenize batch
    tokenized_inputs = tokenizer(
        batch['words'],
        truncation=True,
        is_split_into_words=True,
    )
    # iterate over batch elements
    for i, tags in enumerate(batch['tags']):
        label_ids = []
        previous_word_id = None
        # get word ids for current batch element
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        # iterate over tokens in batch element
        for word_id in word_ids:
            if word_id is None or word_id == previous_word_id:
                # ignore if not a word or word id has already been seen
                label_ids.append(ignore_index)
            else:
                # get tag id for corresponding word
                tag_id = tag_to_index[tags[word_id]]
                label_ids.append(tag_id)
            # remember this word id
            previous_word_id = word_id
        # save label ids for current batch element
        labels.append(label_ids)
    # store labels together with the tokenizer output
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


# In[10]:


train_ds = ds['train'].map(tokenize_and_align_labels, batched=True)
eval_ds = ds['validation'].map(tokenize_and_align_labels, batched=True)
train_ds.to_pandas()


# Create our transformer model:

# In[11]:


from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

# https://github.com/huggingface/transformers/blob/65659a29cf5a079842e61a63d57fa24474288998/src/transformers/models/roberta/modeling_roberta.py#L1346

class XLMRobertaForTokenClassification(RobertaPreTrainedModel):    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            inputs = logits.view(-1, self.num_labels)
            targets = labels.view(-1)
            loss = loss_fn(inputs, targets)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# In[12]:


from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    transformer_name,
    num_labels=len(index_to_tag),
)

model = (
    XLMRobertaForTokenClassification
    .from_pretrained(transformer_name, config=config)
)


# Create the `Trainer` object and train:

# In[13]:


from transformers import TrainingArguments

num_epochs = 2
batch_size = 24
weight_decay = 0.01
model_name = f'{transformer_name}-finetuned-pos-es'

training_args = TrainingArguments(
    output_dir=model_name,
    log_level='error',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy='epoch',
    weight_decay=weight_decay,
)


# In[14]:


from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    # gold labels
    label_ids = eval_pred.label_ids
    # predictions
    pred_ids = np.argmax(eval_pred.predictions, axis=-1)
    # collect gold and predicted labels, ignoring ignore_index label
    y_true, y_pred = [], []
    batch_size, seq_len = pred_ids.shape
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != ignore_index:
                y_true.append(index_to_tag[label_ids[i][j]])
                y_pred.append(index_to_tag[pred_ids[i][j]])
    # return computed metrics
    return {'accuracy': accuracy_score(y_true, y_pred)}


# In[15]:


from transformers import Trainer
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)

trainer.train()


# Evaluate on the test partition:

# In[16]:


test_ds = ds['test'].map(
    tokenize_and_align_labels,
    batched=True,
)
output = trainer.predict(test_ds)


# In[17]:


from sklearn.metrics import classification_report

num_labels = model.num_labels
label_ids = output.label_ids.reshape(-1)
predictions = output.predictions.reshape(-1, num_labels)
predictions = np.argmax(predictions, axis=-1)
mask = label_ids != ignore_index

y_true = label_ids[mask]
y_pred = predictions[mask]
target_names = tags[:-1]

report = classification_report(
    y_true, y_pred,
    target_names=target_names
)
print(report)


# In[18]:


import matplotlib.pyplot as plt
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

