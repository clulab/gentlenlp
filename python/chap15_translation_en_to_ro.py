#!/usr/bin/env python
# coding: utf-8

# # Machine Translation from English (En) to Romanian (Ro) 
# # Using the T5 Transformer without Fine-tuning

# Some initialization:

# In[1]:


import torch
import numpy as np
from transformers import set_seed

# set to True to use the gpu (if there is one available)
use_gpu = True

# select device
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
print(f'device: {device.type}')

# random seed
seed = 42

# set random seed
if seed is not None:
    print(f'random seed: {seed}')
    set_seed(seed)


# In[2]:


transformer_name = 't5-small'
source_lang = 'en'
target_lang = 'ro'
max_source_length = 1024
max_target_length = 128
task_prefix = 'translate English to Romanian: '
num_beams = 1
batch_size = 100


# Load tokenizer and pre-trained model:

# In[3]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(transformer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(transformer_name)
model = model.to(device)


# Load dataset from HuggingFace:

# In[4]:


from datasets import load_dataset

test_ds = load_dataset('wmt16', 'ro-en', split='test')
test_ds


# In[5]:


test_ds['translation'][0]


# Implement the `translate` method and apply on this dataset:

# In[6]:


def translate(batch):
    # get source language examples and prepend task prefix
    inputs = [x[source_lang] for x in batch["translation"]]
    inputs = [task_prefix + x for x in inputs]
    
    # tokenize inputs
    encoded = tokenizer(
        inputs,
        max_length=max_source_length,
        truncation=True,
        padding=True,
        return_tensors='pt',
    )
    
    # move data to gpu if needed
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    
    # generate translated sentences
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beams,
        max_length=max_target_length,
    )
    
    # generate predicted sentences from predicted token ids
    decoded = tokenizer.batch_decode(
        output,
        skip_special_tokens=True,
    )
    
    # get gold sentences in target language
    targets = [x[target_lang] for x in batch["translation"]]
    
    # return gold and predicted sentences
    return {
        'reference': targets,
        'prediction': decoded,
    }


# In[7]:


results = test_ds.map(
    translate,
    batched=True,
    batch_size=batch_size,
    remove_columns=test_ds.column_names,
)

results.to_pandas()


# Now evaluate the quality of translations using the BLEU metric:

# In[8]:


from datasets import load_metric

metric = load_metric('sacrebleu')

for r in results:
    prediction = r['prediction']
    reference = [r['reference']]
    metric.add(prediction=prediction, reference=reference)
    
metric.compute()


# An example of greedy decoding for individual texts:

# In[9]:


def greedy_translation(text):
    # prepend task prefix
    text = task_prefix + text
    
    # tokenize input
    encoded = tokenizer(
        text,
        max_length=max_source_length,
        truncation=True,
        return_tensors='pt',
    )
    
    # encoder input ids
    encoder_input_ids = encoded.input_ids.to(device)
    
    # decoder input ids, initialized with start token id
    start = model.config.decoder_start_token_id
    decoder_input_ids = torch.LongTensor([[start]]).to(device)
    
    # generate tokens, one at a time
    for _ in range(max_target_length):
        # get model predictions
        output = model(
            encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
        )
        # get logits for last token
        next_token_logits = output.logits[0, -1, :]
        # select most probable token
        next_token_id = torch.argmax(next_token_logits)
        # append new token to decoder_input_ids
        output_id = torch.LongTensor([[next_token_id]]).to(device)
        decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
        # if predicted token is the end of sequence, stop iterating
        if next_token_id == tokenizer.eos_token_id:
            break
            
    # return text corresponding to predicted token ids
    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)


# In[10]:


greedy_translation("this is a test")

