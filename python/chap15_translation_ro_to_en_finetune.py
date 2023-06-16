#!/usr/bin/env python
# coding: utf-8

# # Machine Translation from Ro to En
# # Using the T5 Transformer with Fine-tuning

# Some initialization:

# In[1]:


import torch
import numpy as np
from transformers import set_seed

# random seed
seed = 42

# set random seed
if seed is not None:
    print(f'random seed: {seed}')
    set_seed(seed)


# In[2]:


transformer_name = 't5-small'
dataset_name = 'wmt16'
dataset_config_name = 'ro-en'
source_lang = 'ro'
target_lang = 'en'
max_source_length = 1024
max_target_length = 128
task_prefix = 'translate Romanian to English: '
batch_size = 4
label_pad_token_id = -100
save_steps = 25_000
num_beams = 1
learning_rate = 1e-3
num_train_epochs = 3
output_dir = '/media/data2/t5-translation-example' # make sure this is a valid path on your machine!


# Load dataset from HuggingFace:

# In[3]:


from datasets import load_dataset

wmt16 = load_dataset(dataset_name, dataset_config_name)


# Load tokenizer and pre-trained model:

# In[4]:


from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM

config = AutoConfig.from_pretrained(transformer_name)
tokenizer = AutoTokenizer.from_pretrained(transformer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(transformer_name, config=config)


# Tokenize the texts in the dataset:

# In[5]:


def tokenize(batch):
    # get source sentences and prepend task prefix
    sources = [x[source_lang] for x in batch["translation"]]
    sources = [task_prefix + x for x in sources]
    # tokenize source sentences
    output = tokenizer(
        sources,
        max_length=max_source_length,
        truncation=True,
    )

    # get target sentences
    targets = [x[target_lang] for x in batch["translation"]]
    # tokenize target sentences
    labels = tokenizer(
        targets,
        max_length=max_target_length,
        truncation=True,
    )
    # add targets to output
    output["labels"] = labels["input_ids"]

    return output


# In[6]:


train_dataset = wmt16['train']
eval_dataset = wmt16['validation']
column_names = train_dataset.column_names

train_dataset = train_dataset.map(
    tokenize,
    batched=True,
    remove_columns=column_names,
)

eval_dataset = eval_dataset.map(
    tokenize,
    batched=True,
    remove_columns=column_names,
)


# In[7]:


train_dataset.to_pandas()


# Create `Trainer` object and train:

# In[8]:


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
)


# In[9]:


from datasets import load_metric

metric = load_metric('sacrebleu')

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # get text for predictions
    predictions = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )
    # replace -100 in labels with pad token
    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )
    # get text for gold labels
    references = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )
    # metric expects list of references for each prediction
    references = [[ref] for ref in references]
    
    # compute bleu score
    results = metric.compute(
        predictions=predictions,
        references=references,
    )
    results = {'bleu': results['score']}
    
    return results


# In[10]:


from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_steps=save_steps,
    predict_with_generate=True,
    evaluation_strategy='steps',
    eval_steps=save_steps,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
)


# In[11]:


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# In[12]:


import os
from transformers.trainer_utils import get_last_checkpoint

last_checkpoint = None
if os.path.isdir(output_dir):
    last_checkpoint = get_last_checkpoint(output_dir)

if last_checkpoint is not None:
    print(f'Checkpoint detected, resuming training at {last_checkpoint}.')


# In[13]:


train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
trainer.save_model()


# In[14]:


metrics = train_result.metrics
metrics['train_samples'] = len(train_dataset)

trainer.log_metrics('train', metrics)
trainer.save_metrics('train', metrics)
trainer.save_state()


# Now evaluate:

# In[15]:


# https://discuss.huggingface.co/t/evaluation-results-metric-during-training-is-different-from-the-evaluation-results-at-the-end/15401

metrics = trainer.evaluate(
    max_length=max_target_length,
    num_beams=num_beams,
    metric_key_prefix='eval',
)

metrics['eval_samples'] = len(eval_dataset)

trainer.log_metrics('eval', metrics)
trainer.save_metrics('eval', metrics)


# Create a model card with meta data about this model:

# In[16]:


kwargs = {
    'finetuned_from': transformer_name,
    'tasks': 'translation',
    'dataset_tags': dataset_name,
    'dataset_args': dataset_config_name,
    'dataset': f'{dataset_name} {dataset_config_name}',
    'language': [source_lang, target_lang],
}
trainer.create_model_card(**kwargs)


# In[ ]:




