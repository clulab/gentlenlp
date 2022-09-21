---
title: Before You Start
has_children: false
nav_order: 2
---

# Setting Up Your Conda Environment

Before you start, please set up a new conda environment as follows:

```
conda create --name book
conda activate book

conda install pytorch torchvision torchaudio torchtext cpuonly -c pytorch
conda install jupyter pandas matplotlib scikit-learn gensim nltk
pip install conllu
conda install -c huggingface transformers
conda install -c huggingface -c conda-forge datasets
```

Note that as these libraries evolve you may run into versions that are no longer compatible with this code. To control for this situations, we list below the exact environments that were used to test this code. You can install any of these using the command `conda env create -f <ENVIRONMENT-NAME>`, e.g., `conda env create -f environment_gpu.yml` to install the environment for a Linux machine with GPU.

Environments in which this code was tested:
- [Linux machine with a GPU](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_gpu.yml)
- [Linux machine without a GPU](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_cpu.yml)
- [M1 Mac](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_mac.yml)

# Datasets Used 

## Binary Classification

For binary classification we used the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) created by Andrew Maas. Because this dataset allows redistribution, we copied it in this repository at this location: [https://github.com/clulab/gentlenlp/tree/main/notebooks/data/aclImdb](https://github.com/clulab/gentlenlp/tree/main/notebooks/data/aclImdb). Please see the [dataset's README](https://github.com/clulab/gentlenlp/blob/main/notebooks/data/aclImdb/README) for more details, including the appropriate citation if you use this dataset in research publications. 

## Multiclass Classification



## Part-of-speech Tagging



## Machine Translation







