---
title: Before You Start
has_children: false
nav_order: 2
---

# Setting Up Conda Environment

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
