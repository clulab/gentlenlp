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

Note that 