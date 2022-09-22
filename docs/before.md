---
title: Before You Start
has_children: false
nav_order: 2
---

# Setting Up Your Conda Environment

Before you start, please set up a new conda environment as follows.

If your machine does not have a GPU:

```
conda create --name book
conda activate book

conda install pytorch torchvision torchaudio torchtext cpuonly -c pytorch
conda install jupyter pandas matplotlib scikit-learn gensim nltk
pip install conllu
conda install -c huggingface transformers
conda install -c huggingface -c conda-forge datasets
```

If your machine machine has an Nvidia GPU:
```
TODO
```

Note that as these libraries evolve you may run into versions that are no longer compatible with this code. To control for this situation, we list below the exact environments that were used to test this code. You can install any of these using the command `conda env create -f <ENVIRONMENT-NAME>`, e.g., `conda env create -f environment_gpu.yml` to install the environment for a Linux machine with GPU.

Environments in which this code was tested:
- [Linux machine with a GPU](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_gpu.yml)
- [Linux machine without a GPU](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_cpu.yml)
- [M1 Mac](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_mac.yml)

# External Datasets Used 

## Binary Classification

For binary classification we used the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) created by Andrew Maas. Because this dataset allows redistribution, we copied it in this repository at this location: [https://github.com/clulab/gentlenlp/tree/main/notebooks/data/aclImdb](https://github.com/clulab/gentlenlp/tree/main/notebooks/data/aclImdb), which is the location expected by the notebooks that implement binary classifiers. Please see the [dataset's README](https://github.com/clulab/gentlenlp/blob/main/notebooks/data/aclImdb/README) for more details, including the appropriate citation if you use this dataset in research publications. 

## Multiclass Classification

TODO: downloaded from here: http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html. Preprocessed using this script.

## Pre-trained English Word Embeddings

Please download the [these GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip) from the [Stanford GloVe website](https://nlp.stanford.edu/projects/glove/). Once the `glove.6B.zip` file is downloaded, uncompress it and place the extracted `glove.6B.300d.txt` file in the `notebooks/` folder.

## Pre-trained Spanish Word Embeddings

Please download the [these Spanish GloVe embeddings](http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz) from the [Spanish Word Embeddings GitHub repository](https://github.com/dccuchile/spanish-word-embeddings). Once the `glove-sbwc.i25.vec.gz` file is downloaded, uncompress it and place the extracted `glove-sbwc.i25.vec` file in the `notebooks/` folder.

## Part-of-speech Tagging

For part-of-speech tagging with recurrent neural networks we used the Spanish AnCora dataset that is included in the 
[Universal Dependencies version 2.10 dataset](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz?sequence=1&isAllowed=y). Because its license allows redistribution, we copied this dataset at this location: [https://github.com/clulab/gentlenlp/tree/main/notebooks/data/UD_Spanish-AnCora](https://github.com/clulab/gentlenlp/tree/main/notebooks/data/UD_Spanish-AnCora), which is the location expected by the notebook that implements part-of-speech tagging. Please see the [dataset's README](https://github.com/clulab/gentlenlp/blob/main/notebooks/data/UD_Spanish-AnCora/README.md) for more details, including the appropriate citation if you use this dataset in research publications. 








