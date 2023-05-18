---
title: Before You Start Coding
has_children: false
nav_order: 2
---

# Setting Up Your Conda Environment

Before you start, please set up a new conda environment as follows.

If your machine does not have a GPU:

```
conda create --name book
conda activate book

conda install pip
conda install pytorch torchvision torchaudio torchtext cpuonly -c pytorch
conda install jupyter pandas matplotlib scikit-learn gensim nltk
pip install conllu
pip install transformers
pip install datasets
```

If your machine machine has an Nvidia GPU:
```
conda create --name book
conda activate book

conda install pip
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
conda install jupyter pandas matplotlib scikit-learn gensim nltk
pip install conllu
pip install transformers
pip install datasets
```
(See https://pytorch.org/get-started/locally/ for PyTorch installation instructions on other platforms.)

Note that as these libraries evolve you may run into versions that are no longer compatible with this code. To control for this situation, we list below the exact environments that were used to test this code. You can install any of these using the command `conda env create -f <ENVIRONMENT-NAME>`, e.g., `conda env create -f environment_gpu.yml` to install the environment for a Linux machine with GPU.

Environments in which this code was tested:
- [Linux machine with a GPU](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_gpu.yml)
- [Linux machine without a GPU](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_cpu.yml)
- [M1 Mac](https://github.com/clulab/gentlenlp/blob/main/notebooks/environment_mac.yml)

# External Datasets Used 

## Binary Classification

For binary classification we used the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) created by Andrew Maas. Because this dataset allows redistribution, we copied it in this repository at this location: [https://github.com/clulab/gentlenlp/tree/main/notebooks/data/aclImdb](https://github.com/clulab/gentlenlp/tree/main/notebooks/data/aclImdb), which is the location expected by the notebooks that implement binary classifiers. Please see the [dataset's README](https://github.com/clulab/gentlenlp/blob/main/notebooks/data/aclImdb/README) for more details, including the appropriate citation if you use this dataset in research publications. 

## Multiclass Classification

For multiclass classification we used a version of the [AG News dataset](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html). In particular, we used the simplified form of the dataset from the paper [Character-level Convolutional Networks for Text Classification](https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf), which keeps only the four most frequent labels. The license for this dataset does not allow redistribution, so please download the archive `ag_news_csv.tar.gz` yourself from [here](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ). Uncompress the downloaded file, and copy the `ag_news_csv` folder under `notebooks/data/` in your repository.

## Pre-trained English Word Embeddings

Please download the [these GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip) from the [Stanford GloVe website](https://nlp.stanford.edu/projects/glove/). Once the `glove.6B.zip` file is downloaded, uncompress it and place the extracted `glove.6B.300d.txt` file in the `notebooks/` folder.

## Pre-trained Spanish Word Embeddings

Please download the [these Spanish GloVe embeddings](http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz) from the [Spanish Word Embeddings GitHub repository](https://github.com/dccuchile/spanish-word-embeddings). Once the `glove-sbwc.i25.vec.gz` file is downloaded, uncompress it and place the extracted `glove-sbwc.i25.vec` file in the `notebooks/` folder.

## Part-of-speech Tagging

For part-of-speech tagging we used the Spanish AnCora dataset that is included in the 
[Universal Dependencies version 2.8 dataset](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3720/deep-ud-2.8-data.tgz?sequence=1&isAllowed=y). Its license does not allow redistribution, so please download the UD version 2.8 dataset from [here](https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3720/deep-ud-2.8-data.tgz?sequence=1&isAllowed=y). Uncompress the downloaded `deep-ud-2.8-data.tgz` file, and copy the `UD_Spanish-AnCora` folder under `notebooks/data/` in your repository.








