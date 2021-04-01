# Chapter 4

Download the dataset located here: https://ai.stanford.edu/~amaas/data/sentiment/

    wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    tar xzvf aclImdb_v1.tar.gz

Prepare the dataset

    python preprocess_dataset.py ~/data/gentlenlp/chapter4/aclImdb out --compress

Train the perceptron:

    python train_perceptron.py out/train.npz out/perceptron.npz

Evaluate the perceptron on the test data:

    python test_perceptron.py out/test.npz out/perceptron.npz