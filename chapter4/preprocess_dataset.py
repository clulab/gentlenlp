import argparse
from pathlib import Path
import numpy as np
from vocabulary import Vocabulary
import imdb

if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=Path, help='path to the data directory')
    parser.add_argument('out_dir', type=Path, help='path to save output files')
    args = parser.parse_args()
    # ensure output directory exists
    if not args.out_dir.exists():
        args.out_dir.mkdir()
    # collect data
    print('collecting training data ...')
    vocab = Vocabulary()
    train_token_ids, train_labels = imdb.read_imdb_data(args.data_dir/'train', vocab, add_tokens=True)
    print('saving vocabulary ...')
    vocab.save(args.out_dir/'vocab.imdb')
    print('converting training data to numpy ...')
    X_train, y_train = imdb.to_numpy(train_token_ids, train_labels, vocab)
    print('saving training data ...')
    np.savez_compressed(args.out_dir/'train.npz', X=X_train, y=y_train)
    print('collecting test data ...')
    test_token_ids, test_labels = imdb.read_imdb_data(args.data_dir/'test', vocab, add_tokens=False)
    print('converting test data to numpy ...')
    X_test, y_test = imdb.to_numpy(test_token_ids, test_labels, vocab)
    print('saving test data ...')
    np.savez_compressed(args.out_dir/'test.npz', X=X_test, y=y_test)
