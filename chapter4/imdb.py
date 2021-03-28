import string
import unicodedata
from pathlib import Path
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize.casual import TweetTokenizer
from vocabulary import Vocabulary

def read_imdb_data(data_dir: Path, vocabulary: Vocabulary, add_tokens: bool):
    all_token_ids, all_labels = [], []
    for label in ['pos', 'neg']:
        d = data_dir/label
        for f in d.glob('*.txt'):
            tokens = tokenize_imdb_review(f)
            if add_tokens:
                token_ids = vocabulary.add_tokens(tokens)
            else:
                token_ids = vocabulary.get_token_ids(tokens)
            # collect token_ids
            all_token_ids.append(token_ids)
            # collect corresponding label (fold name)
            all_labels.append(label)
    return all_token_ids, all_labels

def tokenize_imdb_review(filename):
    """returns the tokenized contents of a file"""
    with open(filename) as f:
        text = f.read()
        norm = normalize_text(text)
        return tokenize(norm)

def normalize_text(text):
    """perform text normalization suitable for imdb data"""
    # remove html line breaks
    norm = text.replace('<br />', ' ')
    # convert all text to lowercase, aggressively
    norm = norm.casefold()
    # remove diacritics
    norm = remove_diacritics(norm)
    # return normalized text
    return norm

def remove_diacritics(text):
    # Normalization Form Canonical Decomposition
    nfd_text = unicodedata.normalize('NFD', text)
    # remove combining characters
    stripped_text = ''.join(c for c in nfd_text if not unicodedata.combining(c))
    # Normalization Form Canonical Composition
    return unicodedata.normalize('NFC', stripped_text)

def tokenize(text):
    """gets a string and returns a list of tokens"""
    tokens = []
    # we use TweetTokenizer because it is suitable for social media posts
    tokenizer = TweetTokenizer(reduce_len=True)
    for sent in sent_tokenize(text):
        # we help the tokenizer a little bit by adding spaces
        # around dots and double dashes
        sent = sent.replace('.', ' . ').replace('--', ' -- ')
        for token in tokenizer.tokenize(sent):
            # only add valid tokens to the vocabulary
            if is_valid(token):
                tokens.append(token)
    return tokens

def is_valid(token):
    """returns True if `token` is a valid token"""
    invalid = string.punctuation + string.whitespace
    if any(c.isdigit() for c in token):
        # reject any token that contains at least one digit
        return False
    elif all(c in invalid for c in token):
        # reject tokens composed of invalid characters only,
        # except the question and exclamation marks
        return token in ('?', '!')
    else:
        # accept everything else
        return True

def to_numpy(all_token_ids, all_labels, vocabulary):
    n_rows = len(all_token_ids)
    n_cols = len(vocabulary)
    X = np.zeros((n_rows, n_cols), dtype=np.int32)
    y = np.zeros(n_rows, dtype=np.bool)
    for i in range(n_rows):
        y[i] = all_labels[i] == 'pos'
        for j in all_token_ids[i]:
            X[i, j] = True
    return X, y
