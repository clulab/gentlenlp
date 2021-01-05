import sys
from pathlib import Path
import nltk
from vocabulary import Vocabulary

data_dir = Path(sys.argv[1])
pos_dir = data_dir/'pos'
neg_dir = data_dir/'neg'

def tokenize(text: str):
    # collect tokens
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            tokens.append(word.lower())
    return tokens

def tokenize_file(file):
    with open(file) as f:
        text = f.read()
    # remove html line breaks
    text = text.replace('<br />', ' ')
    # discard any non-ascii character
    text = text.encode('ascii', 'ignore').decode('ascii')
    return tokenize(text)

# make empty vocabulary
vocab = Vocabulary()

# populate the vocabulary
for split in ['pos', 'neg']:
    d = data_dir/split
    for f in d.glob('*.txt'):
        print('working on', f)
        tokens = tokenize_file(f)
        vocab.add_words(tokens)

# save to file
vocab.save('vocabulary.txt')

# test
# 84210 vocabulary.txt