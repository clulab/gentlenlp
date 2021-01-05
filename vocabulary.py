class Vocabulary:
    def __init__(self, i2w=None):
        self.i2w = [] if i2w is None else i2w
        # make mapping word->id by flipping id->word
        self.w2i = {w:i for i,w in enumerate(self.i2w)}
        # add some special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.pad = self.add_word(self.pad_token) # it is usually better if pad has id 0
        self.unk = self.add_word(self.unk_token)

    def __len__(self):
        return len(self.i2w)

    @classmethod
    def load(cls, filename):
        with open(filename) as f:
            return cls(f.read().splitlines())

    def save(self, filename):
        with open(filename, 'w') as f:
            for w in self.i2w:
                print(w, file=f)

    def add_word(self, w):
        if w in self.w2i:
            i = self.w2i[w]
        else:
            i = len(self.i2w)
            self.i2w.append(w)
            self.w2i[w] = i
        return i

    def get_word(self, i):
        if 0 <= i < len(self.i2w):
            return self.i2w[i]
        return self.unk_token

    def get_word_id(self, w):
        return self.w2i.get(w, self.unk)

    def add_words(self, words):
        return [self.add_word(w) for w in words]

    def get_words(self, ids):
        return [self.get_word(i) for i in ids]

    def get_word_ids(self, words):
        return [self.get_word_id(w) for w in words]
