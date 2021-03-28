class Vocabulary:

    def __init__(self, i2t=None):
        # id to term
        self.i2t = [] if i2t is None else i2t
        # term to id
        self.t2i = {t:i for i,t in enumerate(self.i2t)}
        # add special tokens
        self.unk_token = '[UNK]'
        self.unk = self.add_token(self.unk_token)

    def __len__(self):
        return len(self.i2t)

    @classmethod
    def load(cls, filename):
        """loads a vocabulary stored in a file"""
        with open(filename) as f:
            return cls(f.read().splitlines())

    def save(self, filename):
        """saves the vocabulary to a file"""
        with open(filename, 'w') as f:
            for t in self.i2t:
                print(t, file=f)

    def add_tokens(self, tokens):
        """adds a list of tokens to the vocabulary,
        and returns a list of token ids"""
        return [self.add_token(t) for t in tokens]

    def get_tokens(self, ids):
        """gets a list of ids and returns a list of tokens"""
        return [self.get_token(i) for i in ids]

    def get_token_ids(self, tokens):
        """gets a list of tokens and returns a list of token ids"""
        return [self.get_token_id(t) for t in tokens]

    def add_token(self, t):
        """adds a token to the vocabulary if it isn't already there,
        and returns the token id"""
        if t in self.t2i:
            i = self.t2i[t]
        else:
            i = len(self.i2t)
            self.i2t.append(t)
            self.t2i[t] = i
        return i

    def get_token(self, i):
        """returns the token corresponding to the provided id if there is one,
        otherwise it returns the [UNK] token"""
        if 0 <= i < len(self.i2t):
            return self.i2t[i]
        return self.unk_token

    def get_token_id(self, t):
        """returns the token id for the corresponding token,
        or the [UNK] token id if the word is not in the vocabulary"""
        return self.t2i.get(t, self.unk)
