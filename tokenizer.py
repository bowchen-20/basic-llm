class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self._stoi = {c: i for i, c in enumerate(chars)}
        self._itos = {i: c for i, c in enumerate(chars)}

    def encode(self, text):
        return [self._stoi[c] for c in text]

    def decode(self, tokens):
        return "".join(self._itos[t] for t in tokens)
