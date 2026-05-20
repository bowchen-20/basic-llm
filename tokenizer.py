import re
import json
from collections import defaultdict

# Splits text into chunks before BPE runs, preventing merges across word/punctuation
# boundaries. Keeps a leading space as part of a word token (e.g. " hello" is one
# chunk), matching GPT-2 convention. Order matters: contractions must come first.
_PRETOK = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[a-zA-Z]+"
    r"| ?[0-9]+"
    r"| ?[^\s\w]+"
    r"|\s+"
)


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


class BPETokenizer:
    """
    Byte-pair encoding tokenizer.

    Base vocabulary: 256 raw bytes.
    Merge tokens: learned by train(), appended above 255.
    Special tokens: fixed strings (e.g. <|endoftext|>) appended above merges;
                    never produced by BPE — only inserted when explicitly allowed.
    """

    def __init__(self, special_tokens: list[str] | None = None):
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {}
        self.vocab_size: int = 0
        self._special_tokens: list[str] = special_tokens or []
        self._special: dict[str, int] = {}  # token string -> id

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, "
            f"merges={len(self.merges)}, "
            f"special={list(self._special)})"
        )

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab_size must be at least 256 (byte base vocab)"

        # Pre-tokenize: split into chunks, then encode each chunk to bytes.
        # BPE merges happen independently within each chunk — never across them.
        chunks: list[list[int]] = [
            list(chunk.encode("utf-8")) for chunk in _PRETOK.findall(text)
        ]

        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        num_merges = vocab_size - 256 - len(self._special_tokens)
        for i in range(num_merges):
            counts: dict[tuple[int, int], int] = defaultdict(int)
            for chunk in chunks:
                for pair in zip(chunk, chunk[1:]):
                    counts[pair] += 1
            if not counts:
                break
            best = max(counts, key=counts.__getitem__)
            new_id = 256 + i
            chunks = [self._merge(chunk, best, new_id) for chunk in chunks]
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if verbose:
                try:
                    display = repr(self.vocab[new_id].decode("utf-8"))
                except UnicodeDecodeError:
                    display = repr(self.vocab[new_id])
                print(f"  [{i + 1:4d}/{num_merges}] {new_id} -> {display}  (freq={counts[best]})")

        # Special tokens are assigned IDs above all merge tokens.
        for tok in self._special_tokens:
            sid = len(self.vocab)
            self._special[tok] = sid
            self.vocab[sid] = tok.encode("utf-8")

        self.vocab_size = len(self.vocab)

    @classmethod
    def from_file(
        cls,
        path: str,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Train a new tokenizer on the contents of a text file."""
        with open(path, encoding="utf-8") as f:
            text = f.read()
        tok = cls(special_tokens=special_tokens)
        tok.train(text, vocab_size, verbose=verbose)
        return tok

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------

    def encode(self, text: str, allowed_special: set[str] | None = None) -> list[int]:
        """
        Encode text to token ids.

        Special tokens are only inserted when their string appears in
        allowed_special — otherwise they are encoded as ordinary bytes.
        This mirrors tiktoken's safety behaviour.
        """
        allowed = allowed_special or set()
        active = {s: sid for s, sid in self._special.items() if s in allowed}

        # Split on any active special tokens first, preserving them as parts.
        if active:
            pat = re.compile("(" + "|".join(re.escape(s) for s in active) + ")")
            parts = pat.split(text)
        else:
            parts = [text]

        ids: list[int] = []
        for part in parts:
            if part in active:
                ids.append(active[part])
            else:
                for chunk in _PRETOK.findall(part):
                    ids.extend(self._bpe_encode_chunk(list(chunk.encode("utf-8"))))
        return ids

    def encode_batch(
        self,
        texts: list[str],
        allowed_special: set[str] | None = None,
    ) -> list[list[int]]:
        """Encode a list of strings, returning a list of token id lists."""
        return [self.encode(t, allowed_special=allowed_special) for t in texts]

    def encode_plus(
        self,
        text: str,
        max_length: int | None = None,
        padding: bool = False,
        truncation: bool = False,
        allowed_special: set[str] | None = None,
    ) -> dict:
        """
        Encode with optional truncation and padding.

        Returns {"input_ids": [...], "attention_mask": [...]}.
        Padding uses the <|pad|> special token id when present, else 0.
        attention_mask is 1 for real tokens and 0 for padding.
        """
        ids = self.encode(text, allowed_special=allowed_special)

        if truncation and max_length is not None:
            ids = ids[:max_length]

        attention_mask = [1] * len(ids)

        if padding and max_length is not None:
            pad_id = self._special.get("<|pad|>", 0)
            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                attention_mask = attention_mask + [0] * pad_len

        return {"input_ids": ids, "attention_mask": attention_mask}

    def encode_with_offsets(
        self, text: str
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """
        Encode text and return (ids, offsets).

        offsets[i] is the (char_start, char_end) span in the original string
        for token ids[i]. Useful for token-to-character alignment in NER/QA.
        """
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        text_encoded = text.encode("utf-8")

        for m in _PRETOK.finditer(text):
            chunk = m.group()
            chunk_start_byte = len(text[: m.start()].encode("utf-8"))
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._bpe_encode_chunk(list(chunk_bytes))

            byte_cursor = 0
            for tid in chunk_ids:
                n = len(self.vocab[tid])
                char_start = len(
                    text_encoded[: chunk_start_byte + byte_cursor].decode(
                        "utf-8", errors="replace"
                    )
                )
                char_end = len(
                    text_encoded[: chunk_start_byte + byte_cursor + n].decode(
                        "utf-8", errors="replace"
                    )
                )
                offsets.append((char_start, char_end))
                byte_cursor += n

            ids.extend(chunk_ids)

        return ids, offsets

    def count_tokens(self, text: str, allowed_special: set[str] | None = None) -> int:
        """Return token count without materialising the full list."""
        return len(self.encode(text, allowed_special=allowed_special))

    def decode(self, tokens: list[int]) -> str:
        raw = b"".join(self.vocab[t] for t in tokens)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # token lookup
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> int | None:
        """Return the id for a token string, or None if not in vocab."""
        bs = token.encode("utf-8")
        for tid, tbytes in self.vocab.items():
            if tbytes == bs:
                return tid
        return None

    def id_to_token(self, token_id: int) -> str | None:
        """Return the string for a token id, or None if not in vocab."""
        bts = self.vocab.get(token_id)
        if bts is None:
            return None
        try:
            return bts.decode("utf-8")
        except UnicodeDecodeError:
            return repr(bts)

    # ------------------------------------------------------------------
    # inspection / stats
    # ------------------------------------------------------------------

    def show_vocab(self, n: int = 30) -> None:
        """Print the n most recently learned merge tokens plus any special tokens."""
        special_ids = set(self._special.values())
        merged = [
            (tid, bts)
            for tid, bts in self.vocab.items()
            if tid >= 256 and tid not in special_ids
        ]
        for tid, bts in merged[-n:]:
            try:
                display = repr(bts.decode("utf-8"))
            except UnicodeDecodeError:
                display = repr(bts)
            print(f"  {tid:5d}: {display}")
        for s, sid in self._special.items():
            print(f"  {sid:5d}: {repr(s)}  <special>")

    def vocab_stats(self) -> None:
        """Print a summary of vocabulary composition and token length distribution."""
        special_ids = set(self._special.values())
        merge_tokens = [
            v for k, v in self.vocab.items() if k >= 256 and k not in special_ids
        ]
        lengths = [len(v) for v in merge_tokens]

        print(f"vocab_size     : {self.vocab_size}")
        print(f"base tokens    : 256  (raw bytes 0-255)")
        print(f"merge tokens   : {len(merge_tokens)}")
        print(f"special tokens : {len(self._special)}  {list(self._special)}")
        if lengths:
            print(f"avg merge len  : {sum(lengths) / len(lengths):.2f} bytes")
            print(f"max merge len  : {max(lengths)} bytes")

    def compression_ratio(self, text: str) -> float:
        """
        Bytes per token on this text.
        Higher means the vocab compresses the corpus more efficiently.
        Byte-level baseline is 1.0; a well-trained BPE on English text is ~4-5.
        """
        n_bytes = len(text.encode("utf-8"))
        n_tokens = self.count_tokens(text)
        return n_bytes / n_tokens if n_tokens else 0.0

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "merges": {f"{a},{b}": c for (a, b), c in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
            "special": {s: sid for s, sid in self._special.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.merges = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in data["merges"].items()
        }
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        self._special = {s: sid for s, sid in data.get("special", {}).items()}
        self._special_tokens = list(self._special.keys())
        self.vocab_size = len(self.vocab)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _bpe_encode_chunk(self, ids: list[int]) -> list[int]:
        while len(ids) >= 2:
            # Find the pair with the lowest merge index (earliest learned merge).
            best = min(
                set(zip(ids, ids[1:])),
                key=lambda p: self.merges.get(p, float("inf")),
            )
            if best not in self.merges:
                break
            ids = self._merge(ids, best, self.merges[best])
        return ids

    @staticmethod
    def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        result: list[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result
