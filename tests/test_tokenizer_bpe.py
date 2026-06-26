import pytest

from tokenizer import BPETokenizer, tokenizer_from_state

CORPUS = "the quick brown fox jumps over the lazy dog. " * 100
UNICODE_TEXT = "café naïve 日本語 emoji: 🎉🔥 résumé"


def test_train_produces_requested_vocab_size():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 256


def test_encode_decode_roundtrip_ascii():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    assert tok.decode(tok.encode(CORPUS)) == CORPUS
    novel = "the lazy fox jumps over the quick dog"
    assert tok.decode(tok.encode(novel)) == novel


def test_encode_decode_roundtrip_unicode():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=260)  # tiny merge set; unicode text is off-corpus
    assert tok.decode(tok.encode(UNICODE_TEXT)) == UNICODE_TEXT


def test_special_tokens_only_inserted_when_allowed():
    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    tok.train(CORPUS, vocab_size=300)
    text = "hello<|endoftext|>world"
    special_id = tok.token_to_id("<|endoftext|>")

    ids_default = tok.encode(text)
    assert special_id not in ids_default
    assert tok.decode(ids_default) == text

    ids_allowed = tok.encode(text, allowed_special={"<|endoftext|>"})
    assert special_id in ids_allowed
    assert ids_allowed.count(special_id) == 1


def test_encode_plus_padding_and_attention_mask():
    tok = BPETokenizer(special_tokens=["<|pad|>"])
    tok.train(CORPUS, vocab_size=300)
    text = "the quick fox"
    true_len = len(tok.encode(text))
    max_length = true_len + 5

    result = tok.encode_plus(text, max_length=max_length, padding=True, truncation=True)
    assert len(result["input_ids"]) == max_length
    assert len(result["attention_mask"]) == max_length
    assert sum(result["attention_mask"]) == true_len

    pad_id = tok.token_to_id("<|pad|>")
    assert result["input_ids"][true_len:] == [pad_id] * (max_length - true_len)


def test_encode_batch_plus_consistent_length():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    texts = ["the fox", "the quick brown fox jumps", "dog"]
    result = tok.encode_batch_plus(texts, padding=True)

    lengths = {len(ids) for ids in result["input_ids"]}
    assert len(lengths) == 1
    expected_len = max(len(tok.encode(t)) for t in texts)
    assert lengths == {expected_len}

    for ids, mask, text in zip(result["input_ids"], result["attention_mask"], texts):
        true_len = len(tok.encode(text))
        assert sum(mask) == true_len
        assert mask[:true_len] == [1] * true_len
        assert mask[true_len:] == [0] * (len(mask) - true_len)


def test_save_load_roundtrip(tmp_path):
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    path = tmp_path / "tok.json"
    tok.save(str(path))

    loaded = BPETokenizer()
    loaded.load(str(path))

    assert loaded.vocab_size == tok.vocab_size
    assert loaded.merges == tok.merges
    assert loaded.encode(CORPUS) == tok.encode(CORPUS)
    assert loaded.decode(loaded.encode(CORPUS)) == CORPUS


def test_get_state_roundtrip():
    tok = BPETokenizer(special_tokens=["<|endoftext|>"])
    tok.train(CORPUS, vocab_size=300)

    tok2 = tokenizer_from_state(tok.get_state())
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode(CORPUS) == tok.encode(CORPUS)
    assert tok2.decode(tok2.encode(CORPUS)) == CORPUS


def test_compression_ratio_at_least_one_on_repetitive_text():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    assert tok.compression_ratio(CORPUS) > 1.0


def test_token_to_id_id_to_token_consistency():
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=300)
    for tid in tok.encode(CORPUS):
        s = tok.id_to_token(tid)
        if s is not None and not s.startswith("b'"):
            assert tok.token_to_id(s) == tid
