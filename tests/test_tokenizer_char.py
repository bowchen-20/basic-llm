import pytest

from tokenizer import CharTokenizer, tokenizer_from_state

TEXT = "hello world"


def test_encode_decode_roundtrip():
    tok = CharTokenizer(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT
    assert tok.decode(tok.encode("oh well")) == "oh well"


def test_vocab_size_matches_unique_chars():
    tok = CharTokenizer(TEXT)
    assert tok.vocab_size == len(set(TEXT))


def test_unseen_char_raises_keyerror():
    tok = CharTokenizer(TEXT)
    with pytest.raises(KeyError):
        tok.encode("xyz123")


def test_get_state_roundtrip():
    tok = CharTokenizer(TEXT)
    state = tok.get_state()
    assert state == {"type": "char", "vocab": tok.vocab}

    tok2 = tokenizer_from_state(state)
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode(TEXT) == tok.encode(TEXT)
    assert tok2.decode(tok.encode(TEXT)) == TEXT
