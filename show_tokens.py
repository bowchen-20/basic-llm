"""
Visualize how text is tokenized, with alternating color highlighting.

Usage:
    python show_tokens.py --text "hello world"
    python show_tokens.py --text "the quick brown fox" --show_ids
    echo "some text" | python show_tokens.py
    python show_tokens.py --checkpoint checkpoint_best.pt --text "..."
"""

import argparse
import sys
import torch
from tokenizer import tokenizer_from_state


_COLORS = [
    "\033[48;5;214m\033[30m",   # orange background, black text
    "\033[48;5;75m\033[30m",    # blue background, black text
]
_RESET = "\033[0m"


def colorize(s: str, idx: int) -> str:
    return f"{_COLORS[idx % 2]}{s}{_RESET}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize tokenization.")
    parser.add_argument("--checkpoint", default="checkpoint.pt",
                        help="Checkpoint to load the tokenizer from")
    parser.add_argument("--text",       default=None,
                        help="Text to tokenize (reads stdin if omitted)")
    parser.add_argument("--show_ids",   action="store_true",
                        help="Print a table of token indices and IDs")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tok  = tokenizer_from_state(ckpt["tokenizer_state"])

    if args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read().rstrip("\n")

    if not text:
        print("No text provided (pass --text or pipe via stdin).")
        return

    ids    = tok.encode(text)
    pieces = [tok.decode([i]) for i in ids]

    print(f"Tokenizer  : {tok.__class__.__name__}  vocab_size={tok.vocab_size}")
    print(f"Input      : {len(text)} chars  →  {len(ids)} tokens  "
          f"({len(text) / max(1, len(ids)):.2f} chars/token)")
    print()
    print("".join(colorize(p, i) for i, p in enumerate(pieces)))
    print()

    if args.show_ids:
        print(f"{'idx':>5}  {'id':>6}  {'token'}")
        print("-" * 32)
        for i, (piece, tid) in enumerate(zip(pieces, ids)):
            print(f"{i:>5}  {tid:>6}  {repr(piece)}")


if __name__ == "__main__":
    main()
