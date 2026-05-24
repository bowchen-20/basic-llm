"""
Standalone inference script.

Usage:
    python generate.py                                  # empty prompt
    python generate.py --prompt "hello"
    python generate.py --prompt "hello" --max_tokens 300 --temperature 0.9 --top_k 50
    python generate.py --checkpoint checkpoint.pt --no_cache
"""

import argparse
import torch
from tokenizer import CharTokenizer
from model import TransformerLM
from config import ModelConfig


def load_checkpoint(path: str, device: str) -> tuple[TransformerLM, CharTokenizer]:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Reconstruct tokenizer from saved vocab list.
    vocab = ckpt["tokenizer_vocab"]
    tok = CharTokenizer.__new__(CharTokenizer)
    tok.vocab      = vocab
    tok.vocab_size = len(vocab)
    tok._stoi      = {c: i for i, c in enumerate(vocab)}
    tok._itos      = {i: c for i, c in enumerate(vocab)}

    # Reconstruct model from saved config.
    cfg = ModelConfig(**ckpt["model_config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), tok


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint",   default="checkpoint.pt",  help="Path to .pt checkpoint")
    parser.add_argument("--prompt",       default="",               help="Prompt string (empty = null token)")
    parser.add_argument("--max_tokens",   type=int,   default=200,  help="Number of tokens to generate")
    parser.add_argument("--temperature",  type=float, default=0.8,  help="Sampling temperature")
    parser.add_argument("--top_k",        type=int,   default=40,   help="Top-k sampling (0 = disabled)")
    parser.add_argument("--no_cache",     action="store_true",      help="Disable KV-cache (slower but simpler)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {model.num_params():,}-param model on {device}")

    if args.prompt:
        ids = tok.encode(args.prompt)
    else:
        ids = [0]

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens = args.max_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k if args.top_k > 0 else None,
            use_cache      = not args.no_cache,
        )

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
