"""
Standalone inference script.

Usage:
    python generate.py
    python generate.py --prompt "hello world"
    python generate.py --checkpoint checkpoint.pt --max_tokens 300 --temperature 0.9 --top_k 50
    python generate.py --no_cache
"""

import argparse
import torch
from tokenizer import tokenizer_from_state
from model import TransformerLM
from config import ModelConfig


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tok   = tokenizer_from_state(ckpt["tokenizer_state"])
    cfg   = ModelConfig(**ckpt["model_config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), tok


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint",  default="checkpoint.pt")
    parser.add_argument("--prompt",      default="")
    parser.add_argument("--max_tokens",  type=int,   default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int,   default=40,  help="0 = disabled")
    parser.add_argument("--no_cache",    action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {model.num_params():,}-param model on {device}")

    ids = tok.encode(args.prompt) if args.prompt else [0]
    x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

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
