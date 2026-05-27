"""
Standalone inference script.

Usage:
    python generate.py
    python generate.py --prompt "hello world"
    python generate.py --checkpoint checkpoint.pt --max_tokens 300 --temperature 0.9 --top_k 50
    python generate.py --top_p 0.9
    python generate.py --interactive
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


def run_generation(model, tok, prompt, args, device):
    ids = tok.encode(prompt) if prompt else [0]
    x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens = args.max_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k if args.top_k > 0 else None,
            top_p          = args.top_p if args.top_p > 0.0 else None,
            use_cache      = not args.no_cache,
        )
    return tok.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint",   default="checkpoint.pt")
    parser.add_argument("--prompt",       default="")
    parser.add_argument("--max_tokens",   type=int,   default=200)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_k",        type=int,   default=40,  help="0 = disabled")
    parser.add_argument("--top_p",        type=float, default=0.0, help="nucleus sampling, 0 = disabled")
    parser.add_argument("--no_cache",     action="store_true")
    parser.add_argument("--interactive",  action="store_true",
                        help="Start an interactive prompt loop (REPL)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {model.num_params():,}-param model on {device}")

    if args.interactive:
        print("Interactive mode. Type a prompt and press Enter. Empty line to quit.\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                break
            print(run_generation(model, tok, prompt, args, device))
            print()
    else:
        print(run_generation(model, tok, args.prompt, args, device))


if __name__ == "__main__":
    main()
