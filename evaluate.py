"""
Evaluate a checkpoint: compute loss, perplexity, and bits-per-token.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoint.pt --data data/test.txt
    python evaluate.py --max_batches 200
"""

import argparse
import math
import torch
from tokenizer import tokenizer_from_state
from model import TransformerLM
from config import ModelConfig
from data import build_datasets, make_loader


def load_checkpoint(path: str, device: str):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    tok   = tokenizer_from_state(ckpt["tokenizer_state"])
    cfg   = ModelConfig(**ckpt["model_config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), tok, cfg


def evaluate(
    model: TransformerLM,
    loader,
    device: str,
    max_batches: int,
) -> dict[str, float]:
    total_loss   = 0.0
    total_tokens = 0
    n_batches    = 0

    with torch.no_grad():
        for x, y in loader:
            if n_batches >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, y)
            B, T = x.shape
            total_loss   += loss.item() * B * T
            total_tokens += B * T
            n_batches    += 1

    avg_loss = total_loss / total_tokens
    return {
        "loss":       avg_loss,
        "perplexity": math.exp(avg_loss),
        "bpt":        avg_loss / math.log(2),   # bits per token
        "tokens":     total_tokens,
        "batches":    n_batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--checkpoint",  default="checkpoint.pt")
    parser.add_argument("--data",        default="data/input.txt",
                        help="Text file to evaluate on (uses the val split)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Cap evaluation at this many batches (0 = all)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg = load_checkpoint(args.checkpoint, device)
    print(f"Model     : {model.num_params():,} params")
    print(f"Tokenizer : {tok.__class__.__name__}  vocab_size={tok.vocab_size}")
    print(f"Device    : {device}")

    try:
        with open(args.data, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Data file not found: {args.data}")
        return

    _, val_ds = build_datasets(text, tok, cfg.block_size)
    loader    = make_loader(val_ds, args.batch_size, shuffle=False)
    cap       = args.max_batches if args.max_batches > 0 else len(loader)

    print(f"\nEvaluating on val split ({len(val_ds):,} windows)...")
    results = evaluate(model, loader, device, max_batches=cap)

    print(f"\n{'Batches evaluated':<22}: {results['batches']}")
    print(f"{'Tokens evaluated':<22}: {results['tokens']:,}")
    print(f"{'Loss':<22}: {results['loss']:.4f}")
    print(f"{'Perplexity':<22}: {results['perplexity']:.2f}")
    print(f"{'Bits per token':<22}: {results['bpt']:.4f}")


if __name__ == "__main__":
    main()
