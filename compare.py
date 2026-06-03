"""
Evaluate two checkpoints on the same data and print a side-by-side comparison.

Usage:
    python compare.py checkpoint.pt checkpoint_best.pt
    python compare.py a.pt b.pt --data data/input.txt --max_batches 100
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
    return model.to(device), tok, cfg, ckpt


@torch.no_grad()
def eval_checkpoint(model, loader, device, max_batches: int) -> dict[str, float]:
    total_loss, total_tokens, n = 0.0, 0, 0
    for x, y in loader:
        if n >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, y)
        B, T = x.shape
        total_loss   += loss.item() * B * T
        total_tokens += B * T
        n += 1
    avg = total_loss / total_tokens
    return {"loss": avg, "ppl": math.exp(avg), "bpt": avg / math.log(2), "batches": n}


def fmt_delta(new: float, old: float, lower_is_better: bool = True) -> str:
    d = new - old
    better = (d < 0) == lower_is_better
    sign = "▼" if d < 0 else "▲"
    color = "\033[32m" if better else "\033[31m"
    return f"{color}{sign}{abs(d):.4f}\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two checkpoints on val data.")
    parser.add_argument("checkpoint_a")
    parser.add_argument("checkpoint_b")
    parser.add_argument("--data",        default="data/input.txt")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        with open(args.data, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Data file not found: {args.data}")
        return

    results = []
    for path in [args.checkpoint_a, args.checkpoint_b]:
        model, tok, cfg, ckpt = load_checkpoint(path, device)
        _, val_ds = build_datasets(text, tok, cfg.block_size)
        loader    = make_loader(val_ds, args.batch_size, shuffle=False)
        cap       = args.max_batches if args.max_batches > 0 else len(loader)
        print(f"Evaluating {path} ...")
        m = eval_checkpoint(model, loader, device, cap)
        results.append((path, ckpt.get("step", "?"), model.num_params(), m))

    print()
    w = max(len(r[0]) for r in results) + 2
    print(f"{'Checkpoint':<{w}} {'Step':>7} {'Params':>12} {'Loss':>8} {'PPL':>8} {'BPT':>8}")
    print("-" * (w + 50))
    for path, step, params, m in results:
        print(
            f"{path:<{w}} {str(step):>7} {params:>12,} "
            f"{m['loss']:>8.4f} {m['ppl']:>8.2f} {m['bpt']:>8.4f}"
        )

    if len(results) == 2:
        a, b = results[0][3], results[1][3]
        print()
        print(f"  {'B vs A':10}  loss {fmt_delta(b['loss'], a['loss'])}  "
              f"ppl {fmt_delta(b['ppl'], a['ppl'])}")
        winner = results[1][0] if b["loss"] < a["loss"] else results[0][0]
        print(f"  lower loss → {winner}")


if __name__ == "__main__":
    main()
