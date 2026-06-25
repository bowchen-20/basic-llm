"""
Inspect a checkpoint: parameter counts, weight statistics, sample output.

Usage:
    python inspect_model.py
    python inspect_model.py --params
    python inspect_model.py --stats
    python inspect_model.py --sample --prompt "hello"
    python inspect_model.py --checkpoint checkpoint_best.pt --params --stats
"""

import argparse
import torch
from tokenizer import tokenizer_from_state
from model import TransformerLM
from config import ModelConfig


def load_checkpoint(path: str, device: str):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    tok   = tokenizer_from_state(ckpt["tokenizer_state"])
    cfg   = ModelConfig(**ckpt["model_config"])
    model = TransformerLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model.to(device), tok, cfg, ckpt


def print_param_table(model: TransformerLM) -> None:
    print(f"{'Parameter':<55} {'Shape':<25} {'Count':>10}")
    print("-" * 94)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        print(f"{name:<55} {str(tuple(p.shape)):<25} {n:>10,}")
    print("-" * 94)
    print(f"{'TOTAL':<55} {'':25} {total:>10,}")


def print_weight_stats(model: TransformerLM) -> None:
    print(f"{'Parameter':<55} {'mean':>9} {'std':>9} {'abs_max':>9}")
    print("-" * 86)
    for name, p in model.named_parameters():
        d = p.detach().float()
        print(
            f"{name:<55} {d.mean().item():>9.4f} "
            f"{d.std().item():>9.4f} {d.abs().max().item():>9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a model checkpoint.")
    parser.add_argument("--checkpoint",  default="checkpoint.pt")
    parser.add_argument("--params",      action="store_true",
                        help="Print parameter shapes and counts per layer")
    parser.add_argument("--stats",       action="store_true",
                        help="Print mean/std/abs_max for every weight tensor")
    parser.add_argument("--sample",      action="store_true",
                        help="Generate a short sample from the model")
    parser.add_argument("--prompt",      default="")
    parser.add_argument("--max_tokens",  type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",       type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg, ckpt = load_checkpoint(args.checkpoint, device)

    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Step        : {ckpt.get('step', 'unknown')}")
    print(f"Tokenizer   : {tok.__class__.__name__}  vocab_size={tok.vocab_size}")
    print(f"Architecture: n_embd={cfg.n_embd}  n_head={cfg.n_head}  n_layer={cfg.n_layer}"
          f"  block_size={cfg.block_size}")
    n_kv = cfg.n_kv_head or cfg.n_head
    if n_kv != cfg.n_head:
        print(f"              GQA: n_kv_head={n_kv}  groups={cfg.n_head // n_kv}")
    print(f"Total params: {model.num_params():,}")
    print()

    if args.params:
        print_param_table(model)
        print()

    if args.stats:
        print_weight_stats(model)
        print()

    if args.sample:
        ids = tok.encode(args.prompt) if args.prompt else [0]
        x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens = args.max_tokens,
                temperature    = args.temperature,
                top_k          = args.top_k,
            )
        print("--- sample ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
