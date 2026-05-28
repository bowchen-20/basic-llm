"""
Throughput benchmark: measure tokens/second for prefill and decode.

Usage:
    python bench.py
    python bench.py --checkpoint checkpoint.pt --gen_len 256 --batch_size 4
    python bench.py --no_cache         # compare cached vs uncached decode
    python bench.py --repeats 10
"""

import argparse
import time
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
    return model.to(device), tok, cfg


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def bench_generate(model, x, gen_len, use_cache, repeats, device):
    _sync(device)
    with torch.no_grad():
        model.generate(x, max_new_tokens=min(gen_len, 8), use_cache=use_cache)
    _sync(device)

    times: list[float] = []
    with torch.no_grad():
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            model.generate(x, max_new_tokens=gen_len, use_cache=use_cache)
            _sync(device)
            times.append(time.perf_counter() - t0)

    mean_t  = sum(times) / len(times)
    tokens  = x.size(0) * gen_len
    tok_sec = tokens / mean_t
    return mean_t, tok_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model throughput.")
    parser.add_argument("--checkpoint",  default="checkpoint.pt")
    parser.add_argument("--prompt_len",  type=int, default=32,
                        help="Number of prompt tokens (random)")
    parser.add_argument("--gen_len",     type=int, default=128,
                        help="Number of new tokens to generate per run")
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--repeats",     type=int, default=5)
    parser.add_argument("--no_cache",    action="store_true",
                        help="Benchmark without KV cache")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, cfg = load_checkpoint(args.checkpoint, device)

    print(f"Model      : {model.num_params():,} params")
    print(f"Tokenizer  : {tok.__class__.__name__}  vocab_size={tok.vocab_size}")
    print(f"Device     : {device}")
    print(f"Batch size : {args.batch_size}")
    print(f"Prompt len : {args.prompt_len} tokens")
    print(f"Gen len    : {args.gen_len} tokens")
    print(f"Repeats    : {args.repeats}")
    print()

    x = torch.randint(0, tok.vocab_size, (args.batch_size, args.prompt_len), device=device)

    for use_cache in ([True, False] if not args.no_cache else [False]):
        label = "with KV-cache" if use_cache else "no cache     "
        mean_t, tok_sec = bench_generate(model, x, args.gen_len, use_cache, args.repeats, device)
        print(
            f"{label}  |  mean={mean_t * 1000:.1f} ms  |  "
            f"{tok_sec:.0f} tok/s  ({args.batch_size * args.gen_len} tokens / run)"
        )


if __name__ == "__main__":
    main()
