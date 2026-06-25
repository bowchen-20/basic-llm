# Changelog

Running log of what changed in this project, session by session. Newest entries on top.

## 2026-06-25

- Renamed `inspect.py` → `inspect_model.py`. It was shadowing Python's standard library
  `inspect` module (imported internally by `torch`), which broke `import torch` entirely
  whenever run from this directory.
- Verified the full pipeline end-to-end on CPU: installed `torch`/`numpy`, downloaded the
  TinyShakespeare corpus into `data/input.txt` (gitignored, not committed), ran a reduced-iteration
  smoke test of `train.py` to confirm training/checkpointing/sampling all work.
- Rewrote `README.md`, which still described the original 5-step bigram-model roadmap even
  though the project had moved to a full transformer — it now reflects actual current features.
- Added this changelog.

## 2026-05-19 – 2026-06-03 (prior sessions)

- Started from a bigram language model with a character tokenizer and a basic training loop.
- Replaced the character tokenizer with a from-scratch BPE tokenizer (pre-tokenization, special
  tokens, batch ops).
- Replaced `BigramLM` with a GPT-style `TransformerLM`: RoPE, RMSNorm, SwiGLU MLP, grouped-query
  attention, KV-cache, flash attention via `scaled_dot_product_attention`.
- Built a real data pipeline (`data.py`) supporting single-file and multi-file corpora, and
  checkpoint resumption in `train.py`.
- Added best-checkpoint tracking, CSV loss logging, repetition-penalty sampling, residual init
  scaling, streaming generation, `torch.compile` support, and an ETA estimate during training.
- Added supporting tools: `bench.py` (throughput), `evaluate.py` (loss/perplexity), `compare.py`
  (side-by-side checkpoint comparison), `plot.py` (ASCII loss curves), `show_tokens.py`
  (tokenization visualizer).
- Set up daily automated commit/push via a local Windows Task Scheduler job
  (`daily_push.ps1` + `basic-llm daily push` task) — commits and pushes any local changes
  to GitHub at 11 PM, no-ops if nothing changed.
