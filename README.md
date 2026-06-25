# Basic LLM from Scratch

A GPT-style language model built from scratch in Python with PyTorch — started as a bigram model,
now a full transformer with RoPE, GQA, flash attention, and a BPE tokenizer.

See [CHANGELOG.md](CHANGELOG.md) for a running log of what's been added each session.

## Features

- **Tokenizer**: character-level or BPE (`tokenizer.py`)
- **Model** (`model.py`): RoPE, RMSNorm, SwiGLU MLP, grouped-query attention, KV-cache,
  flash attention via `scaled_dot_product_attention`, EMA of weights
- **Training** (`train.py`): cosine LR schedule with warmup, gradient accumulation, AMP,
  checkpoint resume, best-checkpoint tracking, CSV loss logging, `torch.compile` support
- **Inference** (`generate.py`): top-k / top-p / repetition-penalty sampling, streaming, interactive REPL
- **Tooling**: `evaluate.py` (loss/perplexity), `compare.py` (side-by-side checkpoints),
  `bench.py` (tokens/sec throughput), `inspect_model.py` (params/weight stats), `plot.py`
  (ASCII loss curves), `show_tokens.py` (tokenization visualizer)

## Usage

```bash
pip install -r requirements.txt

# drop a corpus at data/input.txt, then:
python train.py
python generate.py --interactive
```

## Roadmap

- [x] Character tokenizer → BPE tokenizer
- [x] Bigram baseline → GPT-style transformer
- [x] Training loop with checkpointing, logging, resumption
- [x] Sampling (top-k/top-p/rep-penalty) and streaming generation
- [x] Evaluation, benchmarking, and comparison tooling
- [ ] Multi-GPU / DDP training
- [ ] LoRA fine-tuning
- [ ] Web demo (Gradio/Flask) for interactive generation
