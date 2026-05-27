from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size:  int   = 256
    n_embd:      int   = 128
    n_head:      int   = 4
    n_layer:     int   = 4
    block_size:  int   = 128
    dropout:     float = 0.0
    use_rope:    bool  = True   # rotary positional embeddings (vs learned)
    use_rmsnorm: bool  = True   # RMSNorm (vs LayerNorm)
    use_swiglu:  bool  = True   # SwiGLU MLP (vs GELU)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    # data
    data_path:      str = "data/input.txt"
    out_path:       str = "checkpoint.pt"
    tokenizer_type: str = "char"   # "char" or "bpe"
    bpe_vocab_size: int = 500      # only used when tokenizer_type == "bpe"

    # batching
    batch_size:       int = 32
    block_size:       int = 128
    grad_accum_steps: int = 1   # effective batch = batch_size * grad_accum_steps

    # schedule
    max_iters:    int   = 5000
    warmup_iters: int   = 200
    lr_max:       float = 3e-4
    lr_min:       float = 3e-5

    # optimiser
    weight_decay: float = 0.1
    grad_clip:    float = 1.0

    # evaluation
    eval_interval: int = 500
    eval_iters:    int = 100

    # precision: "float32" or "bfloat16"
    dtype: str = "float32"


# ---------------------------------------------------------------------------
# Architecture presets
# ---------------------------------------------------------------------------

def small_config(**kwargs) -> ModelConfig:
    """~1M params at vocab_size=100: good for character-level experiments."""
    return ModelConfig(n_embd=128, n_head=4, n_layer=4, block_size=128, **kwargs)


def medium_config(**kwargs) -> ModelConfig:
    """~8M params at vocab_size=500: reasonable for BPE on small corpora."""
    return ModelConfig(n_embd=256, n_head=8, n_layer=6, block_size=256, **kwargs)


def large_config(**kwargs) -> ModelConfig:
    """~50M params at vocab_size=4096: suitable for larger text datasets."""
    return ModelConfig(n_embd=512, n_head=8, n_layer=8, block_size=512, **kwargs)
