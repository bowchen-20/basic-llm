from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    vocab_size: int  = 256
    n_embd:     int  = 128
    n_head:     int  = 4
    n_layer:    int  = 4
    block_size: int  = 128
    dropout:    float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    # data
    data_path: str = "data/input.txt"
    out_path:  str = "checkpoint.pt"

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
