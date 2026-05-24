import math
import torch
from tokenizer import CharTokenizer
from model import TransformerLM
from config import ModelConfig, TrainConfig

# ---------------------------------------------------------------------------
# Config — edit these to change the run
# ---------------------------------------------------------------------------
mcfg = ModelConfig(
    n_embd     = 128,
    n_head     = 4,
    n_layer    = 4,
    block_size = 128,
    dropout    = 0.0,
)

tcfg = TrainConfig(
    data_path        = "data/input.txt",
    out_path         = "checkpoint.pt",
    batch_size       = 32,
    block_size       = 128,
    grad_accum_steps = 1,       # increase to simulate larger effective batch
    max_iters        = 5000,
    warmup_iters     = 200,
    lr_max           = 3e-4,
    lr_min           = 3e-5,
    weight_decay     = 0.1,
    grad_clip        = 1.0,
    eval_interval    = 500,
    eval_iters       = 100,
    dtype            = "float32",  # "bfloat16" for GPU speedup
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if tcfg.dtype == "bfloat16" else torch.float32
USE_AMP   = (tcfg.dtype == "bfloat16") and (DEVICE == "cuda")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
try:
    with open(tcfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    text = "hello world " * 2000

tokenizer = CharTokenizer(text)
mcfg.vocab_size = tokenizer.vocab_size

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    d  = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - tcfg.block_size, (tcfg.batch_size,))
    x  = torch.stack([d[i : i + tcfg.block_size]     for i in ix])
    y  = torch.stack([d[i + 1 : i + tcfg.block_size + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model: TransformerLM) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            x, y = get_batch(split)
            with torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step: int) -> float:
    """Linear warmup then cosine decay."""
    if step < tcfg.warmup_iters:
        return tcfg.lr_max * step / tcfg.warmup_iters
    t = (step - tcfg.warmup_iters) / max(1, tcfg.max_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * t))
    return tcfg.lr_min + coeff * (tcfg.lr_max - tcfg.lr_min)


# ---------------------------------------------------------------------------
# Model + optimiser
# ---------------------------------------------------------------------------
model = TransformerLM(mcfg).to(DEVICE)
print(f"Parameters: {model.num_params():,}  |  device: {DEVICE}  |  dtype: {tcfg.dtype}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = tcfg.lr_max,
    betas        = (0.9, 0.95),
    weight_decay = tcfg.weight_decay,
)
scaler = torch.amp.GradScaler(enabled=USE_AMP)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for step in range(tcfg.max_iters):
    # Update LR
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Periodic eval
    if step % tcfg.eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step {step:5d}  lr={lr:.2e}  "
            f"train={losses['train']:.4f}  val={losses['val']:.4f}"
        )

    # Gradient accumulation: accumulate over grad_accum_steps mini-batches
    # before stepping the optimiser, simulating a larger effective batch.
    optimizer.zero_grad(set_to_none=True)
    for micro in range(tcfg.grad_accum_steps):
        x, y = get_batch("train")
        with torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            _, loss, _ = model(x, y)
            loss = loss / tcfg.grad_accum_steps
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
torch.save(
    {
        "model":          model.state_dict(),
        "model_config":   mcfg.to_dict(),
        "tokenizer_vocab": tokenizer.vocab,
        "train_config":   tcfg.__dict__,
    },
    tcfg.out_path,
)
print(f"Saved checkpoint to {tcfg.out_path}")

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
model.eval()
ctx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
sample = tokenizer.decode(
    model.generate(ctx, 200, temperature=0.8, top_k=40, use_cache=False)[0].tolist()
)
print("\n--- sample output ---")
print(sample)
