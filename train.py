import math
import os
import torch
from tokenizer import CharTokenizer
from model import TransformerLM
from config import ModelConfig, TrainConfig
from data import build_datasets, make_loader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
mcfg = ModelConfig(
    n_embd     = 128,
    n_head     = 4,
    n_layer    = 4,
    block_size = 128,
    dropout    = 0.0,
    use_rope   = True,
)

tcfg = TrainConfig(
    data_path        = "data/input.txt",
    out_path         = "checkpoint.pt",
    batch_size       = 32,
    block_size       = 128,
    grad_accum_steps = 1,
    max_iters        = 5000,
    warmup_iters     = 200,
    lr_max           = 3e-4,
    lr_min           = 3e-5,
    weight_decay     = 0.1,
    grad_clip        = 1.0,
    eval_interval    = 500,
    eval_iters       = 100,
    dtype            = "float32",
)

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
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

train_ds, val_ds = build_datasets(text, tokenizer, tcfg.block_size)
train_loader = make_loader(train_ds, tcfg.batch_size, shuffle=True)
val_loader   = make_loader(val_ds,   tcfg.batch_size, shuffle=False)


@torch.no_grad()
def estimate_loss(model: TransformerLM) -> dict[str, float]:
    model.eval()
    out = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses: list[float] = []
        for i, (x, y) in enumerate(loader):
            if i >= tcfg.eval_iters:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
                _, loss, _ = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def get_lr(step: int) -> float:
    if step < tcfg.warmup_iters:
        return tcfg.lr_max * step / tcfg.warmup_iters
    t = (step - tcfg.warmup_iters) / max(1, tcfg.max_iters - tcfg.warmup_iters)
    return tcfg.lr_min + 0.5 * (tcfg.lr_max - tcfg.lr_min) * (1.0 + math.cos(math.pi * t))


# ---------------------------------------------------------------------------
# Model + optimiser
# ---------------------------------------------------------------------------
model = TransformerLM(mcfg).to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = tcfg.lr_max,
    betas        = (0.9, 0.95),
    weight_decay = tcfg.weight_decay,
)
scaler = torch.amp.GradScaler(enabled=USE_AMP)

# ---------------------------------------------------------------------------
# Resume from checkpoint if one exists
# ---------------------------------------------------------------------------
start_step = 0
if os.path.exists(tcfg.out_path):
    print(f"Resuming from {tcfg.out_path} ...")
    ckpt = torch.load(tcfg.out_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt.get("step", 0) + 1
    print(f"Resumed at step {start_step}")
else:
    print(f"Parameters: {model.num_params():,}  |  device: {DEVICE}  |  dtype: {tcfg.dtype}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
train_iter = iter(train_loader)

def next_batch() -> tuple[torch.Tensor, torch.Tensor]:
    global train_iter
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    return x.to(DEVICE), y.to(DEVICE)


for step in range(start_step, tcfg.max_iters):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if step % tcfg.eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step {step:5d}  lr={lr:.2e}  "
            f"train={losses['train']:.4f}  val={losses['val']:.4f}"
        )

    optimizer.zero_grad(set_to_none=True)
    for _ in range(tcfg.grad_accum_steps):
        x, y = next_batch()
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
        "model":           model.state_dict(),
        "optimizer":       optimizer.state_dict(),
        "step":            tcfg.max_iters - 1,
        "model_config":    mcfg.to_dict(),
        "tokenizer_vocab": tokenizer.vocab,
        "train_config":    tcfg.__dict__,
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
