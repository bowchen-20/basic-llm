import csv
import math
import os
import time
import torch
from tokenizer import CharTokenizer, BPETokenizer
from model import TransformerLM, ModelEMA
from config import ModelConfig, TrainConfig
from data import build_datasets, make_loader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
mcfg = ModelConfig(
    n_embd      = 128,
    n_head      = 4,
    n_layer     = 4,
    block_size  = 128,
    dropout     = 0.0,
    use_rope    = True,
    use_rmsnorm = True,
    use_swiglu  = True,
)

tcfg = TrainConfig(
    data_path        = "data/input.txt",
    out_path         = "checkpoint.pt",
    tokenizer_type   = "char",   # switch to "bpe" for subword tokenization
    bpe_vocab_size   = 500,
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

if tcfg.seed >= 0:
    torch.manual_seed(tcfg.seed)
    print(f"Seed: {tcfg.seed}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
try:
    with open(tcfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    text = "hello world " * 2000

if tcfg.tokenizer_type == "bpe":
    print(f"Training BPETokenizer to vocab_size={tcfg.bpe_vocab_size} ...")
    tokenizer = BPETokenizer()
    tokenizer.train(text, tcfg.bpe_vocab_size)
else:
    tokenizer = CharTokenizer(text)

mcfg.vocab_size = tokenizer.vocab_size
print(f"Tokenizer: {tcfg.tokenizer_type}  vocab_size={mcfg.vocab_size}")

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
optimizer = model.configure_optimizer(
    lr           = tcfg.lr_max,
    weight_decay = tcfg.weight_decay,
)
scaler = torch.amp.GradScaler(enabled=USE_AMP)

# ---------------------------------------------------------------------------
# Resume from checkpoint if one exists
# ---------------------------------------------------------------------------
start_step = 0
_resumed_ckpt = None
if os.path.exists(tcfg.out_path):
    print(f"Resuming from {tcfg.out_path} ...")
    _resumed_ckpt = torch.load(tcfg.out_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(_resumed_ckpt["model"])
    optimizer.load_state_dict(_resumed_ckpt["optimizer"])
    start_step = _resumed_ckpt.get("step", 0) + 1
    print(f"Resumed at step {start_step}")
else:
    print(f"Parameters: {model.num_params():,}  |  device: {DEVICE}  |  dtype: {tcfg.dtype}")

if tcfg.compile and hasattr(torch, "compile"):
    model = torch.compile(model)
    print("torch.compile() enabled")

ema: ModelEMA | None = None
if tcfg.ema_decay > 0:
    if _resumed_ckpt is not None and "ema" in _resumed_ckpt:
        ema = ModelEMA.from_state(model, _resumed_ckpt["ema"])
    else:
        ema = ModelEMA(model, tcfg.ema_decay)
    print(f"EMA: decay={tcfg.ema_decay}")

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


last_grad_norm = 0.0
best_val_loss  = float("inf")
best_path      = tcfg.out_path.replace(".pt", "_best.pt")
log_path       = tcfg.out_path.replace(".pt", "_log.csv")
t0             = time.time()

with open(log_path, "w", newline="", encoding="utf-8") as _f:
    csv.writer(_f).writerow(["step", "lr", "train_loss", "val_loss", "grad_norm"])

for step in range(start_step, tcfg.max_iters):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if step % tcfg.eval_interval == 0:
        _backup = ema.apply(model) if ema is not None else None
        losses = estimate_loss(model)
        if _backup is not None:
            ema.restore(model, _backup)
        elapsed = time.time() - t0
        steps_done = max(1, step - start_step)
        eta_mins = (tcfg.max_iters - step) * (elapsed / steps_done) / 60
        eta_str = f"  eta={eta_mins:.0f}min" if step > start_step else ""
        print(
            f"step {step:5d}  lr={lr:.2e}  "
            f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
            f"grad_norm={last_grad_norm:.3f}{eta_str}"
        )
        with open(log_path, "a", newline="", encoding="utf-8") as _f:
            csv.writer(_f).writerow([
                step, f"{lr:.6f}",
                f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                f"{last_grad_norm:.4f}",
            ])
        ckpt_payload = {
            "model":           model.state_dict(),
            "optimizer":       optimizer.state_dict(),
            "step":            step,
            "model_config":    mcfg.to_dict(),
            "tokenizer_state": tokenizer.get_state(),
            "train_config":    tcfg.__dict__,
        }
        if ema is not None:
            ckpt_payload["ema"] = ema.get_state()
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(ckpt_payload, best_path)
            print(f"  -> best val {best_val_loss:.4f}  saved {best_path}")
        if tcfg.checkpoint_interval > 0 and step > 0 and step % tcfg.checkpoint_interval == 0:
            step_path = tcfg.out_path.replace(".pt", f"_{step:06d}.pt")
            torch.save(ckpt_payload, step_path)
            print(f"  -> periodic checkpoint saved {step_path}")

    optimizer.zero_grad(set_to_none=True)
    for _ in range(tcfg.grad_accum_steps):
        x, y = next_batch()
        with torch.amp.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            _, loss, _ = model(x, y)
            loss = loss / tcfg.grad_accum_steps
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip).item()
    scaler.step(optimizer)
    scaler.update()
    if ema is not None:
        ema.update(model)

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
torch.save(
    {
        "model":            model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "step":             tcfg.max_iters - 1,
        "model_config":     mcfg.to_dict(),
        "tokenizer_state":  tokenizer.get_state(),
        "train_config":     tcfg.__dict__,
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
