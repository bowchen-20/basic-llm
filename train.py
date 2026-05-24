import math
import torch
from tokenizer import CharTokenizer
from model import TransformerLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE    = 32
BLOCK_SIZE    = 128
MAX_ITERS     = 5000
EVAL_INTERVAL = 500
EVAL_ITERS    = 100

# Transformer architecture
N_EMBD  = 128
N_HEAD  = 4
N_LAYER = 4
DROPOUT = 0.0

# Learning rate: warm up for WARMUP_ITERS steps, then cosine decay to LR_MIN
LR_MAX    = 3e-4
LR_MIN    = 3e-5
WARMUP_ITERS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
try:
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    text = "hello world " * 2000

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model: TransformerLM) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step: int) -> float:
    """Cosine decay with linear warmup."""
    if step < WARMUP_ITERS:
        return LR_MAX * step / WARMUP_ITERS
    progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + coeff * (LR_MAX - LR_MIN)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
model = TransformerLM(
    vocab_size  = tokenizer.vocab_size,
    n_embd      = N_EMBD,
    n_head      = N_HEAD,
    n_layer     = N_LAYER,
    block_size  = BLOCK_SIZE,
    dropout     = DROPOUT,
).to(DEVICE)

print(f"Model parameters: {model.num_params():,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, betas=(0.9, 0.95), weight_decay=0.1)

for step in range(MAX_ITERS):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(
            f"step {step:5d}  lr={lr:.2e}  "
            f"train={losses['train']:.4f}  val={losses['val']:.4f}"
        )

    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.save(
    {"model": model.state_dict(), "tokenizer_vocab": tokenizer.vocab},
    "transformer_model.pt",
)
print("Saved model to transformer_model.pt")

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
sample = tokenizer.decode(model.generate(context, 200, temperature=0.8, top_k=40)[0].tolist())
print("\n--- sample output ---")
print(sample)
