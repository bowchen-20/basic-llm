import torch
from tokenizer import CharTokenizer
from model import BigramLM

# --- config ---
BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3000
LEARNING_RATE = 1e-2
EVAL_INTERVAL = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- data ---
# Drop a text file at data/input.txt to train on your own corpus.
# Falls back to a tiny built-in sample so the script always runs.
try:
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    text = "hello world " * 500

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(50)
        for k in range(50):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --- train ---
model = BigramLM(tokenizer.vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for step in range(MAX_ITERS):
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss(model)
        print(f"step {step}: train={losses['train']:.4f}  val={losses['val']:.4f}")

    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "bigram_model.pt")
print("Saved model to bigram_model.pt")

# --- sample ---
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
sample = tokenizer.decode(model.generate(context, 200)[0].tolist())
print("\n--- sample output ---")
print(sample)
