import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

# Type alias for one layer's cached (key, value) tensors.
KVCache = tuple[torch.Tensor, torch.Tensor]


class BigramLM(nn.Module):
    """Kept for reference — predicts the next token from the current token only."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.embedding(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens, **_):
        for _ in range(max_new_tokens):
            logits, _ = self(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            x = torch.cat([x, torch.multinomial(probs, 1)], dim=1)
        return x


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional KV-cache for fast inference.

    During training (use_cache=False): standard causal mask, no cache.
    During decoding  (use_cache=True): Q is for new token(s) only; K/V from
    past tokens come from the cache and are concatenated before attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.c_attn   = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj   = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv: KVCache | None = (k, v) if use_cache else None
        T_kv   = k.size(2)
        T_past = T_kv - T  # number of cached tokens prepended

        scale = math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) / scale   # (B, n_head, T, T_kv)

        # Causal mask: position i (absolute index T_past+i) may only attend to
        # positions 0 … T_past+i.  When T==1 (single decode step) every position
        # in K is already in the past, so no masking is needed.
        if T > 1:
            rows = torch.arange(T,    device=x.device).unsqueeze(1)  # (T, 1)
            cols = torch.arange(T_kv, device=x.device).unsqueeze(0)  # (1, T_kv)
            causal = cols <= rows + T_past                             # (T, T_kv)
            att = att.masked_fill(~causal.view(1, 1, T, T_kv), float("-inf"))

        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y)), present_kv


class MLP(nn.Module):
    """Position-wise feed-forward: Linear → GELU → Linear, 4× width."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class TransformerBlock(nn.Module):
    """Pre-norm residual block: LN → Attn → +residual, LN → MLP → +residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.mlp  = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present_kv = self.attn(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """GPT-style causal language model built from a ModelConfig."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size,  cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying halves vocab-boundary parameters and improves perplexity.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[KVCache] | None]:
        B, T = x.shape
        T_past = past_key_values[0][0].size(2) if past_key_values is not None else 0
        assert T + T_past <= self.cfg.block_size

        positions = torch.arange(T_past, T_past + T, device=x.device)
        x = self.drop(self.tok_emb(x) + self.pos_emb(positions))

        present_kvs: list[KVCache | None] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            present_kvs.append(present_kv)

        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss, (present_kvs if use_cache else None)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Auto-regressively sample max_new_tokens tokens.

        use_cache=True  — fast: prefill prompt once, then decode one token at a
                          time reusing cached K/V (O(n) total work per step).
        use_cache=False — simple: re-runs the full context every step (O(n²)).
        """
        def sample(logits: torch.Tensor) -> torch.Tensor:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        if use_cache:
            # Prefill: encode the full prompt and seed the KV-cache.
            logits, _, past_kvs = self(x, use_cache=True)
            next_tok = sample(logits[:, -1, :])
            x = torch.cat([x, next_tok], dim=1)

            for _ in range(max_new_tokens - 1):
                logits, _, past_kvs = self(next_tok, past_key_values=past_kvs, use_cache=True)
                next_tok = sample(logits[:, -1, :])
                x = torch.cat([x, next_tok], dim=1)
        else:
            for _ in range(max_new_tokens):
                x_cond = x[:, -self.cfg.block_size:]
                logits, _, _ = self(x_cond)
                next_tok = sample(logits[:, -1, :])
                x = torch.cat([x, next_tok], dim=1)

        return x
