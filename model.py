import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

KVCache = tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Rotary positional embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(head_dim: int, seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)   # complex64 (seq_len, head_dim/2)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: (B, n_head, T, head_dim)  |  freqs_cis: (T, head_dim/2) complex."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    fc = freqs_cis.view(1, 1, freqs_cis.shape[0], freqs_cis.shape[1])
    return torch.view_as_real(xc * fc).flatten(3).type_as(x)


# ---------------------------------------------------------------------------
# Normalisation layers
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root-mean-square layer norm (no mean subtraction, just scale by RMS).
    Simpler and slightly faster than LayerNorm; used in LLaMA/Mistral.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def make_norm(dim: int, cfg: ModelConfig) -> nn.Module:
    return RMSNorm(dim) if cfg.use_rmsnorm else nn.LayerNorm(dim)


# ---------------------------------------------------------------------------
# MLP variants
# ---------------------------------------------------------------------------

class GeLUMLP(nn.Module):
    """Standard GPT-2 MLP: Linear → GELU → Linear, 4× width."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward (LLaMA style): gate * up → down.

    Uses 8/3 × n_embd hidden width (rounded to nearest 64) so the total
    parameter count stays comparable to a 4× GELU MLP.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(cfg.n_embd * 8 / 3)
        hidden = (hidden + 63) // 64 * 64   # round up to multiple of 64
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up   = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def make_mlp(cfg: ModelConfig) -> nn.Module:
    return SwiGLUMLP(cfg) if cfg.use_swiglu else GeLUMLP(cfg)


# ---------------------------------------------------------------------------
# Causal self-attention with KV-cache + optional RoPE + Flash Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout_p = cfg.dropout
        self.use_rope  = cfg.use_rope

        self.c_attn    = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.c_proj    = nn.Linear(cfg.n_embd, cfg.n_embd,     bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        past_kv:   KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if self.use_rope and freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv: KVCache | None = (k, v) if use_cache else None
        T_kv  = k.size(2)
        T_past = T_kv - T
        dp = self.dropout_p if self.training else 0.0

        if T == 1:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=False)
        elif T_past == 0:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        else:
            rows = torch.arange(T,    device=x.device).unsqueeze(1)
            cols = torch.arange(T_kv, device=x.device).unsqueeze(0)
            mask = (cols <= rows + T_past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dp)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y)), present_kv


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1  = make_norm(cfg.n_embd, cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = make_norm(cfg.n_embd, cfg)
        self.mlp  = make_mlp(cfg)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        past_kv:   KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        attn_out, present_kv = self.attn(
            self.ln1(x), freqs_cis=freqs_cis, past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present_kv


# ---------------------------------------------------------------------------
# Full language model
# ---------------------------------------------------------------------------

class TransformerLM(nn.Module):
    """GPT-style causal language model. Architecture controlled by ModelConfig."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)

        if cfg.use_rope:
            freqs = precompute_rope_freqs(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("freqs_cis", freqs, persistent=False)
        else:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f   = make_norm(cfg.n_embd, cfg)
        self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def configure_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        """
        Split parameters into two groups: tensors with ndim >= 2 get weight
        decay; 1-D params (biases, norm scales) do not.  This follows the
        standard practice from GPT-2 / nanoGPT.
        """
        decay_params  = [p for p in self.parameters() if p.dim() >= 2]
        nodecay_params = [p for p in self.parameters() if p.dim() < 2]
        return torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas,
        )

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

        h = self.tok_emb(x)
        if self.cfg.use_rope:
            freqs_cis: torch.Tensor | None = self.freqs_cis[T_past : T_past + T]
        else:
            pos = torch.arange(T_past, T_past + T, device=x.device)
            h   = h + self.pos_emb(pos)
            freqs_cis = None
        h = self.drop(h)

        present_kvs: list[KVCache | None] = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, present_kv = block(h, freqs_cis=freqs_cis, past_kv=past_kv, use_cache=use_cache)
            present_kvs.append(present_kv)

        logits = self.head(self.ln_f(h))

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
        def sample(logits: torch.Tensor) -> torch.Tensor:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        if use_cache:
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


# ---------------------------------------------------------------------------
# BigramLM — kept as a reference baseline
# ---------------------------------------------------------------------------

class BigramLM(nn.Module):
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
            x = torch.cat([x, torch.multinomial(F.softmax(logits[:, -1, :], dim=-1), 1)], dim=1)
        return x
