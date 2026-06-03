import math
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
        self.proj._is_residual = True
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
        self.down._is_residual = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


def make_mlp(cfg: ModelConfig) -> nn.Module:
    return SwiGLUMLP(cfg) if cfg.use_swiglu else GeLUMLP(cfg)


def drop_path(x: torch.Tensor, drop_rate: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly zero entire samples per residual branch."""
    if drop_rate == 0.0 or not training:
        return x
    keep = 1.0 - drop_rate
    mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
    return x * mask.float() / keep


# ---------------------------------------------------------------------------
# Causal self-attention with KV-cache + optional RoPE + Flash Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head    = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head   # 0 → standard MHA
        self.n_groups  = self.n_head // self.n_kv_head
        self.head_dim  = cfg.n_embd // cfg.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.dropout_p = cfg.dropout
        self.use_rope  = cfg.use_rope

        # Separate Q and KV projections so KV width can be smaller than Q (GQA).
        # When n_kv_head == n_head this is equivalent to a single fused projection.
        self.c_attn_q  = nn.Linear(cfg.n_embd, self.n_head    * self.head_dim, bias=False)
        self.c_attn_kv = nn.Linear(cfg.n_embd, 2 * self.n_kv_head * self.head_dim, bias=False)
        self.c_proj    = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_proj._is_residual = True
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        past_kv:   KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, T, C = x.shape
        q  = self.c_attn_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.c_attn_kv(x)
        k, v = kv.split(self.n_kv_head * self.head_dim, dim=2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        if self.use_rope and freqs_cis is not None:
            q = apply_rope(q, freqs_cis)
            k = apply_rope(k, freqs_cis)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv: KVCache | None = (k, v) if use_cache else None
        T_kv  = k.size(2)
        T_past = T_kv - T

        # Expand KV heads to match Q heads for grouped-query attention.
        # When n_groups == 1 this is a no-op (standard MHA).
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

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
    def __init__(self, cfg: ModelConfig, drop_path_rate: float = 0.0):
        super().__init__()
        self.ln1  = make_norm(cfg.n_embd, cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = make_norm(cfg.n_embd, cfg)
        self.mlp  = make_mlp(cfg)
        self.drop_path_rate = drop_path_rate

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
        x = x + drop_path(attn_out, self.drop_path_rate, self.training)
        x = x + drop_path(self.mlp(self.ln2(x)), self.drop_path_rate, self.training)
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

        # Linearly increase drop-path rate from 0 to cfg.drop_path_rate across layers,
        # as in the original Stochastic Depth paper (Huang et al. 2016).
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.n_layer).tolist()
        self.blocks = nn.ModuleList([TransformerBlock(cfg, dpr[i]) for i in range(cfg.n_layer)])
        self.ln_f   = make_norm(cfg.n_embd, cfg)
        self.head   = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # GPT-2 style: scale residual projections by 1/sqrt(2 * n_layer) so
            # the residual stream variance stays ~1 regardless of depth.
            std = 0.02
            if getattr(module, "_is_residual", False):
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, std=std)
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
        use_gc = self.cfg.use_grad_checkpoint and self.training and not use_cache
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if use_gc:
                # Gradient checkpointing: recompute activations in backward instead
                # of storing them, trading compute for peak memory.
                h, present_kv = torch.utils.checkpoint.checkpoint(
                    block, h, freqs_cis, past_kv, use_cache, use_reentrant=False
                )
            else:
                h, present_kv = block(h, freqs_cis=freqs_cis, past_kv=past_kv, use_cache=use_cache)
            present_kvs.append(present_kv)

        logits = self.head(self.ln_f(h))

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss, (present_kvs if use_cache else None)

    def _sample_token(
        self,
        logits: torch.Tensor,
        context: torch.Tensor,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        rep_penalty: float,
    ) -> torch.Tensor:
        logits = logits.clone()
        if rep_penalty != 1.0:
            for token_id in set(context[0].tolist()):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= rep_penalty
                else:
                    logits[0, token_id] *= rep_penalty
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            logits = logits.scatter(-1, sorted_indices, sorted_logits)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        rep_penalty: float = 1.0,
        use_cache: bool = True,
    ) -> torch.Tensor:
        if use_cache:
            logits, _, past_kvs = self(x, use_cache=True)
            next_tok = self._sample_token(logits[:, -1, :], x, temperature, top_k, top_p, rep_penalty)
            x = torch.cat([x, next_tok], dim=1)
            for _ in range(max_new_tokens - 1):
                logits, _, past_kvs = self(next_tok, past_key_values=past_kvs, use_cache=True)
                next_tok = self._sample_token(logits[:, -1, :], x, temperature, top_k, top_p, rep_penalty)
                x = torch.cat([x, next_tok], dim=1)
        else:
            for _ in range(max_new_tokens):
                x_cond = x[:, -self.cfg.block_size:]
                logits, _, _ = self(x_cond)
                next_tok = self._sample_token(logits[:, -1, :], x, temperature, top_k, top_p, rep_penalty)
                x = torch.cat([x, next_tok], dim=1)
        return x

    @torch.no_grad()
    def generate_stream(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        rep_penalty: float = 1.0,
        use_cache: bool = True,
    ):
        """Yield one token tensor at a time for character-by-character streaming."""
        if use_cache:
            logits, _, past_kvs = self(x, use_cache=True)
            next_tok = self._sample_token(logits[:, -1, :], x, temperature, top_k, top_p, rep_penalty)
            x = torch.cat([x, next_tok], dim=1)
            yield next_tok
            for _ in range(max_new_tokens - 1):
                logits, _, past_kvs = self(next_tok, past_key_values=past_kvs, use_cache=True)
                next_tok = self._sample_token(logits[:, -1, :], x, temperature, top_k, top_p, rep_penalty)
                x = torch.cat([x, next_tok], dim=1)
                yield next_tok
        else:
            for _ in range(max_new_tokens):
                x_cond = x[:, -self.cfg.block_size:]
                logits, _, _ = self(x_cond)
                next_tok = self._sample_token(logits[:, -1, :], x, temperature, top_k, top_p, rep_penalty)
                x = torch.cat([x, next_tok], dim=1)
                yield next_tok


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


# ---------------------------------------------------------------------------
# Exponential Moving Average of model weights
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Maintains a shadow copy of model parameters updated as an exponential
    moving average after every optimizer step.

    EMA weights tend to generalise slightly better than the raw last checkpoint
    because they smooth out the noise inherent in stochastic gradient updates.
    Call update() after every optimiser step, then use apply()/restore() to
    temporarily swap in the EMA weights for evaluation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            name: p.data.clone() for name, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Copy EMA weights into model; return original weights for restore()."""
        backup = {name: p.data.clone() for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Put original weights back after apply()."""
        for name, p in model.named_parameters():
            p.data.copy_(backup[name])

    def get_state(self) -> dict:
        return {
            "decay":  self.decay,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    @classmethod
    def from_state(cls, model: nn.Module, state: dict) -> "ModelEMA":
        ema = cls.__new__(cls)
        ema.decay = state["decay"]
        device = next(model.parameters()).device
        ema.shadow = {k: v.to(device) for k, v in state["shadow"].items()}
        return ema
