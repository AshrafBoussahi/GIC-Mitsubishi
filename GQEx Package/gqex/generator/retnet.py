"""
RetNet-based circuit generator with scaling and recurrent inference.

Priority-8 improvement: scaled RetNet with recurrent (O(1) per step)
inference mode, enabling much longer circuit depths at 40 qubits without
the quadratic memory cost of full self-attention.

Architecture:
  • Multi-head retention layers (parallel mode for training)
  • Recurrent mode for O(length) inference (vs O(length²) in full-attention)
  • Optional multi-scale heads
  • LayerNorm + FFN sublayers
  • Gated MLP for improved expressivity
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RetNetConfig:
    pool_size:   int
    hidden_dim:  int  = 128
    n_layers:    int  = 4
    n_heads:     int  = 4          # multi-head retention
    ffn_mult:    int  = 4          # FFN hidden = hidden_dim * ffn_mult
    max_len:     int  = 256
    dropout:     float = 0.0
    gated_mlp:   bool  = True      # gated MLP (improves training stability)


# ─── Multi-head retention layer ───────────────────────────────────────────────

class MultiScaleRetention(nn.Module):
    """
    Multi-head retention as in Sun et al. 2023.

    Parallel mode  : O(T²·D/H) — used during training.
    Recurrent mode : O(D²/H)   — used during sampling (no quadratic growth).
    """
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        self.head_dim   = hidden_dim // n_heads

        self.q_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.g_proj  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm_q  = nn.LayerNorm(self.head_dim)
        self.norm_k  = nn.LayerNorm(self.head_dim)

        # Per-head decay rates γ_i: log(1 - 2^{-5 - i / n_heads})
        gammas = 1 - 2 ** (-5 - torch.arange(n_heads, dtype=torch.float) / n_heads)
        self.register_buffer('gammas', gammas)           # (H,)
        self.dropout = nn.Dropout(dropout)

    # ── Parallel (training) mode ──────────────────────────────────────────

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        B, T, D = x.shape
        H  = self.n_heads
        Dh = self.head_dim

        Q = self.norm_q(self.q_proj(x).view(B, T, H, Dh))   # (B,T,H,Dh)
        K = self.norm_k(self.k_proj(x).view(B, T, H, Dh))
        V = self.v_proj(x).view(B, T, H, Dh)
        G = torch.sigmoid(self.g_proj(x).view(B, T, H, Dh))

        # Retention mask D[i,j] = γ^(i-j) if i>=j else 0  → (1,H,T,T)
        idx  = torch.arange(T, device=x.device)
        diff = (idx[:, None] - idx[None, :]).float()   # (T,T)
        mask = torch.where(
            diff.unsqueeze(0) >= 0,
            self.gammas[:, None, None] ** diff.clamp(min=0).unsqueeze(0),
            torch.zeros_like(diff.unsqueeze(0)),
        ).unsqueeze(0)  # (1,H,T,T)

        # Q,K,V: reshape to (B,H,T,Dh)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        G = G.permute(0, 2, 1, 3)

        attn  = (Q @ K.transpose(-2, -1)) / math.sqrt(Dh)   # (B,H,T,T)
        ret   = (attn * mask) @ V                             # (B,H,T,Dh)
        ret   = ret * G                                       # gated
        ret   = ret.permute(0, 2, 1, 3).reshape(B, T, D)     # (B,T,D)
        return self.out_proj(self.dropout(ret))

    # ── Recurrent (inference) mode ────────────────────────────────────────

    def forward_recurrent(self, x_t: torch.Tensor,
                          state: Optional[torch.Tensor]) -> tuple:
        """Single-step recurrent retention.

        x_t   : (B, D)
        state : (B, H, Dh, Dh) or None
        Returns (out, new_state).
        """
        B, D = x_t.shape
        H, Dh = self.n_heads, self.head_dim

        # Apply norm while last dim == Dh, then reshape for matmul.
        q = self.norm_q(self.q_proj(x_t).view(B, H, Dh))   # (B, H, Dh)
        k = self.norm_k(self.k_proj(x_t).view(B, H, Dh))   # (B, H, Dh)
        v = self.v_proj(x_t).view(B, H, Dh)                  # (B, H, Dh)
        g = torch.sigmoid(self.g_proj(x_t).view(B, H, Dh))  # (B, H, Dh)

        # Outer product k ⊗ v → state update matrix (B, H, Dh, Dh).
        # k.unsqueeze(-1): (B,H,Dh,1)  *  v.unsqueeze(-2): (B,H,1,Dh)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)   # (B, H, Dh, Dh)

        # Update recurrent state: S_t = γ * S_{t-1} + k_t ⊗ v_t
        if state is None:
            new_state = kv
        else:
            new_state = self.gammas.view(1, H, 1, 1) * state + kv

        # Output: q_t^T S_t  →  (B,H,1,Dh) @ (B,H,Dh,Dh) = (B,H,1,Dh)
        out = (q.unsqueeze(-2) @ new_state).squeeze(-2)  # (B, H, Dh)
        out = out * g / math.sqrt(Dh)
        out = out.reshape(B, D)
        return self.out_proj(out), new_state

    def forward(self, x: torch.Tensor, recurrent: bool = False,
                state=None):
        if recurrent:
            return self.forward_recurrent(x, state)
        return self.forward_parallel(x), None


# ─── FFN sublayer ─────────────────────────────────────────────────────────────

class GatedFFN(nn.Module):
    """Gated FFN (SwiGLU-style)."""
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hid = dim * mult
        self.w1 = nn.Linear(dim, hid, bias=False)
        self.w2 = nn.Linear(dim, hid, bias=False)
        self.w3 = nn.Linear(hid, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


class StandardFFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hid = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid, dim)
        )

    def forward(self, x):
        return self.net(x)


# ─── RetNet block ────────────────────────────────────────────────────────────

class RetNetBlock(nn.Module):
    def __init__(self, cfg: RetNetConfig):
        super().__init__()
        self.retention = MultiScaleRetention(
            cfg.hidden_dim, cfg.n_heads, cfg.dropout
        )
        FFN = GatedFFN if cfg.gated_mlp else StandardFFN
        self.ffn   = FFN(cfg.hidden_dim, cfg.ffn_mult, cfg.dropout)
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, x, recurrent=False, state=None):
        ret_out, new_state = self.retention(self.norm1(x),
                                            recurrent=recurrent, state=state)
        x = x + ret_out
        x = x + self.ffn(self.norm2(x))
        return x, new_state


# ─── Full RetNet generator ────────────────────────────────────────────────────

class RetNetGenerator(nn.Module):
    """
    RetNet sequence model for discrete circuit generation.

    Input token range:  0 .. pool_size-1   (gate ops)
    Token pool_size     = BOS token
    Output logits:      0 .. pool_size-1   (next op probabilities)
    """
    def __init__(self, cfg: RetNetConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_dim
        self.embed  = nn.Embedding(cfg.pool_size + 1, D)   # +1 for BOS
        self.pos    = nn.Embedding(cfg.max_len, D)
        self.blocks = nn.ModuleList([RetNetBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm   = nn.LayerNorm(D)
        self.head   = nn.Linear(D, cfg.pool_size, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    # ── Parallel forward (training) ───────────────────────────────────────

    def forward(self, ops: torch.Tensor) -> torch.Tensor:
        """ops: (B, T) long → logits (B, T, pool_size)"""
        B, T = ops.shape
        pos  = torch.arange(T, device=ops.device)
        x    = self.embed(ops) + self.pos(pos[:min(T, self.cfg.max_len)]).unsqueeze(0)
        for block in self.blocks:
            x, _ = block(x, recurrent=False)
        return self.head(self.norm(x))

    # ── Recurrent forward (inference) ────────────────────────────────────

    def step(self, token: torch.Tensor, states: Optional[List],
             pos: int) -> tuple:
        """Single-token step for recurrent generation.

        token  : (B,) long
        states : list of per-layer states or None
        pos    : current position index

        Returns (logits (B, pool_size), new_states).
        """
        B = token.shape[0]
        pos_t = torch.tensor([pos], device=token.device)
        x = (self.embed(token) +
             self.pos(pos_t.clamp(max=self.cfg.max_len - 1)).squeeze(0))  # (B,D)

        new_states = []
        for i, block in enumerate(self.blocks):
            s = states[i] if states is not None else None
            x, new_s = block(x, recurrent=True, state=s)
            new_states.append(new_s)

        logits = self.head(self.norm(x))   # (B, pool_size)
        return logits, new_states

    # ── Warm-start biasing ────────────────────────────────────────────────

    def warm_start_bias(self, identity_bias: float = 0.5,
                        hf_x_qubits: Optional[List[int]] = None,
                        x_offset: int = 1, hf_bias: float = 0.3,
                        pool=None, n_alpha: int = 0, n_beta: int = 0,
                        boundary_bias: float = 0.8):
        """Bias the output head toward identity + useful gates.

        Two modes:
          • If `pool` is a NumberPreservingPool: bias Givens rotations that
            cross the HOMO-LUMO boundary (one qubit occupied in HF, the other
            empty).  These are the *only* Givens that actually move amplitude.
          • Otherwise (StandardPool): bias X-on-HF-qubit indices as before.
        """
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[0] = identity_bias

            np_done = False
            try:
                from gqex.ansatz.gate_pool import NumberPreservingPool
                if pool is not None and isinstance(pool, NumberPreservingPool):
                    self._warm_start_np(pool, n_alpha, n_beta,
                                        bias=boundary_bias)
                    np_done = True
            except ImportError:
                pass

            if not np_done and hf_x_qubits is not None:
                # Standard pool: bias X on HF-occupied qubits
                for q in hf_x_qubits:
                    idx = x_offset + q
                    if 0 <= idx and idx < self.cfg.pool_size:
                        self.head.bias[idx] = hf_bias

    def _warm_start_np(self, pool, n_alpha: int, n_beta: int, bias: float):
        """NP-pool-specific warm start: favour Givens at the Fermi boundary.

        In JW interleaved α,β ordering qubit 2p = α_p, 2p+1 = β_p.  HF occupies
        α_0..α_{n_α-1} and β_0..β_{n_β-1}.  A Givens(i,j) is non-trivial iff
        exactly one of qubits i,j is occupied in HF (the |01⟩↔|10⟩ subspace).
        """
        n  = pool.n_qubits
        offs = pool.family_offsets()
        hf_occ = set()
        for p in range(n_alpha):
            hf_occ.add(2 * p)
        for p in range(n_beta):
            hf_occ.add(2 * p + 1)

        def useful(qa: int, qb: int) -> bool:
            return (qa in hf_occ) ^ (qb in hf_occ)

        # adjacent Givens
        if 'givens_adj_pos' in offs:
            for p in range(n - 1):
                if useful(p, p + 1):
                    self.head.bias[offs['givens_adj_pos'] + p] = bias
                    if 'givens_adj_neg' in offs:
                        self.head.bias[offs['givens_adj_neg'] + p] = bias

        # long-range Givens (skip-1)
        if 'givens_lr_pos' in offs:
            for p in range(n - 2):
                if useful(p, p + 2):
                    self.head.bias[offs['givens_lr_pos'] + p] = bias
                    if 'givens_lr_neg' in offs:
                        self.head.bias[offs['givens_lr_neg'] + p] = bias

        # double excitations (4-qubit windows)
        if 'dbl_exc_pos' in offs:
            for k in range(n // 4):
                window = (4*k, 4*k+1, 4*k+2, 4*k+3)
                n_occ_in = sum(1 for q in window if q in hf_occ)
                if n_occ_in in (1, 2, 3):   # window straddles boundary
                    self.head.bias[offs['dbl_exc_pos'] + k] = bias
                    if 'dbl_exc_neg' in offs:
                        self.head.bias[offs['dbl_exc_neg'] + k] = bias

    def prior_from_ops(self, ops: List[int], bias_strength: float = 1.0,
                       supervised_epochs: int = 0, lr: float = 1e-3):
        """Bias the policy toward producing the gate multiset present in `ops`.

        Two modes:
          • bias_strength > 0 (default): add log-count bias to head, fast O(1).
          • supervised_epochs > 0: fine-tune the network to autoregressively
            reproduce `ops` (proper imitation learning).  Slower but stronger.
        """
        if not ops:
            return

        if bias_strength > 0:
            from collections import Counter
            counts = Counter(ops)
            total = sum(counts.values())
            with torch.no_grad():
                for op, c in counts.items():
                    if 0 <= op < self.cfg.pool_size:
                        # log-frequency bias, scaled
                        self.head.bias[op] += bias_strength * math.log(1 + c)

        if supervised_epochs > 0:
            self._supervised_pretrain(ops, epochs=supervised_epochs, lr=lr)

    def _supervised_pretrain(self, ops: List[int], epochs: int, lr: float):
        """Briefly fine-tune the network to autoregressively reproduce `ops`."""
        P = self.cfg.pool_size
        device = next(self.parameters()).device
        bos = torch.tensor([[P]], device=device)
        x   = torch.tensor([ops], device=device)
        target = x[0]                                        # (T,)
        inputs = torch.cat([bos, x[:, :-1]], dim=1)          # (1, T)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for ep in range(epochs):
            opt.zero_grad()
            logits = self(inputs)[0]                          # (T, P)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            opt.step()
        self.eval()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
