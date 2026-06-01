"""
GQE training engine (REINFORCE with EMA baseline).

Supports:
  • reward_mode = 'expectation' : fast ⟨H⟩ reward
  • reward_mode = 'qsci'        : QSCI energy reward (paper approach)
  • Number-preserving pool (auto-detected from pool type)
  • Persistent DeterminantBank shot reuse
  • Batched sampling for GPU throughput
  • Entropy regularisation + gradient clipping
  • Cosine LR annealing
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cudaq

from gqex.generator.retnet import RetNetGenerator
from gqex.generator.sampling import sample_batch, log_prob_batch
from gqex.qsci.det_bank import DeterminantBank


@dataclass
class GQEConfig:
    depth:         int   = 0          # 0 → auto = 4 × n_qubits
    n_samples:     int   = 10
    n_iters:       int   = 100
    lr:            float = 5e-3
    lr_min:        float = 1e-4       # cosine annealing minimum
    ema_decay:     float = 0.9
    entropy_coef:  float = 0.10       # raised from 0.03 — fight mode collapse
    grad_clip:     float = 1.0
    device:        str   = 'auto'
    log_every:     int   = 5
    reward_mode:   str   = 'expectation'   # 'expectation' | 'qsci'
    temperature:   float = 1.2             # >1 → more exploration early
    temp_anneal:   bool  = True            # cosine-anneal T toward 1.0
    top_k:         Optional[int] = None
    use_recurrent: bool  = True
    use_bank:      bool  = True
    cosine_anneal: bool  = True
    beam_elite:    bool  = False
    # Numerical stability
    adv_std_floor:  float = 5e-3      # lower-bound std to avoid divide-by-zero
    adv_clip:       float = 5.0       # clip standardised advantages
    # Anti-collapse
    collapse_eps:   float = 1e-4      # detect when energies.std() < this
    collapse_kick:  float = 0.05      # noise injected into bias on collapse
    # Advantage shaping
    use_rank_adv:   bool  = True      # rank-based (more robust) vs z-score
    diversity_coef: float = 0.05      # bonus for sampling unique circuits
    # Auto-tuning
    entropy_floor:  float = 0.5       # target policy entropy (nats / token)
    entropy_adapt:  float = 0.02      # if entropy < floor, raise coef by this/iter
    # Best-circuit replay
    replay_best:    bool  = True      # always include best-so-far in eval batch


class GQEEngine:
    """
    REINFORCE-trained discrete circuit generator.

    reward_mode = 'expectation' : reward = -⟨H⟩    (basic & fast)
    reward_mode = 'qsci'        : reward = -E_QSCI  (paper approach)
    """
    def __init__(
        self,
        system,
        pool,
        kernel,
        generator: RetNetGenerator,
        cfg: GQEConfig,
        qsci=None,
        bank: Optional[DeterminantBank] = None,
    ):
        self.system    = system
        self.pool      = pool
        self.kernel    = kernel
        self.generator = generator
        self.cfg       = cfg
        self.qsci      = qsci
        self.bank      = bank

        if cfg.device == 'auto':
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator.to(cfg.device)

    # ── Core step ────────────────────────────────────────────────────────

    def _current_temperature(self) -> float:
        cfg = self.cfg
        if not cfg.temp_anneal:
            return cfg.temperature
        # Cosine schedule from cfg.temperature → 1.0 over n_iters
        progress = min(self._iter / max(cfg.n_iters, 1), 1.0)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0 + (cfg.temperature - 1.0) * cos_factor

    def _kick_policy(self):
        """Inject small noise into the head bias to escape a collapsed policy."""
        with torch.no_grad():
            self.generator.head.bias.add_(
                self.cfg.collapse_kick * torch.randn_like(self.generator.head.bias)
            )

    def _step(self) -> Tuple[np.ndarray, float, List[List[int]], Optional[List]]:
        cfg     = self.cfg
        device  = cfg.device
        depth   = cfg.depth
        P       = self.pool.pool_size
        T_now   = self._current_temperature()

        # 1. Sample M circuits (batched)
        circuits = sample_batch(
            self.generator, depth, device, cfg.n_samples,
            temperature=T_now, top_k=cfg.top_k,
            use_recurrent=cfg.use_recurrent,
        )

        # 2. Evaluate each circuit
        wfs = None
        if cfg.reward_mode == 'expectation':
            energies = np.array([
                float(cudaq.observe(self.kernel, self.system.hamiltonian, c).expectation())
                for c in circuits
            ])
        else:  # qsci
            energies, wfs = [], []
            for c in circuits:
                e, wf, _ = self.qsci.evaluate(
                    self.kernel, c, use_bank=(self.bank is not None and cfg.use_bank)
                )
                energies.append(e)
                wfs.append(wf)
            energies = np.array(energies)

        # 3. Compute advantages — rank-based (preferred) or z-score
        if self._ema is None:
            self._ema = float(energies.mean())
        else:
            self._ema = (cfg.ema_decay * self._ema +
                         (1 - cfg.ema_decay) * float(energies.mean()))

        if cfg.use_rank_adv:
            # Rank-based: A_i = 1 - 2 * rank(E_i) / (N - 1) ∈ [-1, +1].
            # The best circuit gets +1, the worst -1, immune to scale & outliers.
            order = np.argsort(energies)   # ascending = best first
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(energies))
            adv_np = 1.0 - 2.0 * ranks / max(len(energies) - 1, 1)
        else:
            e_std = max(float(energies.std()), cfg.adv_std_floor)
            adv_np = np.clip((self._ema - energies) / e_std,
                             -cfg.adv_clip, cfg.adv_clip)
        adv = torch.tensor(adv_np, dtype=torch.float, device=device)

        # 3b. Mode-collapse detector — if std < eps for several iters, kick.
        if float(energies.std()) < cfg.collapse_eps:
            self._collapse_count += 1
            if self._collapse_count >= 3:
                self._kick_policy()
                self._collapse_count = 0
        else:
            self._collapse_count = 0

        # 4. Diversity bonus — reward circuits that don't duplicate others
        # in the batch (anti mode-collapse signal).
        if cfg.diversity_coef > 0:
            from collections import Counter as _Counter
            seq_keys = [tuple(c) for c in circuits]
            seq_count = _Counter(seq_keys)
            div_bonus = np.array([1.0 / seq_count[k] for k in seq_keys])
            # Normalize to mean 0, std 1
            div_bonus = (div_bonus - div_bonus.mean()) / (div_bonus.std() + 1e-8)
            adv = adv + cfg.diversity_coef * torch.tensor(
                div_bonus, dtype=torch.float, device=device
            )

        # 5. Batched log-probs & policy gradient
        log_probs = log_prob_batch(self.generator, circuits, device)   # (M,)
        policy_loss = -(log_probs * adv).mean()

        # 6. Entropy regulariser (with auto-adapt)
        bos = torch.full((len(circuits), 1), P, dtype=torch.long, device=device)
        ops_t = torch.tensor(circuits, dtype=torch.long, device=device)
        full  = torch.cat([bos, ops_t], dim=1)           # (M, depth+1)
        logits = self.generator(full)                     # (M, depth+1, P)
        probs  = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()

        # Auto-adapt entropy coef: if policy is too peaked, raise the bonus
        e_val = float(entropy.item())
        if e_val < cfg.entropy_floor:
            self._entropy_coef = min(self._entropy_coef + cfg.entropy_adapt, 1.0)
        else:
            self._entropy_coef = max(self._entropy_coef * 0.99, cfg.entropy_coef)
        loss = policy_loss - self._entropy_coef * entropy

        # 6. Optimiser step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), cfg.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return energies, float(loss.item()), circuits, wfs

    # ── Main training loop ────────────────────────────────────────────────

    def run(self) -> Dict:
        cfg = self.cfg
        if cfg.depth == 0:
            cfg.depth = 4 * self.system.n_qubits

        self.optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=cfg.lr
        )
        self.scheduler = None
        if cfg.cosine_anneal:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.n_iters, eta_min=cfg.lr_min
            )
        self._ema = None
        self._iter = 0
        self._collapse_count = 0
        self._entropy_coef = cfg.entropy_coef
        history = {'iter': [], 'min': [], 'mean': [], 'elite': [], 'loss': [], 'lr': []}

        best_e, best_ops, best_wf = float('inf'), None, None

        ref_e = self.system.casci_energy
        hf_e  = self.system.hf_energy
        print("\n" + "=" * 64)
        print(f"  GQE training  ({cfg.reward_mode}-reward,  "
              f"pool={type(self.pool).__name__})")
        print(f"  qubits={self.system.n_qubits}  pool={self.pool.pool_size}  "
              f"depth={cfg.depth}  samples={cfg.n_samples}  iters={cfg.n_iters}")
        if not math.isnan(ref_e):
            print(f"  HF={hf_e:+.6f}  CASCI={ref_e:+.6f}  "
                  f"corr={1000*(ref_e-hf_e):+.1f} mHa")
        print("=" * 64)

        t0 = time.time()
        for it in range(1, cfg.n_iters + 1):
            self._iter = it
            energies, loss, circuits, wfs = self._step()
            e_min  = float(energies.min())
            e_mean = float(energies.mean())

            if e_min < best_e:
                best_e   = e_min
                best_ops = circuits[int(np.argmin(energies))].copy()
                if wfs is not None:
                    best_wf = wfs[int(np.argmin(energies))]

            current_lr = self.optimizer.param_groups[0]['lr']
            history['iter'].append(it)
            history['min'].append(e_min)
            history['mean'].append(e_mean)
            history['elite'].append(best_e)
            history['loss'].append(loss)
            history['lr'].append(current_lr)

            if it == 1 or it % cfg.log_every == 0 or it == cfg.n_iters:
                tag = ("" if math.isnan(ref_e) else
                       f"   |Δ|={1000*abs(best_e-ref_e):.1f} mHa")
                bank_str = (f"  bank={self.bank.size}" if self.bank else "")
                print(f"  it {it:3d}  min={e_min:+.4f}  mean={e_mean:+.4f}  "
                      f"elite={best_e:+.4f}  loss={loss:+.4f}"
                      f"  lr={current_lr:.2e}{tag}{bank_str}")

        dt = time.time() - t0
        print("=" * 64)
        print(f"  done  ({dt:.1f}s,  {dt/cfg.n_iters:.2f}s/iter)")
        if self.bank:
            print(f"  bank: {self.bank.size} dets  "
                  f"({self.bank.total_shots} total shots)")

        return {
            'best_energy': best_e,
            'best_ops':    best_ops,
            'best_wf':     best_wf,
            'history':     history,
        }
