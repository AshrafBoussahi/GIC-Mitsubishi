"""
Pipeline configuration dataclasses.

All hyperparameters for the GQEx pipeline are centralised here.
Sensible defaults are provided for every option; override only what
you need.

Quick-start example:
    from gqex.pipeline.config import PipelineConfig, quick_config
    cfg = quick_config('small')   # 8–12 qubit fast demo
    cfg = quick_config('medium')  # 16–24 qubit standard
    cfg = quick_config('large')   # 32–40 qubit full pipeline
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from gqex.gqe.engine import GQEConfig
from gqex.qsci.evaluator import QSCIConfig
from gqex.generator.retnet import RetNetConfig
from gqex.forging.entanglement import ForgingConfig
from gqex.adapt.bootstrap import ADAPTConfig


@dataclass
class PoolConfig:
    """Gate pool selection and options."""
    pool_type: str = 'np'          # 'standard' | 'np' | 'fermionic'
    angle:     float = 0.785398    # π/4
    # Standard pool options
    include_x:        bool = True
    include_ry_pos:   bool = True
    include_ry_neg:   bool = True
    include_rz:       bool = False
    include_cnot:     bool = True
    include_cnot_lr:  bool = False
    # NP pool options
    include_givens_adj:  bool = True
    include_givens_lr:   bool = False
    include_fswap:       bool = False
    include_double_exc:  bool = False

    def to_kwargs(self) -> Dict:
        if self.pool_type == 'standard':
            return {
                'angle':          self.angle,
                'include_x':       self.include_x,
                'include_ry_pos':  self.include_ry_pos,
                'include_ry_neg':  self.include_ry_neg,
                'include_rz':      self.include_rz,
                'include_cnot':    self.include_cnot,
                'include_cnot_lr': self.include_cnot_lr,
            }
        elif self.pool_type in ('np', 'number_preserving'):
            return {
                'angle':               self.angle,
                'include_givens_adj':  self.include_givens_adj,
                'include_givens_lr':   self.include_givens_lr,
                'include_fswap':       self.include_fswap,
                'include_double_exc':  self.include_double_exc,
            }
        return {}


@dataclass
class TaperingConfig:
    """Qubit tapering options."""
    enabled:  bool = False
    method:   str  = 'z2'    # 'z2' | 'scbk' | 'none'
    pauli_tol: float = 1e-12


@dataclass
class BankConfig:
    """Persistent determinant bank configuration."""
    enabled:        bool  = True
    max_size:       int   = 50_000
    apply_symmetry: bool  = True
    flush_on_start: bool  = False    # clear bank at pipeline start


@dataclass
class FinetuneConfig:
    """L-BFGS-B fine-tuning configuration."""
    enabled:  bool  = True
    max_iter: int   = 400
    ftol:     float = 1e-10
    gtol:     float = 1e-8
    method:   str   = 'L-BFGS-B'


@dataclass
class RefinementConfig:
    """Global QSCI refinement configuration."""
    enabled:         bool = True
    n_extra_circuits: int = 5     # extra circuits for diversity
    only_in_qsci_mode: bool = True


@dataclass
class ForgingPipelineConfig:
    """Entanglement forging pipeline options."""
    enabled: bool = False
    cfg:     ForgingConfig = field(default_factory=ForgingConfig)


@dataclass
class PipelineConfig:
    """Master configuration for the GQEx pipeline."""
    gqe:       GQEConfig      = field(default_factory=GQEConfig)
    qsci:      QSCIConfig     = field(default_factory=QSCIConfig)
    gen:       RetNetConfig   = field(default_factory=lambda: RetNetConfig(pool_size=0))
    pool:      PoolConfig     = field(default_factory=PoolConfig)
    taper:     TaperingConfig = field(default_factory=TaperingConfig)
    bank:      BankConfig     = field(default_factory=BankConfig)
    finetune:  FinetuneConfig = field(default_factory=FinetuneConfig)
    refine:    RefinementConfig = field(default_factory=RefinementConfig)
    forging:   ForgingPipelineConfig = field(default_factory=ForgingPipelineConfig)
    adapt:     ADAPTConfig = field(default_factory=lambda: ADAPTConfig(enabled=True))
    adapt_prior_strength: float = 1.5   # bias mass added to ADAPT gates
    adapt_supervised_epochs: int = 5    # SL pretrain steps on ADAPT circuit
    compute_casci: bool = False
    verbose:       bool = True


# ─── Quick-config presets ────────────────────────────────────────────────────

def quick_config(size: str = 'medium') -> PipelineConfig:
    """Return a sensible PipelineConfig preset.

    size : 'small'  — fast demo,  ≤ 12 qubits
           'medium' — standard,   16–24 qubits  (default)
           'large'  — production, 28–40 qubits
    """
    if size == 'small':
        return PipelineConfig(
            gqe=GQEConfig(
                depth=0, n_samples=8, n_iters=40, lr=1e-2,
                reward_mode='expectation', device='auto',
                use_recurrent=False,
            ),
            qsci=QSCIConfig(n_shots=10_000, d_max=200, verbose=False),
            gen=RetNetConfig(pool_size=0, hidden_dim=64, n_layers=2,
                             n_heads=2, max_len=128),
            pool=PoolConfig(pool_type='np'),
            bank=BankConfig(enabled=False),
            finetune=FinetuneConfig(max_iter=200),
        )
    elif size == 'large':
        return PipelineConfig(
            adapt=ADAPTConfig(enabled=False, max_gates=50,
                              n_candidates_per_iter=40,  # subsample for speed
                              energy_tol=5e-4,
                              plateau_patience=4),
            adapt_prior_strength=1.5,
            adapt_supervised_epochs=10,
            gqe=GQEConfig(
                depth=0, n_samples=24, n_iters=250, lr=3e-3,
                reward_mode='qsci', device='auto',
                use_recurrent=True, use_bank=True, cosine_anneal=True,
                entropy_coef=0.12, temperature=1.8, temp_anneal=True,
                adv_std_floor=5e-3, adv_clip=5.0,
                use_rank_adv=True, diversity_coef=0.05,
                entropy_floor=0.6, entropy_adapt=0.02,
            ),
            qsci=QSCIConfig(n_shots=50_000, d_max=1500,
                            apply_symmetry=True, verbose=False,
                            use_sqd=True, sqd_n_iter=5),
            gen=RetNetConfig(pool_size=0, hidden_dim=256, n_layers=6,
                             n_heads=8, max_len=512, gated_mlp=True),
            pool=PoolConfig(pool_type='np',
                            include_givens_adj=True,
                            include_givens_lr=True,
                            include_fswap=False,
                            include_double_exc=True),
            taper=TaperingConfig(enabled=True, method='z2'),
            bank=BankConfig(enabled=True, max_size=200_000),
            finetune=FinetuneConfig(max_iter=800),
            refine=RefinementConfig(n_extra_circuits=15),
            compute_casci=True,
        )
    else:  # medium
        return PipelineConfig(
            gqe=GQEConfig(
                depth=0, n_samples=16, n_iters=120, lr=3e-3,
                reward_mode='qsci', device='auto',
                use_recurrent=True, use_bank=True,
                entropy_coef=0.10, temperature=1.5, temp_anneal=True,
                adv_std_floor=5e-3, adv_clip=5.0,
                use_rank_adv=True, diversity_coef=0.05,
                entropy_floor=0.5, entropy_adapt=0.02,
            ),
            qsci=QSCIConfig(n_shots=30_000, d_max=800,
                            apply_symmetry=True, verbose=False),
            gen=RetNetConfig(pool_size=0, hidden_dim=128, n_layers=4,
                             n_heads=4, max_len=256, gated_mlp=True),
            # Richer NP pool: long-range Givens + double excitations
            pool=PoolConfig(pool_type='np',
                            include_givens_adj=True,
                            include_givens_lr=True,
                            include_double_exc=True),
            taper=TaperingConfig(enabled=False),
            bank=BankConfig(enabled=True, max_size=100_000),
            finetune=FinetuneConfig(max_iter=600),
            refine=RefinementConfig(n_extra_circuits=10),
            # ADAPT bootstrap before RL — biggest single quality lever
            adapt=ADAPTConfig(enabled=True, max_gates=30,
                              n_candidates_per_iter=0,   # try whole pool
                              energy_tol=5e-4,
                              plateau_patience=3),
            adapt_prior_strength=1.5,
            adapt_supervised_epochs=8,
            compute_casci=True,
        )
