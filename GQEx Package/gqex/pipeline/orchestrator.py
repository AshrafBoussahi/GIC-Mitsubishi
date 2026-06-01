"""
GQEx pipeline orchestrator.

Assembles all modules and runs the full pipeline:
  A. Build gate pool + CUDA-Q kernel
  B. Initialise RetNet generator
  C. Run GQE RL training (with persistent det bank)
  D. QSCI evaluation of elite circuit
  E. Global refinement across diverse circuits
  F. L-BFGS-B continuous-angle fine-tuning
  G. Final bank-based QSCI diagonalisation
  H. Summary

Optional paths:
  • Entanglement forging (if cfg.forging.enabled)
  • Qubit tapering (if cfg.taper.enabled)
"""

import math
from typing import Any, Dict, Optional

from gqex.hamiltonian.molecular_system import MolecularSystem, build_molecular_hamiltonian
from gqex.ansatz.gate_pool import build_pool
from gqex.ansatz.kernel_factory import build_kernel
from gqex.generator.retnet import RetNetGenerator
from gqex.generator.sampling import sample_circuit
from gqex.qsci.evaluator import QSCIEvaluator
from gqex.qsci.det_bank import DeterminantBank
from gqex.gqe.engine import GQEEngine
from gqex.gqe.finetune import finetune_circuit
from gqex.pipeline.config import PipelineConfig


class GQExPipeline:
    """
    End-to-end GQEx pipeline.

    Usage
    -----
    >>> from gqex.pipeline.config import quick_config
    >>> from gqex.hamiltonian.molecular_system import MolecularSystem
    >>> system = MolecularSystem('H2', 'H 0 0 0\nH 0 0 0.74', basis='sto-3g',
    ...                          n_active_electrons=2, n_active_orbitals=2)
    >>> pipeline = GQExPipeline(system, quick_config('small'))
    >>> results = pipeline.run()
    """

    def __init__(self, system: MolecularSystem, cfg: PipelineConfig):
        self.system = system
        self.cfg    = cfg
        self._built = False

    # ── Build phase ───────────────────────────────────────────────────────

    def build(self):
        """Construct all components (lazy; called automatically by run())."""
        cfg    = self.cfg
        system = self.system

        # 1. Hamiltonian
        if system.hamiltonian is None:
            system.compute_casci = cfg.compute_casci
            build_molecular_hamiltonian(
                system, verbose=cfg.verbose,
                taper=cfg.taper.enabled,
                taper_method=cfg.taper.method,
                pauli_tol=cfg.taper.pauli_tol,
            )

        # 2. Pool + kernel
        pool_kwargs = cfg.pool.to_kwargs()
        self.pool   = build_pool(cfg.pool.pool_type, system.n_qubits, **pool_kwargs)
        self.kernel = build_kernel(self.pool, system.hf_x_qubits)

        if cfg.verbose:
            print(self.pool.describe())

        # 3. Generator
        gcfg           = cfg.gen
        gcfg.pool_size = self.pool.pool_size
        depth          = cfg.gqe.depth or 4 * system.n_qubits
        if gcfg.max_len < depth + 4:
            gcfg.max_len = depth + 8
        self.generator = RetNetGenerator(gcfg)

        # Warm start: bias toward identity + useful gates.
        # For NumberPreservingPool: favour Givens crossing the Fermi level.
        # For StandardPool: favour X-on-HF-qubit slots.
        x_offset = self.pool.family_offsets().get('x', 1)
        self.generator.warm_start_bias(
            identity_bias=0.5,
            hf_x_qubits=system.hf_x_qubits,
            x_offset=x_offset,
            hf_bias=0.3,
            pool=self.pool,
            n_alpha=system.n_alpha,
            n_beta=system.n_beta,
            boundary_bias=0.3,    # gentle: a hint, not a command
        )
        if cfg.verbose:
            print(f"  RetNet: {self.generator.n_params():,} parameters  "
                  f"(hidden={gcfg.hidden_dim}, layers={gcfg.n_layers}, "
                  f"heads={gcfg.n_heads})")

        # 4. Determinant bank
        self.bank = None
        if cfg.bank.enabled:
            self.bank = DeterminantBank(
                max_size=cfg.bank.max_size,
                n_alpha=system.n_alpha,
                n_beta=system.n_beta,
                n_qubits=system.n_qubits,
                apply_symmetry=cfg.bank.apply_symmetry,
            )

        # 5. QSCI evaluator
        self.qsci = QSCIEvaluator(system, cfg.qsci, bank=self.bank)

        # 6. GQE engine
        cfg.gqe.use_bank = cfg.bank.enabled
        self.engine = GQEEngine(
            system=system, pool=self.pool, kernel=self.kernel,
            generator=self.generator, cfg=cfg.gqe,
            qsci=self.qsci, bank=self.bank,
        )

        self._built = True

    # ── Run phase ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        if not self._built:
            self.build()

        cfg    = self.cfg
        system = self.system

        # ── (0) ADAPT-VQE bootstrap ──────────────────────────────────────
        # Greedy gate-by-gate construction gives RL a strong starting
        # circuit — typically captures 30–70% of correlation energy
        # before any RL training begins.
        adapt_ops: Optional[list] = None
        adapt_e:   Optional[float] = None
        if cfg.adapt.enabled:
            from gqex.adapt.bootstrap import adapt_bootstrap
            adapt_ops, adapt_e = adapt_bootstrap(
                pool=self.pool, system=system, kernel=self.kernel,
                cfg=cfg.adapt,
            )
            if adapt_ops:
                # Seed the generator with the ADAPT circuit:
                # 1. log-frequency bias (cheap)
                # 2. supervised SL pretrain to learn the actual sequence
                self.generator.prior_from_ops(
                    adapt_ops,
                    bias_strength=cfg.adapt_prior_strength,
                    supervised_epochs=cfg.adapt_supervised_epochs,
                    lr=1e-3,
                )
                # Push ADAPT circuit dets into bank so QSCI sees them
                if self.bank is not None:
                    import cudaq
                    counts = cudaq.sample(self.kernel, adapt_ops,
                                          shots_count=cfg.qsci.n_shots)
                    self.bank.update_from_counter({b: c for b, c in counts.items()})

        # ── (A) GQE RL training ──────────────────────────────────────────
        rl_out    = self.engine.run()
        best_e_rl = rl_out['best_energy']
        best_ops  = rl_out['best_ops']
        if cfg.verbose:
            print(f"\n  RL best (raw):  {best_e_rl:+.6f} Ha")

        # ── (B) QSCI on elite circuit ────────────────────────────────────
        if cfg.verbose:
            print("\n  Running QSCI on elite circuit ...")
        e_q, wf_q, n_q = self.qsci.evaluate(self.kernel, best_ops, use_bank=True)
        if cfg.verbose:
            print(f"     dets={n_q}   E_QSCI={e_q:+.6f} Ha")

        # ── (C) Bank-based QSCI (accumulated shots) ──────────────────────
        e_bank, wf_bank = e_q, wf_q
        if self.bank and self.bank.size > 0:
            if cfg.verbose:
                print(f"\n  Bank QSCI ({self.bank.size} dets) ...")
            e_bank, wf_bank, n_bank = self.qsci.evaluate_from_bank()
            if cfg.verbose:
                print(f"     dets={n_bank}   E_bank={e_bank:+.6f} Ha")

        # ── (D) Global refinement ────────────────────────────────────────
        global_e, global_wf = e_q, wf_q
        if (cfg.refine.enabled and
                (not cfg.refine.only_in_qsci_mode or
                 cfg.gqe.reward_mode == 'qsci')):
            if cfg.verbose:
                print(f"\n  Global refinement (extra={cfg.refine.n_extra_circuits}) ...")
            extra_wfs = []
            for _ in range(cfg.refine.n_extra_circuits):
                c = sample_circuit(
                    self.generator, cfg.gqe.depth or 4 * system.n_qubits,
                    cfg.gqe.device,
                )
                _, wf_, _ = self.qsci.evaluate(self.kernel, c, use_bank=True)
                if wf_:
                    extra_wfs.append(wf_)
            wf_pool = [wf_q] + (extra_wfs if extra_wfs else [])
            if len(wf_pool) > 1 or wf_bank:
                wf_pool_full = [wf_bank or wf_q] + extra_wfs
                e_glob, wf_glob, n_glob = self.qsci.refine(wf_pool_full)
                if cfg.verbose:
                    print(f"     dets={n_glob}   E_global={e_glob:+.6f} Ha")
                global_e, global_wf = e_glob, wf_glob

        # ── (E) Continuous-angle fine-tuning ─────────────────────────────
        ft_e = None
        if cfg.finetune.enabled and best_ops is not None:
            if cfg.verbose:
                print("\n  Fine-tuning best discrete circuit ...")
            _, ft_e = finetune_circuit(
                best_ops, self.pool, system,
                max_iter=cfg.finetune.max_iter,
                ftol=cfg.finetune.ftol,
                gtol=cfg.finetune.gtol,
                method=cfg.finetune.method,
                verbose=cfg.verbose,
            )

        # ── (F) Summary ───────────────────────────────────────────────────
        self._print_summary(best_e_rl, e_q, global_e, e_bank, ft_e,
                            adapt_e=adapt_e)

        return {
            'adapt_energy':      adapt_e,
            'adapt_ops':         adapt_ops,
            'rl_best_energy':    best_e_rl,
            'qsci_energy':       e_q,
            'global_energy':     global_e,
            'bank_energy':       e_bank,
            'finetune_energy':   ft_e,
            'best_ops':          best_ops,
            'history':           rl_out['history'],
            'n_qubits':          system.n_qubits,
            'pool_size':         self.pool.pool_size,
        }

    # ── Summary printer ───────────────────────────────────────────────────

    def _print_summary(self, best_e_rl, e_q, global_e, e_bank, ft_e,
                       adapt_e=None):
        s = self.system
        ref = s.casci_energy

        def err(e):
            if e is None or math.isnan(e): return ""
            if math.isnan(ref):            return ""
            return f"   |Δ|={1000*abs(e-ref):6.2f} mHa"

        print("\n" + "=" * 64)
        print(f"  SUMMARY  ({s.name},  {s.n_qubits} qubits,  "
              f"pool={type(self.pool).__name__})")
        print("=" * 64)
        print(f"  HF reference     : {s.hf_energy:+.6f} Ha")
        if not math.isnan(ref):
            print(f"  CASCI reference  : {ref:+.6f} Ha  "
                  f"(corr={1000*s.correlation_energy():+.1f} mHa)")
        if adapt_e is not None:
            print(f"  ADAPT bootstrap  : {adapt_e:+.6f} Ha{err(adapt_e)}")
        print(f"  RL best (raw)    : {best_e_rl:+.6f} Ha{err(best_e_rl)}")
        print(f"  QSCI on elite    : {e_q:+.6f} Ha{err(e_q)}")
        if e_bank != e_q:
            print(f"  Bank QSCI        : {e_bank:+.6f} Ha{err(e_bank)}")
        if global_e != e_q:
            print(f"  Global refined   : {global_e:+.6f} Ha{err(global_e)}")
        if ft_e is not None and not math.isnan(ft_e):
            print(f"  L-BFGS fine-tune : {ft_e:+.6f} Ha{err(ft_e)}")
        print("=" * 64)
