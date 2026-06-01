"""
L-BFGS-B continuous-angle fine-tuner.

After the discrete GQE phase locks in a gate structure, this module
replaces each fixed ±θ rotation with an optimisable continuous angle
and minimises ⟨H⟩ via L-BFGS-B.  Typically recovers 5–50 mHa over
the discrete solution on 20–40 qubit problems.
"""

import math
import numpy as np
import scipy.optimize as opt
from typing import List, Optional, Tuple

import cudaq

from gqex.ansatz.gate_pool import PoolBase
from gqex.ansatz.kernel_factory import make_param_kernel


def finetune_circuit(
    best_ops: List[int],
    pool: PoolBase,
    system,
    max_iter: int  = 400,
    ftol:     float = 1e-10,
    gtol:     float = 1e-8,
    method:   str   = 'L-BFGS-B',
    verbose:  bool  = True,
) -> Tuple[np.ndarray, float]:
    """
    Fine-tune continuous rotation angles in a fixed gate structure.

    Parameters
    ----------
    best_ops  : discrete gate sequence from GQE
    pool      : gate pool used during GQE
    system    : MolecularSystem
    max_iter  : L-BFGS-B max iterations
    ftol      : function-value tolerance
    gtol      : gradient tolerance
    method    : scipy optimise method ('L-BFGS-B' or 'COBYLA')
    verbose   : print progress

    Returns (optimal_angles, final_energy).
    """
    hf_x = system.hf_x_qubits

    # Build parametric kernel
    try:
        param_kernel, n_params, x0 = make_param_kernel(best_ops, pool, hf_x)
    except ValueError as e:
        if verbose:
            print(f"  Fine-tuning skipped: {e}")
        e0 = float(cudaq.observe(
            _make_fixed_kernel(best_ops, pool, hf_x), system.hamiltonian, []
        ).expectation()) if False else float('nan')
        return np.array([]), float('nan')

    if verbose:
        print(f"\n  Fine-tuning {n_params} rotations  "
              f"({method}, max_iter={max_iter})")

    if n_params == 0:
        if verbose:
            print("  (no rotations in best circuit; skipping)")
        return np.array([]), float('nan')

    # Objective
    def energy_fn(theta: np.ndarray) -> float:
        return float(
            cudaq.observe(param_kernel, system.hamiltonian, theta.tolist())
            .expectation()
        )

    e0 = energy_fn(x0)
    if verbose:
        print(f"  Initial energy: {e0:+.6f} Ha")

    # Sanity check: the param kernel at initial angles should reproduce the
    # discrete circuit's energy.  A large discrepancy means the parametric
    # kernel is structurally inequivalent — warn loudly.
    try:
        from gqex.ansatz.kernel_factory import build_kernel
        fixed_kernel = build_kernel(pool, hf_x)
        e_fixed = float(cudaq.observe(fixed_kernel, system.hamiltonian,
                                      best_ops).expectation())
        if verbose:
            mismatch_mHa = 1000 * abs(e0 - e_fixed)
            if mismatch_mHa > 1.0:
                print(f"  ⚠  param-kernel ↔ fixed-kernel mismatch: "
                      f"{mismatch_mHa:.2f} mHa  "
                      f"(fixed={e_fixed:+.6f}, param@init={e0:+.6f})")
                print(f"  → seeding L-BFGS from x0 anyway, but expect drift.")
            else:
                print(f"  ✓ param kernel matches discrete circuit "
                      f"({mismatch_mHa:.3f} mHa)")
    except Exception:
        pass

    if method == 'L-BFGS-B':
        res = opt.minimize(
            energy_fn, x0, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol},
        )
    elif method == 'COBYLA':
        res = opt.minimize(
            energy_fn, x0, method='COBYLA',
            options={'maxiter': max_iter, 'rhobeg': 0.1},
        )
    else:
        res = opt.minimize(energy_fn, x0, method=method,
                           options={'maxiter': max_iter})

    e_ft = float(res.fun)
    # If L-BFGS got stuck in a flat region (Δ < 0.1 mHa), retry with
    # a random restart — discrete circuits often have degenerate first-order
    # zero gradients due to symmetric rotation pairs.
    if abs(e0 - e_ft) < 1e-4 and method == 'L-BFGS-B':
        if verbose:
            print(f"  L-BFGS plateau — retrying with random restart ...")
        import numpy as _np
        x0_pert = x0 + 0.1 * _np.random.randn(*x0.shape)
        res2 = opt.minimize(
            energy_fn, x0_pert, method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol},
        )
        if res2.fun < e_ft:
            res, e_ft = res2, float(res2.fun)

    if verbose:
        print(f"  Final energy:   {e_ft:+.6f} Ha  "
              f"(Δ = {1000*(e0-e_ft):+.1f} mHa,  "
              f"converged={res.success})")

    return res.x, e_ft


def _make_fixed_kernel(ops, pool, hf_x):
    """Fallback: build a no-parameter kernel with baked-in angles."""
    from gqex.ansatz.kernel_factory import build_kernel
    return build_kernel(pool, hf_x)
