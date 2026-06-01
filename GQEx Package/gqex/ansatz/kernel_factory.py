"""
CUDA-Q kernel factories.

make_kernel()     — builds a kernel for StandardPool
make_np_kernel()  — builds a number-preserving kernel for NumberPreservingPool
make_param_kernel()— builds a continuous-angle kernel for L-BFGS-B fine-tuning

Givens rotation decomposition (number-preserving, 6-gate circuit):
    G(p, p+1, θ):
        CNOT(ctrl=p, tgt=p+1)
        Ry(+θ, p)
        CNOT(ctrl=p+1, tgt=p)
        Ry(-θ, p)
        CNOT(ctrl=p+1, tgt=p)
        CNOT(ctrl=p, tgt=p+1)
    Result:
        |00⟩ → |00⟩
        |01⟩ → cos θ |01⟩ + sin θ |10⟩
        |10⟩ → -sin θ |01⟩ + cos θ |10⟩
        |11⟩ → |11⟩
"""

import math
from typing import List

import cudaq

from gqex.ansatz.gate_pool import StandardPool, NumberPreservingPool, PoolBase


# ─── Standard kernel ──────────────────────────────────────────────────────────

def make_kernel(pool: StandardPool, hf_x_qubits: List[int]):
    """Build a CUDA-Q kernel for a StandardPool."""
    n           = pool.n_qubits
    angle_pos   =  pool.angle
    angle_neg   = -pool.angle
    has_x       = pool.include_x
    has_ry_pos  = pool.include_ry_pos
    has_ry_neg  = pool.include_ry_neg
    has_rz      = pool.include_rz
    has_cnot    = pool.include_cnot
    has_cnot_lr = pool.include_cnot_lr

    cur  = 1
    bx     = cur; cur += n if has_x      else 0
    bryp   = cur; cur += n if has_ry_pos else 0
    bryn   = cur; cur += n if has_ry_neg else 0
    brz    = cur; cur += n if has_rz     else 0
    bcnot  = cur; cur += n if has_cnot   else 0
    bcnotl = cur; cur += n if has_cnot_lr else 0

    @cudaq.kernel
    def kernel(ops: list[int]):
        q = cudaq.qvector(n)
        for k in range(len(hf_x_qubits)):
            x(q[hf_x_qubits[k]])
        for j in range(len(ops)):
            op = ops[j]
            if op == 0:
                rx(0.0, q[0])
            elif has_x and op < bx + n:
                x(q[op - bx])
            elif has_ry_pos and op < bryp + n:
                ry(angle_pos, q[op - bryp])
            elif has_ry_neg and op < bryn + n:
                ry(angle_neg, q[op - bryn])
            elif has_rz and op < brz + n:
                rz(angle_pos, q[op - brz])
            elif has_cnot and op < bcnot + n:
                ctrl = op - bcnot
                tgt  = (ctrl + 1) % n
                x.ctrl(q[ctrl], q[tgt])
            elif has_cnot_lr and op < bcnotl + n:
                ctrl = op - bcnotl
                tgt  = (ctrl + 2) % n
                x.ctrl(q[ctrl], q[tgt])

    return kernel


# ─── Number-preserving kernel ─────────────────────────────────────────────────

def make_np_kernel(pool: NumberPreservingPool, hf_x_qubits: List[int]):
    """Build a CUDA-Q kernel for a NumberPreservingPool.

    All gate families preserve the total number of |1⟩ qubits.
    """
    n    = pool.n_qubits
    apos =  pool.angle
    aneg = -pool.angle

    has_gadj  = pool.include_givens_adj
    has_glr   = pool.include_givens_lr
    has_fswap = pool.include_fswap
    has_dbl   = pool.include_double_exc

    n_adj  = n - 1
    n_lr   = n - 2 if n >= 3 else 0
    n_dbl  = n // 4

    cur = 1
    bgap  = cur; cur += n_adj if has_gadj  else 0   # Givens adj +
    bgan  = cur; cur += n_adj if has_gadj  else 0   # Givens adj -
    bglrp = cur; cur += n_lr  if has_glr   else 0   # Givens lr  +
    bglrn = cur; cur += n_lr  if has_glr   else 0   # Givens lr  -
    bfs   = cur; cur += n_adj if has_fswap else 0   # fSWAP
    bdp   = cur; cur += n_dbl if has_dbl   else 0   # double exc +
    bdn   = cur; cur += n_dbl if has_dbl   else 0   # double exc -

    @cudaq.kernel
    def kernel(ops: list[int]):
        q = cudaq.qvector(n)
        for k in range(len(hf_x_qubits)):
            x(q[hf_x_qubits[k]])
        for j in range(len(ops)):
            op = ops[j]
            if op == 0:
                rx(0.0, q[0])
            elif has_gadj and op < bgap + n_adj:
                # Givens(+θ) on qubits p, p+1
                p = op - bgap
                x.ctrl(q[p],   q[p + 1])
                ry(apos, q[p])
                x.ctrl(q[p + 1], q[p])
                ry(aneg, q[p])
                x.ctrl(q[p + 1], q[p])
                x.ctrl(q[p],   q[p + 1])
            elif has_gadj and op < bgan + n_adj:
                # Givens(-θ) on qubits p, p+1
                p = op - bgan
                x.ctrl(q[p],   q[p + 1])
                ry(aneg, q[p])
                x.ctrl(q[p + 1], q[p])
                ry(apos, q[p])
                x.ctrl(q[p + 1], q[p])
                x.ctrl(q[p],   q[p + 1])
            elif has_glr and op < bglrp + n_lr:
                # Givens(+θ) on qubits p, p+2
                p = op - bglrp
                x.ctrl(q[p],   q[p + 2])
                ry(apos, q[p])
                x.ctrl(q[p + 2], q[p])
                ry(aneg, q[p])
                x.ctrl(q[p + 2], q[p])
                x.ctrl(q[p],   q[p + 2])
            elif has_glr and op < bglrn + n_lr:
                p = op - bglrn
                x.ctrl(q[p],   q[p + 2])
                ry(aneg, q[p])
                x.ctrl(q[p + 2], q[p])
                ry(apos, q[p])
                x.ctrl(q[p + 2], q[p])
                x.ctrl(q[p],   q[p + 2])
            elif has_fswap and op < bfs + n_adj:
                # fSWAP(p, p+1) — number-preserving (Givens at θ=π/2)
                p = op - bfs
                x.ctrl(q[p], q[p + 1])
                ry(1.5707963267948966, q[p])   # π/2
                x.ctrl(q[p + 1], q[p])
                ry(-1.5707963267948966, q[p])
                x.ctrl(q[p + 1], q[p])
                x.ctrl(q[p], q[p + 1])
            elif has_dbl and op < bdp + n_dbl:
                # Double excitation +θ on 4-qubit window 4k..4k+3
                k  = op - bdp
                p0 = 4 * k
                # Sequence: Givens on (p0,p0+1) then (p0+2,p0+3) then (p0+1,p0+2)
                x.ctrl(q[p0],     q[p0 + 1])
                ry(apos, q[p0])
                x.ctrl(q[p0 + 1], q[p0])
                ry(aneg, q[p0])
                x.ctrl(q[p0 + 1], q[p0])
                x.ctrl(q[p0],     q[p0 + 1])
                x.ctrl(q[p0 + 2], q[p0 + 3])
                ry(apos, q[p0 + 2])
                x.ctrl(q[p0 + 3], q[p0 + 2])
                ry(aneg, q[p0 + 2])
                x.ctrl(q[p0 + 3], q[p0 + 2])
                x.ctrl(q[p0 + 2], q[p0 + 3])
            elif has_dbl and op < bdn + n_dbl:
                k  = op - bdn
                p0 = 4 * k
                x.ctrl(q[p0],     q[p0 + 1])
                ry(aneg, q[p0])
                x.ctrl(q[p0 + 1], q[p0])
                ry(apos, q[p0])
                x.ctrl(q[p0 + 1], q[p0])
                x.ctrl(q[p0],     q[p0 + 1])
                x.ctrl(q[p0 + 2], q[p0 + 3])
                ry(aneg, q[p0 + 2])
                x.ctrl(q[p0 + 3], q[p0 + 2])
                ry(apos, q[p0 + 2])
                x.ctrl(q[p0 + 3], q[p0 + 2])
                x.ctrl(q[p0 + 2], q[p0 + 3])

    return kernel


# ─── Parametric kernel (for fine-tuning) ─────────────────────────────────────

def make_param_kernel(best_ops: List[int], pool: PoolBase, hf_x_qubits: List[int]):
    """Build a parametric CUDA-Q kernel with continuous angles.

    Each discrete ±θ rotation in best_ops is replaced by an optimisable
    parameter in `angles`.  Non-rotation ops (X, CNOT, Givens-without-angle
    when angle is baked in) remain fixed.

    Returns (param_kernel, n_params, initial_angles).
    """
    n    = pool.n_qubits
    offs = pool.family_offsets()

    # Decide which op indices are rotations and record their initial angle
    rot_positions  = []   # (position_in_ops, initial_angle)
    is_std = isinstance(pool, StandardPool)
    is_np  = isinstance(pool, NumberPreservingPool)

    for idx, op in enumerate(best_ops):
        angle = None
        if is_std:
            if 'ry_pos' in offs and offs['ry_pos'] <= op and op < offs['ry_pos'] + n:
                angle = pool.angle
            elif 'ry_neg' in offs and offs['ry_neg'] <= op and op < offs['ry_neg'] + n:
                angle = -pool.angle
            elif 'rz' in offs and offs['rz'] <= op and op < offs['rz'] + n:
                angle = pool.angle
        elif is_np:
            n_adj = n - 1
            n_lr  = n - 2 if n >= 3 else 0
            if 'givens_adj_pos' in offs and offs['givens_adj_pos'] <= op and op < offs['givens_adj_pos'] + n_adj:
                angle = pool.angle
            elif 'givens_adj_neg' in offs and offs['givens_adj_neg'] <= op and op < offs['givens_adj_neg'] + n_adj:
                angle = -pool.angle
            elif 'givens_lr_pos' in offs and offs['givens_lr_pos'] <= op and op < offs['givens_lr_pos'] + n_lr:
                angle = pool.angle
            elif 'givens_lr_neg' in offs and offs['givens_lr_neg'] <= op and op < offs['givens_lr_neg'] + n_lr:
                angle = -pool.angle
        if angle is not None:
            rot_positions.append((idx, angle))

    n_params = len(rot_positions)
    import numpy as np
    initial_angles = np.array([a for _, a in rot_positions])

    # ---- build parametric kernel ----
    ops_fixed  = list(best_ops)

    # Precompute family boundaries for use inside kernel
    _is_std = is_std
    _is_np  = is_np

    if is_std:
        _pool_x_lo   = offs.get('x',       -1)
        _pool_x_hi   = _pool_x_lo + n
        _pool_ryp_lo = offs.get('ry_pos',  -1)
        _pool_ryp_hi = _pool_ryp_lo + n
        _pool_ryn_lo = offs.get('ry_neg',  -1)
        _pool_ryn_hi = _pool_ryn_lo + n
        _pool_rz_lo  = offs.get('rz',      -1)
        _pool_rz_hi  = _pool_rz_lo + n
        _pool_cn_lo  = offs.get('cnot',    -1)
        _pool_cn_hi  = _pool_cn_lo + n
        _pool_cl_lo  = offs.get('cnot_lr', -1)
        _pool_cl_hi  = _pool_cl_lo + n

        @cudaq.kernel
        def param_kernel(angles: list[float]):
            q = cudaq.qvector(n)
            for k in range(len(hf_x_qubits)):
                x(q[hf_x_qubits[k]])
            a_idx = 0
            for j in range(len(ops_fixed)):
                op = ops_fixed[j]
                if op == 0:
                    rx(0.0, q[0])
                elif _pool_x_lo <= op and op < _pool_x_hi:
                    x(q[op - _pool_x_lo])
                elif _pool_ryp_lo <= op and op < _pool_ryp_hi:
                    ry(angles[a_idx], q[op - _pool_ryp_lo]); a_idx = a_idx + 1
                elif _pool_ryn_lo <= op and op < _pool_ryn_hi:
                    ry(angles[a_idx], q[op - _pool_ryn_lo]); a_idx = a_idx + 1
                elif _pool_rz_lo  <= op and op < _pool_rz_hi:
                    rz(angles[a_idx], q[op - _pool_rz_lo]);  a_idx = a_idx + 1
                elif _pool_cn_lo  <= op and op < _pool_cn_hi:
                    ctrl = op - _pool_cn_lo
                    tgt  = (ctrl + 1) % n
                    x.ctrl(q[ctrl], q[tgt])
                elif _pool_cl_lo  <= op and op < _pool_cl_hi:
                    ctrl = op - _pool_cl_lo
                    tgt  = (ctrl + 2) % n
                    x.ctrl(q[ctrl], q[tgt])

    elif is_np:
        n_adj  = n - 1
        n_lr   = n - 2 if n >= 3 else 0
        _bgap  = offs.get('givens_adj_pos', -1)
        _bgan  = offs.get('givens_adj_neg', -1)
        _bglrp = offs.get('givens_lr_pos',  -1)
        _bglrn = offs.get('givens_lr_neg',  -1)
        _bfs   = offs.get('fswap',          -1)

        @cudaq.kernel
        def param_kernel(angles: list[float]):
            q = cudaq.qvector(n)
            for k in range(len(hf_x_qubits)):
                x(q[hf_x_qubits[k]])
            a_idx = 0
            for j in range(len(ops_fixed)):
                op = ops_fixed[j]
                if op == 0:
                    rx(0.0, q[0])
                elif _bgap <= op and op < _bgap + n_adj:
                    p = op - _bgap
                    x.ctrl(q[p], q[p + 1])
                    ry(angles[a_idx], q[p])
                    x.ctrl(q[p + 1], q[p])
                    ry(-angles[a_idx], q[p])
                    x.ctrl(q[p + 1], q[p])
                    x.ctrl(q[p], q[p + 1])
                    a_idx = a_idx + 1
                elif _bgan <= op and op < _bgan + n_adj:
                    p = op - _bgan
                    x.ctrl(q[p], q[p + 1])
                    ry(angles[a_idx], q[p])
                    x.ctrl(q[p + 1], q[p])
                    ry(-angles[a_idx], q[p])
                    x.ctrl(q[p + 1], q[p])
                    x.ctrl(q[p], q[p + 1])
                    a_idx = a_idx + 1
                elif _bglrp <= op and op < _bglrp + n_lr:
                    p = op - _bglrp
                    x.ctrl(q[p], q[p + 2])
                    ry(angles[a_idx], q[p])
                    x.ctrl(q[p + 2], q[p])
                    ry(-angles[a_idx], q[p])
                    x.ctrl(q[p + 2], q[p])
                    x.ctrl(q[p], q[p + 2])
                    a_idx = a_idx + 1
                elif _bglrn <= op and op < _bglrn + n_lr:
                    p = op - _bglrn
                    x.ctrl(q[p], q[p + 2])
                    ry(angles[a_idx], q[p])
                    x.ctrl(q[p + 2], q[p])
                    ry(-angles[a_idx], q[p])
                    x.ctrl(q[p + 2], q[p])
                    x.ctrl(q[p], q[p + 2])
                    a_idx = a_idx + 1
                elif _bfs <= op and op < _bfs + n_adj:
                    p = op - _bfs
                    x.ctrl(q[p], q[p + 1])
                    ry(1.5707963267948966, q[p])
                    x.ctrl(q[p + 1], q[p])
                    ry(-1.5707963267948966, q[p])
                    x.ctrl(q[p + 1], q[p])
                    x.ctrl(q[p], q[p + 1])
    else:
        raise ValueError("make_param_kernel only supports StandardPool and NumberPreservingPool")

    return param_kernel, n_params, initial_angles


# ─── Dispatch helper ──────────────────────────────────────────────────────────

def build_kernel(pool: PoolBase, hf_x_qubits: List[int]):
    """Build the appropriate kernel for the given pool type."""
    if isinstance(pool, StandardPool):
        return make_kernel(pool, hf_x_qubits)
    elif isinstance(pool, NumberPreservingPool):
        return make_np_kernel(pool, hf_x_qubits)
    else:
        raise ValueError(f"No kernel factory for pool type {type(pool).__name__}")
