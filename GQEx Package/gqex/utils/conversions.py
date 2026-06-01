"""
Conversion utilities between openfermion QubitOperator and cudaq SpinOperator.
"""

import re
import numpy as np
from typing import Optional

from openfermion import QubitOperator

try:
    import cudaq
    from cudaq import spin as S
    _CUDAQ_AVAILABLE = True
except ImportError:
    _CUDAQ_AVAILABLE = False


def of_to_cudaq(of_op: QubitOperator, tol: float = 1e-14):
    """openfermion QubitOperator → cudaq SpinOperator."""
    if not _CUDAQ_AVAILABLE:
        raise ImportError("cudaq is required for of_to_cudaq")
    result = None
    for term, coeff in of_op.terms.items():
        if abs(coeff) < tol:
            continue
        t_op = None
        for qi, p in sorted(term):
            s = S.x(qi) if p == 'X' else S.y(qi) if p == 'Y' else S.z(qi)
            t_op = s if t_op is None else t_op * s
        if t_op is None:
            t_op = S.i(0)
        scaled = float(coeff.real) * t_op
        result = scaled if result is None else result + scaled
    return result


def cudaq_to_of(spin_op) -> QubitOperator:
    """cudaq SpinOperator → openfermion QubitOperator (via string parse)."""
    of_op = QubitOperator()
    raw = spin_op.to_string()
    pat = re.compile(r'[\[\(]([^)\]]+)[\]\)]\s*\*?\s*([IXYZ]+)')
    for m in pat.finditer(raw):
        try:
            coeff = complex(m.group(1).replace('−', '-').replace(' ', ''))
        except ValueError:
            continue
        word = m.group(2)
        term = [(i, p) for i, p in enumerate(word) if p != 'I']
        of_op += QubitOperator(term if term else '', coeff)
    return of_op


def qubit_op_to_matrix(qubit_op: QubitOperator, n_qubits: int) -> np.ndarray:
    """Convert QubitOperator to dense numpy matrix (for small systems)."""
    from openfermion.linalg import get_sparse_operator
    sparse = get_sparse_operator(qubit_op, n_qubits=n_qubits)
    return sparse.toarray()


def bitstring_to_index(b: str, n: int) -> int:
    """Convert a bitstring to an integer index."""
    return sum(int(b[k]) << (n - 1 - k) for k in range(n))


def index_to_bitstring(idx: int, n: int) -> str:
    """Convert an integer index to a bitstring of length n."""
    return format(int(idx), '0' + str(n) + 'b')


def apply_pauli_to_state(term, state: int, n: int):
    """Apply a Pauli string term to a computational basis state.

    Returns (new_state, phase) where phase is the complex coefficient.
    """
    ph, r = 1 + 0j, state
    for q, p in term:
        bp = n - 1 - q
        b = (state >> bp) & 1
        if   p == 'X': r ^= 1 << bp
        elif p == 'Y': r ^= 1 << bp; ph *= (1j if b == 0 else -1j)
        elif p == 'Z':
            if b: ph *= -1
    return r, ph
