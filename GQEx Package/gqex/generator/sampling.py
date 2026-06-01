"""
Circuit sampling and log-probability utilities.

Supports both parallel (full sequence) and recurrent (step-by-step)
generation modes.  Recurrent mode is preferred for long circuits
(depth > 64) as it avoids O(T²) attention.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from gqex.generator.retnet import RetNetGenerator


def sample_circuit(
    generator: RetNetGenerator,
    depth: int,
    device: str,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    use_recurrent: bool = True,
) -> List[int]:
    """Autoregressively sample one circuit of length `depth`.

    Parameters
    ----------
    generator    : trained RetNetGenerator
    depth        : number of gate slots
    device       : torch device string
    temperature  : softmax temperature (>1 = more random, <1 = sharper)
    top_k        : if set, restrict sampling to top-k logits
    use_recurrent: use O(T) recurrent mode (recommended for depth>64)
    """
    P = generator.cfg.pool_size
    with torch.no_grad():
        if use_recurrent:
            ops_list = []
            token  = torch.full((1,), P, dtype=torch.long, device=device)  # BOS
            states = None
            for t in range(depth):
                logits, states = generator.step(token, states, t)
                logits = logits[0] / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[-1]] = -float('inf')
                probs = torch.softmax(logits, dim=-1)
                op    = torch.multinomial(probs, 1)
                ops_list.append(int(op.item()))
                token = op
            return ops_list
        else:
            ops = torch.full((1, 1), P, dtype=torch.long, device=device)
            for _ in range(depth):
                logits = generator(ops)[0, -1] / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[-1]] = -float('inf')
                probs = torch.softmax(logits, dim=-1)
                op    = torch.multinomial(probs, 1)
                ops   = torch.cat([ops, op.unsqueeze(0)], dim=1)
            return ops[0, 1:].tolist()


def sample_batch(
    generator: RetNetGenerator,
    depth: int,
    device: str,
    n_samples: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    use_recurrent: bool = True,
) -> List[List[int]]:
    """Sample n_samples circuits in parallel (batched).

    top_k : keep only top-k logits per step
    top_p : keep smallest set whose cumulative probability ≥ top_p (nucleus)
    """
    P = generator.cfg.pool_size
    B = n_samples

    def _filter(logits: torch.Tensor) -> torch.Tensor:
        if top_k is not None:
            v, _ = torch.topk(logits, top_k, dim=-1)
            logits = logits.masked_fill(logits < v[:, -1:], -float('inf'))
        if top_p is not None:
            sorted_l, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.softmax(sorted_l, dim=-1).cumsum(dim=-1)
            mask = cum > top_p
            # shift right so we keep the first token that crosses top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0]  = False
            sorted_l = sorted_l.masked_fill(mask, -float('inf'))
            logits = torch.full_like(logits, -float('inf')).scatter_(
                -1, sorted_idx, sorted_l
            )
        return logits

    with torch.no_grad():
        if use_recurrent:
            all_ops = [[] for _ in range(B)]
            tokens = torch.full((B,), P, dtype=torch.long, device=device)
            states = None
            for t in range(depth):
                logits, states = generator.step(tokens, states, t)
                logits = _filter(logits / max(temperature, 1e-6))
                probs  = torch.softmax(logits, dim=-1)
                tokens = torch.multinomial(probs, 1).squeeze(1)
                for b in range(B):
                    all_ops[b].append(int(tokens[b].item()))
            return all_ops
        else:
            ops = torch.full((B, 1), P, dtype=torch.long, device=device)
            for _ in range(depth):
                logits = _filter(generator(ops)[:, -1] / max(temperature, 1e-6))
                probs = torch.softmax(logits, dim=-1)
                tok   = torch.multinomial(probs, 1)
                ops   = torch.cat([ops, tok], dim=1)
            return ops[:, 1:].tolist()


def log_prob(
    generator: RetNetGenerator,
    op_seq: List[int],
    device: str,
) -> torch.Tensor:
    """Differentiable sum of log-probabilities: Σ_t log P(op_t | op_{<t})."""
    P    = generator.cfg.pool_size
    full = [P] + op_seq
    ops  = torch.tensor([full], dtype=torch.long, device=device)
    logits = generator(ops)          # (1, T+1, pool_size)
    logp   = torch.log_softmax(logits[0], dim=-1)
    return torch.stack([logp[t, op_seq[t]] for t in range(len(op_seq))]).sum()


def log_prob_batch(
    generator: RetNetGenerator,
    op_seqs: List[List[int]],
    device: str,
) -> torch.Tensor:
    """Batched differentiable log-prob computation.  Returns (B,) tensor."""
    P   = generator.cfg.pool_size
    B   = len(op_seqs)
    T   = max(len(s) for s in op_seqs)
    # Pad sequences to same length
    padded = [[P] + s + [0] * (T - len(s)) for s in op_seqs]
    ops    = torch.tensor(padded, dtype=torch.long, device=device)  # (B, T+1)
    logits = generator(ops)                                           # (B, T+1, pool_size)
    logp   = torch.log_softmax(logits, dim=-1)                       # (B, T+1, pool_size)
    result = []
    for b, seq in enumerate(op_seqs):
        lp = sum(logp[b, t, seq[t]].unsqueeze(0) for t in range(len(seq)))
        result.append(lp)
    return torch.stack(result).squeeze(-1)


def beam_search(
    generator: RetNetGenerator,
    depth: int,
    device: str,
    beam_width: int = 8,
) -> Tuple[List[int], float]:
    """Greedy beam search.  Returns (best_ops, log_prob_score)."""
    P = generator.cfg.pool_size
    with torch.no_grad():
        # Initialise beams: (log_prob, ops_list, states)
        beams = [(0.0, [], None)]
        for t in range(depth):
            candidates = []
            for lp, ops_so_far, states in beams:
                token = torch.tensor(
                    [ops_so_far[-1] if ops_so_far else P],
                    dtype=torch.long, device=device
                )
                logits, new_states = generator.step(token, states, t)
                log_probs = torch.log_softmax(logits[0], dim=-1)
                top_vals, top_idx = torch.topk(log_probs, beam_width)
                for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
                    candidates.append((lp + val, ops_so_far + [idx], new_states))
            # Keep top beam_width
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:beam_width]
        best_lp, best_ops, _ = beams[0]
    return best_ops, best_lp
