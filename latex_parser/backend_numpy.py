r"""
NumPy backend wrapper.

This module compiles LaTeX-defined models by delegating to the QuTiP backend
and converting resulting operators to NumPy arrays. Time-dependent terms are
returned as (array, scalar_envelope) pairs; callable-operator terms are returned
as callables producing arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from qutip import Qobj  # type: ignore

from latex_parser.backend_qutip import (
    CompiledOpenSystemQutip,
    compile_open_system_from_latex as _compile_qutip,
)
from latex_parser.dsl import HilbertConfig


def _qobj_to_np(obj: Qobj) -> np.ndarray:
    r"""Convert a QuTiP ``Qobj`` to a NumPy array."""
    return np.array(obj.full())


def _convert_H(
    H_q: Any,
) -> Tuple[np.ndarray | None, List[Tuple[np.ndarray | None, Callable]]]:
    r"""
    Convert QuTiP H structure to NumPy.

    Returns (H0, time_terms) where time_terms entries are (Hk_array or None, fk).
    """

    def _wrap_callable_to_np(fn: Callable) -> Callable:
        r"""
        Wrap a QuTiP-compat callable fk(t, args) that returns Qobj into one
        that returns a NumPy array. If the callable already returns an array,
        it is passed through untouched.
        """

        def _wrapped(t, args):
            r"""Return the callable output as a NumPy array."""
            val = fn(t, args)
            if hasattr(val, "full"):
                return np.array(val.full())
            return np.array(val)

        return _wrapped

    if isinstance(H_q, list):
        H0 = _qobj_to_np(H_q[0])
        time_terms: List[Tuple[np.ndarray | None, Callable]] = []
        for term in H_q[1:]:
            if isinstance(term, list) and len(term) == 2:
                Hk, fk = term
                Hk_np = _qobj_to_np(Hk)
                time_terms.append((Hk_np, fk))
            else:
                # Callable operator term (from operator-valued function with a
                # time-dependent scalar factor)
                fk = term
                time_terms.append((None, _wrap_callable_to_np(fk)))
        return H0, time_terms
    else:
        return _qobj_to_np(H_q), []


def _convert_c_ops(c_ops_q: List[Any]) -> List[Any]:
    r"""Convert QuTiP-style collapse operators to NumPy arrays/callables."""
    out: List[Any] = []
    for c in c_ops_q:
        if isinstance(c, list) and len(c) == 2:
            op, f = c
            out.append([_qobj_to_np(op), f])
        else:
            out.append(_qobj_to_np(c))
    return out


@dataclass
class CompiledOpenSystemNumpy:
    r"""NumPy representation of a compiled open quantum system."""

    H: Any  # np.ndarray or list mix as above
    c_ops: List[Any]
    args: Dict[str, Any]
    config: HilbertConfig
    time_dependent: bool


def compile_open_system_from_latex(
    H_latex: str,
    params: Dict[str, complex],
    *,
    config: HilbertConfig,
    c_ops_latex: List[str] | None = None,
    t_name: str = "t",
    time_symbols: tuple[str, ...] | None = None,
) -> CompiledOpenSystemNumpy:
    r"""
    Compile a LaTeX-defined open system to NumPy arrays by reusing the QuTiP backend.
    """
    compiled_q: CompiledOpenSystemQutip = _compile_qutip(
        H_latex=H_latex,
        params=params,
        config=config,
        c_ops_latex=c_ops_latex or [],
        t_name=t_name,
        time_symbols=time_symbols,
    )

    H0, time_terms = _convert_H(compiled_q.H)
    H_np: Any
    if time_terms:
        H_np = [H0] + [([Hk, fk] if Hk is not None else fk) for Hk, fk in time_terms]
    else:
        H_np = H0

    c_ops_np = _convert_c_ops(compiled_q.c_ops)

    return CompiledOpenSystemNumpy(
        H=H_np,
        c_ops=c_ops_np,
        args=compiled_q.args,
        config=compiled_q.config,
        time_dependent=compiled_q.time_dependent,
    )
