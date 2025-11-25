# flake8: noqa
"""
Example: defining a custom backend by subclassing BackendBase.

This backend uses NumPy to build simple diagonal matrices from IR terms,
and demonstrates how to register a custom LaTeX macro and operator function.

It also shows how to reuse `BaseOperatorCache` for subsystem bookkeeping,
so backend authors can avoid duplicating identity/kron logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.backend_base import BackendBase, BackendOptions
from latex_parser.backend_cache import BaseOperatorCache
from latex_parser.dsl import CustomSpec, HilbertConfig, register_operator_macro
from latex_parser.dsl_constants import register_operator_function
from latex_parser.ir import latex_to_ir


class NumpyOperatorCache(BaseOperatorCache[np.ndarray]):
    """Minimal cache that builds identity tensors for NumPy backends."""

    def _local_identity(self, dim: int) -> np.ndarray:
        return np.eye(dim)

    def _kron(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.kron(a, b)


class NumpyDiagBackend(BackendBase):
    def _make_cache(self, config: HilbertConfig, options: BackendOptions | None):
        return NumpyOperatorCache(config)

    def _compile_static(self, ir, cache, params, options=None):
        diag = []
        for term in ir.terms:
            coeff = complex(term.scalar_expr.subs(params))
            diag.append(coeff)
        if not diag:
            diag = [0.0]
        return np.diag(diag)

    def _compile_time_dependent(
        self, ir, cache, params, *, t_name, time_symbols, options=None
    ):
        return self._compile_static(ir, cache, params, options=options)


if __name__ == "__main__":
    # Register a simple macro foo_j -> Jx_j and allow sinh
    register_operator_macro("foo", "Jx")
    register_operator_function("sinh")

    # Custom subsystem with Jx operator
    Jx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    custom = CustomSpec(label="c", index=1, dim=2, operators={"Jx": Jx})
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[custom])

    ir = latex_to_ir(r"\sinh(\foo_{1})", cfg)
    backend = NumpyDiagBackend()
    H = backend.compile_static_from_ir(ir, cfg, params={})
    print("Compiled Hamiltonian (diagonal):\n", H)
