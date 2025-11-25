# flake8: noqa
"""
Backend extensibility walkthrough (hands-on).

This example is aimed at developers who want to:
- Understand the compile pipeline (LaTeX → IR → backend).
- Register a custom backend that uses BaseOperatorCache.
- Reuse the shared backend registry and parameter validation.

It contains several self-contained steps you can run individually by calling
the functions from a Python shell. Nothing runs on import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.backend_base import BackendBase, BackendOptions
from latex_parser.backend_cache import BaseOperatorCache
from latex_parser.compile_core import (
    available_backends,
    compile_model_core,
    register_backend,
)
from latex_parser.dsl import CustomSpec, HilbertConfig, QubitSpec
from latex_parser.ir import latex_to_ir


class NumpyCache(BaseOperatorCache[np.ndarray]):
    """BaseOperatorCache specialization for NumPy arrays."""

    def _local_identity(self, dim: int) -> np.ndarray:
        return np.eye(dim)

    def _kron(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.kron(a, b)


@dataclass
class NumpyDiagOptions(BackendOptions):
    """Options for the demo backend."""

    dtype: Any = complex


class NumpyDiagBackend(BackendBase):
    """
    Diagonal-only backend.

    For each IR term, we evaluate the scalar and place it on the diagonal of a
    dense NumPy matrix. This is intentionally simple but shows the required
    BackendBase hooks.
    """

    def _make_cache(
        self, config: HilbertConfig, options: BackendOptions | None
    ) -> NumpyCache:
        return NumpyCache(config)

    def _compile_static(
        self,
        ir,
        cache: NumpyCache,
        params: Dict[str, complex],
        options: BackendOptions | None = None,
    ) -> np.ndarray:
        opts = options if isinstance(options, NumpyDiagOptions) else NumpyDiagOptions()
        diag: list[complex] = []
        for term in ir.terms:
            coeff = complex(term.scalar_expr.subs(params))
            diag.append(coeff)
        if not diag:
            diag = [0.0]
        return np.diag(np.asarray(diag, dtype=opts.dtype))

    def _compile_time_dependent(
        self,
        ir,
        cache: NumpyCache,
        params: Dict[str, complex],
        *,
        t_name: str,
        time_symbols: tuple[str, ...] | None,
        options: BackendOptions | None = None,
    ) -> np.ndarray:
        # This backend ignores time dependence and just reuses the static path.
        return self._compile_static(ir, cache, params, options=options)


def register_demo_backend() -> None:
    """
    Register the diagonal backend under the name "numpy_diag".
    After calling this, compile_model_core(..., backend="numpy_diag") works.
    """

    def _compiler(
        *,
        H_latex: str,
        params: Dict[str, complex],
        config: HilbertConfig,
        c_ops_latex,
        t_name: str,
        time_symbols,
    ):
        backend = NumpyDiagBackend()
        ir = latex_to_ir(H_latex, config, t_name=t_name, time_symbols=time_symbols)
        cache = backend._make_cache(config, options=None)
        return backend._compile_time_dependent(
            ir,
            cache,
            params,
            t_name=t_name,
            time_symbols=time_symbols,
            options=None,
        )

    register_backend("numpy_diag", _compiler)


def demo_compile_with_registered_backend() -> None:
    """
    Compile a 2x2 diagonal Hamiltonian with the registered backend.
    """
    if "numpy_diag" not in available_backends():
        register_demo_backend()

    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\Delta \sigma_{z,1}"
    params = {"Delta": 1.25}
    H_np = compile_model_core(
        backend="numpy_diag",
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=None,
        t_name="t",
        time_symbols=None,
    )
    print("Available backends:", available_backends())
    print("Diagonal matrix:\n", H_np)


def demo_custom_subsystem() -> None:
    """
    Use the diagonal backend with a custom subsystem (Jx/Jy/Jz).
    """
    if "numpy_diag" not in available_backends():
        register_demo_backend()

    Jx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Jy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    Jz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    custom = CustomSpec(
        label="c",
        index=1,
        dim=2,
        operators={"Jx": Jx, "Jy": Jy, "Jz": Jz},
    )
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[custom])
    H_latex = r"\alpha Jx_{1} + \beta Jy_{1} + \gamma Jz_{1}"
    params = {"alpha": 0.1, "beta": 0.2, "gamma": 0.3}
    H_np = compile_model_core(
        backend="numpy_diag",
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=None,
        t_name="t",
        time_symbols=None,
    )
    print("Custom subsystem diagonal matrix:\n", H_np)


if __name__ == "__main__":
    register_demo_backend()
    demo_compile_with_registered_backend()
    demo_custom_subsystem()
