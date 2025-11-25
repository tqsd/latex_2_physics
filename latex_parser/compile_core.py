from __future__ import annotations

"""
Backend dispatch and configuration validation.

This module holds the registry used by :func:`compile_model_core` to route
LaTeX compilation to concrete backends (QuTiP, NumPy, JAX, or user-provided).
It also performs early parameter validation by parsing the LaTeX into IR and
ensuring that every scalar symbol can be resolved via the standard alias rules.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Sequence

import latex_parser.backend_jax as backend_jax  # type: ignore
import latex_parser.backend_numpy as backend_numpy  # type: ignore
import latex_parser.backend_qutip as backend_qutip  # type: ignore
from latex_parser.backend_utils import (
    collect_parameter_names,
    validate_required_params,
)
from latex_parser.dsl import DSLValidationError, HilbertConfig
from latex_parser.ir import latex_to_ir

BackendCompiler = Callable[..., Any]


@dataclass
class BackendRegistration:
    """
    Registry entry describing one backend compiler.

    Attributes
    ----------
    compiler : callable
        Backend entry point matching the ``compile_open_system_from_latex``
        signature.
    capabilities : dict or None
        Optional lightweight capability flags (informational only).
    config_schema : dict or None
        Optional schema describing backend-specific configuration (future use).
    """

    compiler: BackendCompiler
    capabilities: Optional[dict[str, Any]] = None
    config_schema: Optional[dict[str, type]] = None


# Default registry; users can extend via register_backend.
_BACKEND_REGISTRY: Dict[str, BackendRegistration] = {
    "qutip": BackendRegistration(
        compiler=backend_qutip.compile_open_system_from_latex,
        capabilities={"time_dependent": True, "open_systems": True},
    ),
    "numpy": BackendRegistration(
        compiler=backend_numpy.compile_open_system_from_latex,
        capabilities={"time_dependent": False, "open_systems": False},
    ),
    "jax": BackendRegistration(
        compiler=backend_jax.compile_open_system_from_latex,
        capabilities={"time_dependent": True, "autodiff": True},
    ),
}


def register_backend(name: str, compiler: BackendCompiler) -> None:
    """
    Register a backend compiler callable.

    The callable must accept keyword arguments
    ``(H_latex, params, config, c_ops_latex, t_name, time_symbols)``.
    """
    _BACKEND_REGISTRY[name.lower()] = BackendRegistration(compiler=compiler)


def available_backends() -> list[str]:
    """Return a list of registered backend names."""
    return sorted(_BACKEND_REGISTRY.keys())


def backend_capabilities(name: str) -> dict[str, Any] | None:
    """Return declared capabilities for a registered backend."""
    reg = _BACKEND_REGISTRY.get(name.lower())
    return reg.capabilities if reg else None


def compile_model_core(
    *,
    backend: Literal["qutip", "numpy", "jax"],
    H_latex: str,
    params: Dict[str, complex],
    config: HilbertConfig,
    c_ops_latex: Sequence[str] | None,
    t_name: str,
    time_symbols: Sequence[str] | None,
):
    """
    Dispatch compilation to the selected backend.

    Steps:

    - Normalize backend name and look up the registry entry.
    - Parse Hamiltonian into IR and collect scalar parameter names.
    - Parse each collapse operator (if any) into IR and union their
      parameter names with those from the Hamiltonian.
    - Validate that each required parameter can be resolved from the provided
      params dict via alias rules (e.g., ``omega`` vs ``omega_{}``).
    - Invoke the backend compiler with the original inputs.
    """
    backend_norm = backend.lower()
    reg = _BACKEND_REGISTRY.get(backend_norm)
    if reg is None:
        raise DSLValidationError(
            f"Unknown backend '{backend}'. Available: {sorted(_BACKEND_REGISTRY)}."
        )

    time_set = {t_name} | (set(time_symbols) if time_symbols else set())
    ir = latex_to_ir(H_latex, config, t_name=t_name, time_symbols=time_symbols)
    required_params = collect_parameter_names(ir, config, time_set)

    for c_latex in c_ops_latex or []:
        ir_c = latex_to_ir(c_latex, config, t_name=t_name, time_symbols=time_symbols)
        required_params |= collect_parameter_names(ir_c, config, time_set)

    validate_required_params(required_params, params, time_set)

    return reg.compiler(
        H_latex=H_latex,
        params=params,
        config=config,
        c_ops_latex=c_ops_latex or [],
        t_name=t_name,
        time_symbols=tuple(time_symbols) if time_symbols else None,
    )
