"""
User-facing API for compiling physics-style LaTeX into backend-ready models.

This module intentionally stays thin: it validates/simplifies user input and
delegates all parsing, configuration, and backend work to core utilities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Sequence

from latex_parser.backend_qutip import CompiledOpenSystemQutip
from latex_parser.compile_core import compile_model_core
from latex_parser.config_utils import make_config, resolve_config
from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    HilbertConfig,
    QubitSpec,
    DSLValidationError,
)
from latex_parser.ir import latex_to_ir
from latex_parser.auto_config import infer_config_from_latex

logger = logging.getLogger(__name__)

__all__ = [
    "make_config",
    "compile_model",
    "lint_latex_model",
    "CompiledOpenSystemQutip",
    "HilbertConfig",
    "CustomSpec",
    "BosonSpec",
    "QubitSpec",
    "DSLValidationError",
    "infer_config_from_latex",
]


def compile_model(
    H_latex: str,
    params: Dict[str, complex],
    *,
    backend: Literal["qutip", "numpy", "jax"] = "qutip",
    config: HilbertConfig | None = None,
    c_ops_latex: List[str] | None = None,
    qubits: int | Sequence[int] = 0,
    bosons: Sequence[int] | Sequence[tuple[int, str]] | None = None,
    customs: Sequence[CustomSpec] | None = None,
    t_name: str = "t",
    time_symbols: tuple[str, ...] | None = None,
    auto_config: bool = False,
    default_boson_cutoff: int = 2,
    custom_templates: Dict[str, CustomSpec] | None = None,
    diagnostics: bool = False,
) -> Any:
    r"""
    Compile a LaTeX-defined model to the chosen backend.

    Set ``diagnostics=True`` to also return parsing diagnostics
    ``(model, diagnostics_dict)`` where diagnostics contains
    ``term_count``, ``time_symbols``, and ``time_dependent``.
    """
    cfg = resolve_config(
        config=config,
        auto_config=auto_config,
        H_latex=H_latex,
        c_ops_latex=c_ops_latex,
        qubits=qubits,
        bosons=bosons or [],
        customs=customs,
        default_boson_cutoff=default_boson_cutoff,
        custom_templates=custom_templates or {},
    )
    return (
        compile_model_core(
            backend=backend,
            H_latex=H_latex,
            params=params,
            config=cfg,
            c_ops_latex=c_ops_latex or [],
            t_name=t_name,
            time_symbols=time_symbols,
        )
        if not diagnostics
        else _compile_with_diag(
            backend=backend,
            H_latex=H_latex,
            params=params,
            cfg=cfg,
            c_ops_latex=c_ops_latex,
            t_name=t_name,
            time_symbols=time_symbols,
            auto_config=auto_config,
            default_boson_cutoff=default_boson_cutoff,
            custom_templates=custom_templates,
        )
    )


def lint_latex_model(
    H_latex: str,
    *,
    config: HilbertConfig,
    t_name: str = "t",
    time_symbols: tuple[str, ...] | None = None,
) -> Dict[str, Any]:
    r"""
    Analyze a LaTeX Hamiltonian without compiling to a backend.

    Returns structural diagnostics (term count, operator signature, time usage)
    to help users understand parsed content without numerical compilation.
    """
    ir = latex_to_ir(H_latex, config, t_name=t_name, time_symbols=time_symbols)
    sig = {(r.kind, r.op_name, r.index, r.power) for term in ir.terms for r in term.ops}
    time_syms_used = {
        s.name
        for term in ir.terms
        for s in term.scalar_expr.free_symbols
        if s.name in ({t_name} | set(time_symbols or ()))
    }
    return {
        "term_count": len(ir.terms),
        "time_symbols_used": sorted(time_syms_used),
        "has_time_dep": ir.has_time_dep,
        "operator_signature": sorted(sig),
    }


def _compile_with_diag(
    *,
    backend: Literal["qutip", "numpy", "jax"],
    H_latex: str,
    params: Dict[str, complex],
    cfg: HilbertConfig,
    c_ops_latex: List[str] | None,
    t_name: str,
    time_symbols: tuple[str, ...] | None,
    auto_config: bool,
    default_boson_cutoff: int,
    custom_templates: Dict[str, CustomSpec] | None,
) -> tuple[Any, Dict[str, Any]]:
    r"""Internal helper to compile and build diagnostics."""
    model = compile_model_core(
        backend=backend,
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=c_ops_latex or [],
        t_name=t_name,
        time_symbols=time_symbols,
    )
    ir = latex_to_ir(H_latex, cfg, t_name=t_name, time_symbols=time_symbols)
    diag = {
        "term_count": len(ir.terms),
        "time_symbols": [t_name] + list(time_symbols or []),
        "time_dependent": ir.has_time_dep,
    }
    return model, diag
