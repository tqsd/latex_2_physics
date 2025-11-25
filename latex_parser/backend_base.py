from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import sympy as sp

from latex_parser.dsl import DSLValidationError, HilbertConfig
from latex_parser.ir import HamiltonianIR, OperatorFunctionRef


@dataclass
class BackendOptions:
    r"""Common options passed to backends; extend in subclasses as needed."""

    dtype: Any | None = None
    platform: str | None = None


@dataclass
class CompiledHamiltonianBase:
    r"""Backend-agnostic compiled Hamiltonian."""

    H: Any
    H0: Any
    time_terms: List[Tuple[Any, Any]]
    args: Dict[str, Any]
    time_dependent: bool


@dataclass
class CompiledOpenSystemBase:
    r"""Backend-agnostic compiled open quantum system."""

    H: Any
    c_ops: List[Any]
    args: Dict[str, Any]
    config: HilbertConfig
    time_dependent: bool


class BackendBase(abc.ABC):
    r"""
    Abstract backend template; subclasses implement numerical details.
    """

    def compile_static_from_ir(
        self,
        ir: HamiltonianIR,
        config: HilbertConfig,
        params: Dict[str, complex],
        options: BackendOptions | None = None,
    ) -> Any:
        r"""
        Compile a time-independent Hamiltonian IR into backend objects.
        """
        if ir.has_time_dep:
            raise DSLValidationError(
                "compile_static_from_ir called on time-dependent IR; use "
                "compile_time_dependent_from_ir."
            )
        cache = self._make_cache(config, options=options)
        return self._compile_static(ir, cache, params, options=options)

    def compile_time_dependent_from_ir(
        self,
        ir: HamiltonianIR,
        config: HilbertConfig,
        params: Dict[str, complex],
        t_name: str = "t",
        time_symbols: Tuple[str, ...] | None = None,
        options: BackendOptions | None = None,
    ) -> Any:
        r"""
        Compile a time-dependent Hamiltonian IR into backend objects.
        """
        cache = self._make_cache(config, options=options)
        return self._compile_time_dependent(
            ir,
            cache,
            params,
            t_name=t_name,
            time_symbols=time_symbols,
            options=options,
        )

    def compile_collapse_ops_from_latex(
        self,
        c_ops_latex: list[str],
        config: HilbertConfig,
        params: Dict[str, complex],
        t_name: str = "t",
        time_symbols: tuple[str, ...] | None = None,
        options: BackendOptions | None = None,
    ) -> tuple[list[Any], dict[str, Any], bool]:
        r"""
        Compile collapse operators from LaTeX into backend objects.

        Backends that do not support open-system compilation should override
        this method and raise a clear error.
        """
        raise DSLValidationError(
            "Collapse-operator compilation is not supported for this backend."
        )

    @abc.abstractmethod
    def _make_cache(self, config: HilbertConfig, options: BackendOptions | None) -> Any:
        r"""Prepare reusable backend-specific cache/state for compilation."""
        raise NotImplementedError("Backend must implement _make_cache.")

    @abc.abstractmethod
    def _compile_static(
        self,
        ir: HamiltonianIR,
        cache: Any,
        params: Dict[str, complex],
        options: BackendOptions | None = None,
    ) -> Any:
        r"""Compile a static Hamiltonian IR using precomputed cache."""
        raise NotImplementedError("Backend must implement _compile_static.")

    @abc.abstractmethod
    def _compile_time_dependent(
        self,
        ir: HamiltonianIR,
        cache: Any,
        params: Dict[str, complex],
        *,
        t_name: str,
        time_symbols: Tuple[str, ...] | None,
        options: BackendOptions | None = None,
    ) -> Any:
        r"""Compile a time-dependent Hamiltonian IR using precomputed cache."""
        raise NotImplementedError("Backend must implement _compile_time_dependent.")

    # Shared utilities for subclasses
    @staticmethod
    def _param_aliases(name: str) -> List[str]:
        r"""
        Generate candidate parameter names to match LaTeX-escaped variants.
        """
        from latex_parser.backend_utils import param_aliases

        return param_aliases(name)

    @classmethod
    def _lookup_param_name(
        cls, name: str, params: Dict[str, complex]
    ) -> tuple[str, complex]:
        r"""
        Resolve a scalar symbol name to the key present in ``params``.
        """
        from latex_parser.backend_utils import lookup_param_name

        return lookup_param_name(name, params)

    @staticmethod
    def _expr_has_time(expr: sp.Expr, time_names: set[str]) -> bool:
        r"""Return True if an expression references any time symbol."""
        return any(s.name in time_names for s in expr.free_symbols)

    @staticmethod
    def _is_operator_function_allowed(
        func: OperatorFunctionRef, allowed: set[str]
    ) -> bool:
        r"""Check whether an operator-valued function is permitted."""
        return func.func_name in allowed
