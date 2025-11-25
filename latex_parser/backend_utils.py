from __future__ import annotations

import sympy as sp

from latex_parser.dsl import (
    DSLValidationError,
    HilbertConfig,
    try_parse_operator_symbol,
)
from latex_parser.ir import HamiltonianIR, OperatorFunctionRef

__all__ = [
    "param_aliases",
    "lookup_param_name",
    "resolve_param",
    "collect_parameter_names",
    "validate_required_params",
    "expr_has_time",
]


def param_aliases(name: str) -> list[str]:
    r"""Generate alias spellings for a parameter name."""
    name_str = str(name)
    candidates: list[str] = [name_str]
    no_braces = name_str.replace("{", "").replace("}", "")
    if no_braces not in candidates:
        candidates.append(no_braces)
    no_star = no_braces.replace("*", "")
    if no_star not in candidates:
        candidates.append(no_star)
    no_spaces = no_braces.replace(" ", "")
    if no_spaces not in candidates:
        candidates.append(no_spaces)
    return candidates


def lookup_param_name(name: str, params: dict[str, complex]) -> tuple[str, complex]:
    r"""Resolve a parameter name using aliases and return ``(key, value)``."""
    return resolve_param(name, params)


def resolve_param(
    name: str,
    params: dict[str, complex],
    *,
    warn_on_multiple: bool = False,
    logger: object | None = None,
) -> tuple[str, complex]:
    r"""
    Resolve ``name`` in ``params`` using :func:`param_aliases`.

    Parameters
    ----------
    name : str
        Symbol name to resolve.
    params : dict
        Parameter dictionary provided by the user.
    warn_on_multiple : bool, optional
        If True, emit a warning via ``logger`` when multiple aliases
        match and the first match is chosen.
    logger : logging.Logger or None, optional
        Logger used for warnings when ``warn_on_multiple`` is True.

    Returns
    -------
    (str, complex)
        Tuple of the matched key in ``params`` and its value.

    Raises
    ------
    DSLValidationError
        If no alias matches.
    """
    candidates = param_aliases(name)
    matches = [key for key in candidates if key in params]
    if matches:
        if warn_on_multiple and logger is not None and len(matches) > 1:
            logger.warning(
                "Parameter name '%s' matched multiple keys %s; using first match '%s'.",
                name,
                matches,
                matches[0],
            )
        key = matches[0]
        return key, params[key]
    raise DSLValidationError(
        f"Missing numeric value for scalar symbol '{name}' in parameters dict. "
        f"Tried keys: {candidates}. Available keys: {sorted(params.keys())}."
    )


def collect_parameter_names(
    ir: HamiltonianIR, config: HilbertConfig, time_names: set[str]
) -> set[str]:
    r"""
    Collect scalar parameter names referenced in an IR, excluding time symbols.

    Operator symbols are skipped by reusing :func:`try_parse_operator_symbol`
    so that only scalar parameters are returned.
    """
    names: set[str] = set()
    for term in ir.terms:
        for sym in term.scalar_expr.free_symbols:
            if sym.name in time_names:
                continue
            if try_parse_operator_symbol(sym, config):
                continue
            names.add(sym.name)
        for op in term.ops:
            if isinstance(op, OperatorFunctionRef):
                for sym in op.scalar_factor.free_symbols:
                    if sym.name in time_names:
                        continue
                    if try_parse_operator_symbol(sym, config):
                        continue
                    names.add(sym.name)
    return names


def validate_required_params(
    required: set[str], params: dict[str, complex], time_names: set[str]
) -> None:
    r"""
    Ensure all ``required`` symbols can be resolved from ``params`` via aliases.

    ``time_names`` are ignored when present in ``required``. Raises a single
    :class:`DSLValidationError` listing all missing symbols to improve UX.
    """
    # SymPy built-ins that never require user-provided values.
    builtin_symbols = {"I", "E", "pi"}
    missing: list[str] = []
    for name in sorted(required):
        if name in time_names:
            continue
        if name in builtin_symbols:
            continue
        try:
            resolve_param(name, params)
        except DSLValidationError:
            missing.append(name)
    if missing:
        raise DSLValidationError(
            "Missing required parameters: "
            f"{missing}. Provided keys: {sorted(params.keys())}."
        )


def expr_has_time(expr: sp.Expr, time_names: set[str]) -> bool:
    r"""Return True if ``expr`` references any symbol in ``time_names``."""
    return any(s.name in time_names for s in expr.free_symbols)
