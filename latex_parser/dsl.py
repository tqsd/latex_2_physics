# dsl.py
#
# Physics-LaTeX DSL for local operators on composite Hilbert spaces.
#
# This module is intentionally narrow in scope:
# - It does NOT build Hamiltonians or QuTiP objects.
# - It does NOT handle term-level powers (a_1^2, n_1^2, ...).
# - It ONLY:
#     * canonicalizes physics-style LaTeX into a restricted internal LaTeX,
#     * parses that LaTeX with SymPy,
#     * maps individual SymPy symbols to LocalOperatorRef objects,
#       given a HilbertConfig that describes which subsystems exist.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import sympy as sp
from sympy.parsing.latex import parse_latex

logger = logging.getLogger(__name__)

Role = Literal["system", "environment"]


class DSLValidationError(Exception):
    r"""Raised when DSL validation fails (bad config or unmappable operator)."""


@dataclass(frozen=True)
class QubitSpec:
    r"""
    Specification for a single qubit subsystem.

    Parameters
    ----------
    label : str
        Logical label for this kind of subsystem. In the current DSL, this
        MUST be "q"; other values are rejected in HilbertConfig.
    index : int
        1-based index (as used in the LaTeX DSL; e.g. ``\sigma_{x,1}``).
    role : Literal["system", "environment"], optional
        The DSL layer does not use this field directly, but it is useful for
        higher-level orchestration.
    """

    label: str
    index: int
    role: Role = "system"

    def __post_init__(self) -> None:
        r"""Validate qubit specification fields."""
        if not self.label:
            raise DSLValidationError("QubitSpec label must be non-empty.")
        if self.index <= 0:
            raise DSLValidationError("QubitSpec index must be positive.")
        if self.role not in ("system", "environment"):
            raise DSLValidationError(
                "QubitSpec role must be 'system' or 'environment'."
            )


@dataclass(frozen=True)
class BosonSpec:
    r"""
    Specification for a single truncated bosonic mode.

    Parameters
    ----------
    label : str
        Logical label for this kind of subsystem. In the current DSL, this
        MUST be "a"; other values are rejected in HilbertConfig.
    index : int
        1-based index (as used in the LaTeX DSL; e.g. ``a_{1}``, ``\hat{n}_{2}``).
    cutoff : int
        Local Hilbert-space dimension (photon-number cutoff).
    role : Literal["system", "environment"], optional
        Role flag; informational only at the DSL layer.
    deformation : callable or None, optional
        Optional callable implementing an f-deformation on the ladder operators.
        Receives an array of number eigenvalues ``n = [0, 1, ... cutoff-1]``
        and must return an array of the same shape representing ``f(n)``. When
        provided, the DSL exposes deformed operators ``\af_{j}`` and
        ``\adagf_{j}`` that map to ``a f(n)`` and ``f(n) a^{\dagger}``,
        respectively. If ``None``, only the standard ``a``, ``a^{\dagger}``,
        and ``\hat{n}`` are available.
    deformation_latex : str or None, optional
        Optional LaTeX string that produced the deformation callable; stored
        for provenance. Ignored by the backend.
    """

    label: str
    index: int
    cutoff: int
    role: Role = "system"
    deformation: Optional[Callable[[object], object]] = None
    deformation_latex: Optional[str] = None

    def __post_init__(self) -> None:
        r"""Validate boson specification fields."""
        if not self.label:
            raise DSLValidationError("BosonSpec label must be non-empty.")
        if self.index <= 0:
            raise DSLValidationError("BosonSpec index must be positive.")
        if self.cutoff <= 0:
            raise DSLValidationError("BosonSpec cutoff must be positive.")
        if self.role not in ("system", "environment"):
            raise DSLValidationError(
                "BosonSpec role must be 'system' or 'environment'."
            )
        if self.deformation is not None and not callable(self.deformation):
            raise DSLValidationError("BosonSpec deformation must be callable or None.")


@dataclass(frozen=True)
class CustomSpec:
    r"""
    Specification for a user-defined finite-dimensional subsystem.

    Parameters
    ----------
    label : str
        Logical label for this kind of subsystem. In the current DSL, this MUST
        be "c"; other values are rejected in HilbertConfig.
    index : int
        1-based index (used as ``A_{index}`` in the DSL).
    dim : int
        Hilbert-space dimension.
    operators : Mapping[str, object]
        Mapping from operator names to backend-specific objects. The DSL only
        cares about the keys; values are opaque here.

        For example::

            operators = {
                "A": some_operator_object,
                "B": another_operator_object,
                "Jx": Jx_matrix,
                "Jp": J_plus_matrix,
            }

        The names here are used when parsing custom operators from symbols like
        ``A_{1}``, ``J_{x,1}``, ``J_{+,1}``, ``J_{-,1}``, etc.
    role : Literal["system", "environment"], optional
        Informational role flag.
    """

    label: str
    index: int
    dim: int
    operators: Mapping[str, object]
    role: Role = "system"

    def __post_init__(self) -> None:
        r"""Validate custom subsystem specification fields."""
        if not self.label:
            raise DSLValidationError("CustomSpec label must be non-empty.")
        if self.index <= 0:
            raise DSLValidationError("CustomSpec index must be positive.")
        if self.dim <= 0:
            raise DSLValidationError("CustomSpec dim must be positive.")
        if self.role not in ("system", "environment"):
            raise DSLValidationError(
                "CustomSpec role must be 'system' or 'environment'."
            )
        if not isinstance(self.operators, Mapping) or not self.operators:
            raise DSLValidationError(
                "CustomSpec.operators must be a non-empty mapping."
            )


@dataclass
class HilbertConfig:
    r"""
    Minimal configuration for the DSL layer.

    Parameters
    ----------
    qubits : list[QubitSpec]
        Qubit subsystems (label is user-defined, default ``q``).
    bosons : list[BosonSpec]
        Bosonic modes with cutoffs (label is user-defined, default ``a``).
    customs : list[CustomSpec]
        Custom subsystems with user-defined operators (label is user-defined,
        default ``c``).

    Notes
    -----
    Labels are flexible; lookups fall back to matching by (kind, index) if
    labels differ between LaTeX and configuration.
    """

    qubits: List[QubitSpec]
    bosons: List[BosonSpec]
    customs: List[CustomSpec]

    def all_subsystems(
        self,
    ) -> List[Tuple[str, Union[QubitSpec, BosonSpec, CustomSpec]]]:
        r"""
        Return a flat list of (kind, spec) entries.

        kind is one of: "qubit", "boson", "custom".
        """
        result: List[Tuple[str, Union[QubitSpec, BosonSpec, CustomSpec]]] = []
        for q in self.qubits:
            result.append(("qubit", q))
        for b in self.bosons:
            result.append(("boson", b))
        for c in self.customs:
            result.append(("custom", c))
        return result

    def __post_init__(self) -> None:
        r"""Validate the HilbertConfig entries for type, uniqueness, and indices."""
        seen = set()
        for kind, spec in self.all_subsystems():
            if kind == "qubit" and not isinstance(spec, QubitSpec):
                raise DSLValidationError(
                    f"Qubit entries must be QubitSpec; got {type(spec)!r}."
                )
            if kind == "boson" and not isinstance(spec, BosonSpec):
                raise DSLValidationError(
                    f"Boson entries must be BosonSpec; got {type(spec)!r}."
                )
            if kind == "custom" and not isinstance(spec, CustomSpec):
                raise DSLValidationError(
                    f"Custom entries must be CustomSpec; got {type(spec)!r}."
                )
            if not hasattr(spec, "label") or not hasattr(spec, "index"):
                raise DSLValidationError(
                    f"Subsystem entries must have label and index; got {spec!r}."
                )
            key = (kind, spec.label, spec.index)
            if key in seen:
                raise DSLValidationError(f"Duplicate subsystem {key} in HilbertConfig.")
            seen.add(key)
        # Optional index gap logging (non-fatal)
        for kind, specs in (
            ("qubit", self.qubits),
            ("boson", self.bosons),
            ("custom", self.customs),
        ):
            by_label: Dict[str, List[int]] = {}
            for spec in specs:
                by_label.setdefault(spec.label, []).append(spec.index)
            for label, indices in by_label.items():
                sorted_idx = sorted(indices)
                expected = list(range(1, len(sorted_idx) + 1))
                if sorted_idx != expected:
                    logger.debug(
                        "Non-sequential indices for %s label %s: %s (expected %s)",
                        kind,
                        label,
                        sorted_idx,
                        expected,
                    )


@dataclass(frozen=True)
class LocalOperatorRef:
    r"""
    Reference to a local operator acting on a single subsystem.

    Attributes
    ----------
    kind : Literal["qubit", "boson", "custom"]
        Subsystem kind.
    label : str
        Subsystem label.
    index : int
        1-based subsystem index (as in the DSL/LaTeX).
    op_name : str
        Operator name (e.g., ``a``, ``adag``, ``af``, ``adagf``, ``n``,
        ``sx``, ``Jx``).
    power : int
        Integer power of the operator (default ``1``). Higher powers are set
        later by the IR layer when parsing ``Pow`` nodes.
    """

    kind: Literal["qubit", "boson", "custom"]
    label: str
    index: int
    op_name: str
    power: int = 1


"""
Operator macro rewriting (LaTeX → internal DSL)
-----------------------------------------------

We keep the user-facing LaTeX as close to physics notation as possible:

    * **Bosons** (mode index ``j``): ``a_{j}``, ``a_{j}^{\\dagger}``, ``\\hat{n}_{j}``
    * **Qubits** (index ``j``): ``\\sigma_{x,j}``, ``\\sigma_{y,j}``, ``\\sigma_{z,j}``,
      ``\\sigma_{+,j}``, ``\\sigma_{-,j}``
    * **Custom spin-like subsystems** (index ``j``):
      ``J_{x,j}``, ``J_{y,j}``, ``J_{z,j}``, ``J_{+,j}``, ``J_{-,j}``,
      ``J_x^{(j)}``, ``J_y^{(j)}``, ``J_z^{(j)}``

To make these easier for SymPy's LaTeX parser and our DSL, we rewrite specific
patterns into simpler internal macros that SymPy parses as symbols with
subscript indices. For example::

    a_{j}^{\\dagger}   -> \\adag_{j}
    \\hat{n}_{j}       -> n_{j}
    \\sigma_{x,j}      -> \\sx_{j}
    J_{x,j}           -> \\Jx_{j}
    J_{+,j}           -> \\Jp_{j}

These unknown macros (``\\adag``, ``\\sx``, ``\\Jx``, …) are interpreted by
``sympy.parsing.latex.parse_latex`` as symbols with names like ``adag_{1}``,
``sx_{1}``, ``Jx_{1}``, which we then normalize and feed to
``parse_operator_symbol`` for IR construction.
"""

_LATEX_OP_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Bosonic number operator in DSL notation:
    #   \hat{n}_{j} or \hat{n}_j  ->  n_{j}
    (
        re.compile(r"\\hat\s*\{\s*n\s*\}\s*_\s*\{?\s*(\d+)\s*\}?"),
        r"n_{\1}",
    ),
    # (Optionally: handle \hat n_{j} without braces around n)
    (
        re.compile(r"\\hat\s*n\s*_\s*\{?\s*(\d+)\s*\}?"),
        r"n_{\1}",
    ),
    # Bosonic annihilation with hat: \hat{a}_{j} or \hat a_j -> a_j
    (re.compile(r"\\hat\s*\{\s*a\s*\}\s*_\s*\{?\s*(\d+)\s*\}?"), r"a_{\1}"),
    (re.compile(r"\\hat\s*a\s*_\s*\{?\s*(\d+)\s*\}?"), r"a_{\1}"),
    # Bosonic creation: a_{j}^{\dagger} or a_j^{\dagger} -> \adag_{j}
    (
        re.compile(r"a_\{?(\d+)\}?\s*\^\s*\{?\\dagger\}?"),
        r"\\adag_{\1}",  # a_{1}^\dagger, a_1^\dagger, a_{1}^{\dagger}
    ),
    # Bosonic creation with hat: \hat{a}_{j}^{\dagger}
    (
        re.compile(
            r"\\hat\s*\{\s*a\s*\}\s*_\s*\{?\s*(\d+)\s*\}?\\?\s*\^\s*\{?\\dagger\}?"
        ),
        r"\\adag_{\1}",
    ),
    # Optional f-deformed bosonic ladder operators:
    #   \tilde{a}_{j} or a_{j}^{(f)} -> \af_{j}
    #   \tilde{a}_{j}^{\dagger}     -> \adagf_{j}
    (
        re.compile(
            r"\\tilde\s*\{?\s*a\s*\}?\s*_\s*\{?\s*(\d+)\s*\}?\\?\s*\^\s*\{?\\dagger\}?"
        ),
        r"\\adagf_{\1}",
    ),
    (
        re.compile(r"\\tilde\s*\{?\s*a\s*\}?\s*_\s*\{?\s*(\d+)\s*\}?"),
        r"\\af_{\1}",
    ),
    (re.compile(r"a_\{?(\d+)\}?\s*\^\s*\{\s*\(f\)\s*\}"), r"\\af_{\1}"),
    (
        re.compile(r"a_\{?(\d+)\}?\s*\^\s*\{\s*\\dagger\s*\(\s*f\s*\)\s*\}"),
        r"\\adagf_{\1}",
    ),
    # Pauli X,Y,Z with comma: \sigma_{x,1}, ...
    (re.compile(r"\\sigma_\{x,(\d+)\}"), r"\\sx_{\1}"),
    (re.compile(r"\\sigma_\{y,(\d+)\}"), r"\\sy_{\1}"),
    (re.compile(r"\\sigma_\{z,(\d+)\}"), r"\\sz_{\1}"),
    # Pauli X,Y,Z with space instead of comma: \sigma_{x 1}, ...
    (re.compile(r"\\sigma_\{x\s+(\d+)\}"), r"\\sx_{\1}"),
    (re.compile(r"\\sigma_\{y\s+(\d+)\}"), r"\\sy_{\1}"),
    (re.compile(r"\\sigma_\{z\s+(\d+)\}"), r"\\sz_{\1}"),
    # Pauli X,Y,Z with parenthesized superscript index: \sigma_x^{(1)}, ...
    (re.compile(r"\\sigma_x\^\{\((\d+)\)\}"), r"\\sx_{\1}"),
    (re.compile(r"\\sigma_y\^\{\((\d+)\)\}"), r"\\sy_{\1}"),
    (re.compile(r"\\sigma_z\^\{\((\d+)\)\}"), r"\\sz_{\1}"),
    # Pauli ladder with comma: \sigma_{+,1}, \sigma_{-,1}
    (re.compile(r"\\sigma_\{\+,(\d+)\}"), r"\\sp_{\1}"),
    (re.compile(r"\\sigma_\{-,(\d+)\}"), r"\\sm_{\1}"),
    # Pauli ladder with space: \sigma_{+ 1}, \sigma_{- 1}
    (re.compile(r"\\sigma_\{\+\s+(\d+)\}"), r"\\sp_{\1}"),
    (re.compile(r"\\sigma_\{-\s+(\d+)\}"), r"\\sm_{\1}"),
    # Pauli ladder with parenthesized superscript index: \sigma_+^{(1)}, ...
    (re.compile(r"\\sigma_\+\^\{\((\d+)\)\}"), r"\\sp_{\1}"),
    (re.compile(r"\\sigma_-\^\{\((\d+)\)\}"), r"\\sm_{\1}"),
    # Collective spin operators Jx, Jy, Jz, J± on custom subsystems
    # Jx,Jy,Jz with comma: J_{x,1}, J_{y,1}, J_{z,1}
    (re.compile(r"J_\{x,(\d+)\}"), r"\\Jx_{\1}"),
    (re.compile(r"J_\{y,(\d+)\}"), r"\\Jy_{\1}"),
    (re.compile(r"J_\{z,(\d+)\}"), r"\\Jz_{\1}"),
    # Jx,Jy,Jz with space instead of comma: J_{x 1}, ...
    (re.compile(r"J_\{x\s+(\d+)\}"), r"\\Jx_{\1}"),
    (re.compile(r"J_\{y\s+(\d+)\}"), r"\\Jy_{\1}"),
    (re.compile(r"J_\{z\s+(\d+)\}"), r"\\Jz_{\1}"),
    # Jx,Jy,Jz with parenthesized superscript index: J_x^{(1)}, ...
    (re.compile(r"J_x\^\{\((\d+)\)\}"), r"\\Jx_{\1}"),
    (re.compile(r"J_y\^\{\((\d+)\)\}"), r"\\Jy_{\1}"),
    (re.compile(r"J_z\^\{\((\d+)\)\}"), r"\\Jz_{\1}"),
    # J+,J- with comma: J_{+,1}, J_{-,1}
    (re.compile(r"J_\{\+,(\d+)\}"), r"\\Jp_{\1}"),
    (re.compile(r"J_\{-,(\d+)\}"), r"\\Jm_{\1}"),
    # J+,J- with space: J_{+ 1}, J_{- 1}
    (re.compile(r"J_\{\+\s+(\d+)\}"), r"\\Jp_{\1}"),
    (re.compile(r"J_\{-\s+(\d+)\}"), r"\\Jm_{\1}"),
    # J+,J- with parenthesized superscript index: J_+^{(1)}, J_-^{(1)}
    (re.compile(r"J_\+\^\{\((\d+)\)\}"), r"\\Jp_{\1}"),
    (re.compile(r"J_-\^\{\((\d+)\)\}"), r"\\Jm_{\1}"),
]


def register_latex_pattern(pattern: str, replacement: str) -> None:
    r"""
    Allow users to add custom LaTeX rewrite patterns to the DSL canonicalizer.
    """
    compiled = re.compile(pattern)
    _LATEX_OP_PATTERNS.append((compiled, replacement))


def register_operator_macro(alias: str, target: str) -> None:
    r"""
    Register a simple operator macro alias of the form \\alias_{j} -> \\target_{j}.
    """
    pattern = rf"\\{alias}_\{{?(\d+)\}}?"
    replacement = rf"\\{target}_{{\1}}"
    register_latex_pattern(pattern, replacement)


def register_latex_patterns(patterns: List[tuple[str, str]]) -> None:
    r"""Bulk register multiple LaTeX rewrite patterns."""
    for pattern, repl in patterns:
        register_latex_pattern(pattern, repl)


def get_registered_latex_patterns() -> List[Tuple[re.Pattern[str], str]]:
    r"""Return a copy of the registered LaTeX rewrite patterns."""
    return list(_LATEX_OP_PATTERNS)


# Functions/words we should NOT wrap as bare identifiers. Keep this minimal;
# common math functions are intentionally left out so that bare 'cos', 'sin',
# etc. are wrapped into function-like macros (\\cos, \\sin) for SymPy.
_BARE_SCALAR_SKIP: set[str] = set()


def _wrap_bare_identifiers(text: str) -> str:
    r"""
    Wrap bare multi-letter identifiers (e.g. nbar, kappa) as unknown macros
    so SymPy treats them as single scalar symbols instead of products of
    single-letter symbols.

    Example:
        "nbar" -> "\\nbar"
        "kappa" -> "\\kappa"

    We skip known function-like words (exp, sin, cos, ...).
    """

    def repl(match: re.Match[str]) -> str:
        r"""Wrap a matched bare identifier as a LaTeX macro."""
        word = match.group(1)
        if word in _BARE_SCALAR_SKIP:
            return word
        return f"\\{word}"

    # Match bare words of length >=2 consisting of letters, not preceded by
    # a backslash/letter/digit, and not followed by a letter/digit.
    pattern = re.compile(r"(?<![\\A-Za-z0-9])([A-Za-z]{2,})(?![A-Za-z0-9])")
    return pattern.sub(repl, text)


def _collapse_whitespace(text: str) -> str:
    r"""Collapse whitespace across lines for consistent parsing."""
    return " ".join(line.strip() for line in text.strip().splitlines())


def _apply_operator_patterns(
    text: str, patterns: Optional[List[Tuple[re.Pattern[str], str]]] = None
) -> str:
    """
    Apply registered operator rewrite patterns to the LaTeX text.

    `patterns` can include extra ad-hoc entries to ease experimentation
    with new syntax (e.g., custom sums or macros) without changing core code.
    """
    all_patterns = list(_LATEX_OP_PATTERNS)
    if patterns:
        all_patterns.extend(patterns)
    for pattern, repl in all_patterns:
        text = pattern.sub(repl, text)
    return text


def canonicalize_physics_latex(
    latex_str: str,
    *,
    extra_patterns: Optional[List[Tuple[re.Pattern[str], str]]] = None,
) -> str:
    r"""
    Normalize physics-style LaTeX into the internal DSL macro form.

    Parameters
    ----------
    latex_str : str
        LaTeX containing operator symbols such as ``a_{j}``,
        ``a_{j}^{\dagger}``, ``\hat{n}_{j}``, ``\sigma_{x,j}``,
        ``\sigma_{+,j}``, ``J_{x,j}``, and scalar symbols.

    Returns
    -------
    str
        Canonicalized LaTeX with operators rewritten to internal macros
        (``\adag_{j}``, ``\sx_{j}``, etc.), bare multi-letter scalars wrapped
        as single symbols, and whitespace collapsed.

    Examples
    --------
    >>> canonicalize_physics_latex(r"a_{1}^{\\dagger} \\sigma_{-,1}")
    '\\\\adag_{1} \\\\sm_{1}'
    >>> canonicalize_physics_latex(r"\\sigma_{x,1} + \\sigma_{x 2}")
    '\\\\sx_{1} + \\\\sx_{2}'
    """
    text = _collapse_whitespace(latex_str)
    # First, wrap bare identifiers like "nbar" into unknown macros so they
    # become single symbols under SymPy's parser.
    text = _wrap_bare_identifiers(text)
    return _apply_operator_patterns(text, patterns=extra_patterns)


def make_finite_sum_pattern(
    index_var: str, op_base: str, start: int, end: int
) -> Tuple[re.Pattern[str], str]:
    r"""
    Build a LaTeX rewrite pattern for a simple finite sum of identical operators.

    Example
    -------
    >>> pattern, repl = make_finite_sum_pattern("j", "a", 1, 3)
    >>> canonicalize_physics_latex(
    ...     r"\\sum_{j=1}^{3} a_{j}", extra_patterns=[(pattern, repl)]
    ... )
    '\\\\a_{1} + \\\\a_{2} + \\\\a_{3}'
    """
    if start > end:
        raise DSLValidationError("Sum start must be <= end.")
    expansion = " + ".join([f"{op_base}_{{{i}}}" for i in range(start, end + 1)])
    pattern = re.compile(
        rf"\\sum_\{{\s*{index_var}\s*=\s*{start}\s*\}}\^\{{\s*{end}\s*\}}"
        rf"\s*{op_base}_\{{\s*{index_var}\s*\}}"
    )
    return pattern, expansion


def deformation_callable_from_latex(
    latex: str, cutoff: int
) -> Tuple[str, Callable[[object], object]]:
    r"""
    Build a deformation callable from LaTeX in the scalar variable ``n``.

    Returns the canonicalized LaTeX and the callable that evaluates ``f(n)`` on
    integer eigenvalues ``0..cutoff-1``.
    """
    n_sym = sp.Symbol("n")
    canonical = canonicalize_physics_latex(latex)
    expr = parse_latex(canonical)

    replacements = {}
    for s in expr.free_symbols:
        name = s.name.replace("{", "").replace("}", "")
        if name == "n" or name.startswith("n_"):
            replacements[s] = n_sym
        elif name.lower() == "i":
            replacements[s] = sp.I
        else:
            raise DSLValidationError(
                f"Deformation LaTeX may depend only on n; found symbol '{name}'."
            )
    if replacements:
        expr = expr.xreplace(replacements)

    free = {str(s) for s in expr.free_symbols}
    if free not in (set(), {"n"}):
        raise DSLValidationError(
            f"Deformation LaTeX may depend only on n; free symbols: {sorted(free)}."
        )

    deform_fn = sp.lambdify(n_sym, expr, modules="numpy")

    def deformation(n_values: object) -> object:
        r"""Evaluate deformation on provided eigenvalues array."""
        arr = np.asarray(n_values, dtype=complex)
        vals = deform_fn(arr)
        vals_arr = np.asarray(vals, dtype=complex)
        if vals_arr.shape != arr.shape:
            raise DSLValidationError(
                "Deformation evaluated to shape "
                f"{vals_arr.shape}, expected {arr.shape}."
            )
        return vals_arr

    return canonical, deformation


def _normalize_symbol_name(raw: str) -> str:
    r"""
    Normalize SymPy symbol names coming from LaTeX parsing.

    Examples:
        "a_{1}"       -> "a_1"
        "sx_{2}"      -> "sx_2"
        "Jx_{1}"      -> "Jx_1"
        "omega_{c}"   -> "omega_c"

    This is purely a syntactic normalization; it does not decide whether
    the symbol is an operator or a scalar.
    """
    return raw.replace("{", "").replace("}", "")


def _lookup_subsystem(
    kind: Literal["qubit", "boson", "custom"],
    label: str,
    index: int,
    config: HilbertConfig,
) -> Union[QubitSpec, BosonSpec, CustomSpec]:
    r"""
    Find the subsystem spec by (kind, label, index).

    For flexibility, if no exact label match is found, a subsystem with the
    same kind and index is returned (useful when operator notation uses
    canonical labels but config uses custom labels).

    Raises
    ------
    DSLValidationError
        If no matching subsystem is found.
    """
    if kind == "qubit":
        for q in config.qubits:
            if q.label == label and q.index == index:
                return q
        for q in config.qubits:
            if q.index == index:
                return q
    elif kind == "boson":
        for b in config.bosons:
            if b.label == label and b.index == index:
                return b
        for b in config.bosons:
            if b.index == index:
                return b
    elif kind == "custom":
        for c in config.customs:
            if c.label == label and c.index == index:
                return c
        for c in config.customs:
            if c.index == index:
                return c

    raise DSLValidationError(
        f"No {kind!r} subsystem with label={label!r}, index={index} in HilbertConfig."
    )


def parse_operator_symbol(sym: sp.Symbol, config: HilbertConfig) -> LocalOperatorRef:
    r"""
    Interpret a SymPy ``Symbol`` as a DSL operator.

    Parameters
    ----------
    sym : sympy.Symbol
        Symbol in the form ``base_index`` (e.g., ``a_1``, ``adag_2``, ``sx_1``).
    config : HilbertConfig
        Declares available subsystems and custom operators.

    Returns
    -------
    LocalOperatorRef
        Operator reference with ``power=1``.

    Raises
    ------
    DSLValidationError
        If the symbol matches an operator pattern but no matching subsystem
        or custom operator exists in ``config``.
    """
    name = _normalize_symbol_name(sym.name)

    if "_" not in name:
        raise DSLValidationError(
            "Symbol "
            f"{name!r} is not a recognized operator (expected pattern base_index)."
        )

    parts = name.split("_")
    if len(parts) == 2:
        base, idx_str = parts
        suffix = None
    elif len(parts) == 3:
        base, suffix, idx_str = parts
    else:
        raise DSLValidationError(
            f"Symbol {name!r} has too many '_' parts; "
            f"expected patterns like 'a_1', 'adag_1', 'sx_2', 'A_3'."
        )

    try:
        idx = int(idx_str)
    except ValueError as exc:
        raise DSLValidationError(
            f"Symbol {name!r}: expected an integer subsystem index at the end."
        ) from exc

    # Bosonic operators (label 'a')
    if base == "a":
        kind: Literal["boson"] = "boson"
        spec = _lookup_subsystem(kind, label="a", index=idx, config=config)
        op_name = "a"
    elif base == "adag":
        kind = "boson"
        spec = _lookup_subsystem(kind, label="a", index=idx, config=config)
        op_name = "adag"
    elif base == "af":
        kind = "boson"
        spec = _lookup_subsystem(kind, label="a", index=idx, config=config)
        op_name = "af"
    elif base == "adagf":
        kind = "boson"
        spec = _lookup_subsystem(kind, label="a", index=idx, config=config)
        op_name = "adagf"
    elif base == "n":
        kind = "boson"
        spec = _lookup_subsystem(kind, label="a", index=idx, config=config)
        op_name = "n"

    # Qubit operators (label 'q')
    elif base in ("sx", "sy", "sz", "sp", "sm"):
        kind = "qubit"
        spec = _lookup_subsystem(kind, label="q", index=idx, config=config)
        op_name = base

    # Custom subsystem (label 'c')
    else:
        kind = "custom"
        spec = _lookup_subsystem(kind, label="c", index=idx, config=config)
        op_name = base if suffix is None else f"{base}_{suffix}"
        if op_name not in spec.operators:
            raise DSLValidationError(
                f"Custom operator {op_name!r} not defined for custom subsystem "
                f"label={spec.label!r}, index={spec.index}."
            )

    return LocalOperatorRef(
        kind=kind, label=spec.label, index=spec.index, op_name=op_name
    )


def try_parse_operator_symbol(
    sym: sp.Symbol, config: HilbertConfig
) -> Optional[LocalOperatorRef]:
    r"""
    Soft version of :func:`parse_operator_symbol`.

    Parameters
    ----------
    sym : sympy.Symbol
        Symbol to interpret.
    config : HilbertConfig
        Declares available subsystems/operators.

    Returns
    -------
    LocalOperatorRef or None
        Operator reference if recognized; ``None`` if treated as scalar or
        if resolution fails.
    """
    name = _normalize_symbol_name(sym.name)

    # Symbols without '_' are never operators in this DSL.
    if "_" not in name:
        return None

    parts = name.split("_")
    if len(parts) == 2:
        base, idx_str = parts
        suffix = None
    elif len(parts) == 3:
        base, suffix, idx_str = parts
    else:
        # Name is too weird to be an operator in this DSL; treat as scalar.
        return None

    # NEW: if the "index" is not purely digits, treat as scalar immediately.
    # This prevents things like "n_th" or "omega_c*1" from ever going through
    # parse_operator_symbol and triggering integer-index expectations.
    if not idx_str.isdigit():
        return None

    boson_bases = {"a", "adag", "af", "adagf", "n"}
    qubit_bases = {"sx", "sy", "sz", "sp", "sm"}

    # Custom operator names from config
    custom_op_names: set[str] = set()
    for c in config.customs:
        custom_op_names.update(c.operators.keys())

    name_wo_index = base if suffix is None else f"{base}_{suffix}"

    looks_like_operator = (
        base in boson_bases or base in qubit_bases or name_wo_index in custom_op_names
    )

    if not looks_like_operator:
        # Treat as scalar parameter (e.g. J_x, omega_c, h_1) and return None.
        return None

    # At this point it *looks* like an operator. Try to parse; if that fails,
    # we still treat it as scalar in this soft path.
    try:
        return parse_operator_symbol(sym, config)
    except DSLValidationError:
        return None


def extract_operator_refs_from_latex(
    expr_latex: str, config: HilbertConfig
) -> List[LocalOperatorRef]:
    r"""
    Parse LaTeX and return operator references only.

    Parameters
    ----------
    expr_latex : str
        Physics-style LaTeX expression (operators + scalars).
    config : HilbertConfig
        Declares available subsystems/operators.

    Returns
    -------
    list[LocalOperatorRef]
        Operator references found in the expression. Scalar symbols are ignored.

    Examples
    --------
    >>> cfg = HilbertConfig(qubits=[QubitSpec('q',1)], bosons=[], customs=[])
    >>> refs = extract_operator_refs_from_latex(r\"\\sigma_{x,1} + \\omega t\", cfg)
    >>> {(r.kind, r.op_name, r.index) for r in refs}
    {('qubit', 'sx', 1)}
    """
    canonical = canonicalize_physics_latex(expr_latex)
    expr = parse_latex(canonical)

    refs: List[LocalOperatorRef] = []
    for s in expr.free_symbols:
        ref = try_parse_operator_symbol(s, config)
        if ref is not None:
            refs.append(ref)

    return refs


if __name__ == "__main__":  # pragma: no cover
    print("Run pytest to execute the test suite.")
