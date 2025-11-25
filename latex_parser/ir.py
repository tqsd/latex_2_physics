# ir.py
#
# Intermediate representation (IR) layer for Hamiltonians and collapse
# operators built from physics-style LaTeX.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Set, Tuple, Union

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.latex import parse_latex

from latex_parser.dsl import (  # type: ignore
    DSLValidationError,
    HilbertConfig,
    LocalOperatorRef,
    canonicalize_physics_latex,
    try_parse_operator_symbol,
)
from latex_parser.dsl_constants import (
    ALLOWED_OPERATOR_FUNCTIONS,
    ERROR_HINT_OPERATOR_FUNC,
)

logger = logging.getLogger(__name__)

# Upper bound on number of additive terms after expansion.
MAX_EXPANDED_TERMS = 512


@dataclass
class OperatorFunctionRef:
    r"""
    Reference to an operator-valued function f(O) applied to a single local
    operator (possibly with an integer power).
    """

    func_name: str
    arg: LocalOperatorRef
    scalar_factor: sp.Expr = sp.Integer(1)


# Factors inside Term.ops can now be a plain operator or an operator function.
OperatorFactor = Union[LocalOperatorRef, OperatorFunctionRef]


@dataclass
class Term:
    r"""
    One monomial term in a Hamiltonian or collapse operator.

    Each term has the form

    .. math::

        \text{scalar\_expr} \times \hat{O}_1 \hat{O}_2 \dots \hat{O}_m.

    Attributes
    ----------
    scalar_expr : sympy.Expr
        Scalar SymPy expression containing only scalar symbols and
        scalar-valued functions (parameters, time symbols, numbers,
        trigonometric functions, ``\exp``, and so on). The expression
        must not contain DSL-recognized operator symbols such as
        ``a_{1}``, ``\sigma_{x,1}``, or ``J_{z,1}``.
    ops : list of OperatorFactor
        Ordered list describing the operator product
        :math:`\hat{O}_1 \hat{O}_2 \dots \hat{O}_m`. Each
        :class:`dsl.LocalOperatorRef` or :class:`OperatorFunctionRef`
        may carry a power greater than one if the original term contained powers such as
        ``\hat{n}_{1}^{2}`` or ``a_{1}^{3}``.

    Examples
    --------
    A term corresponding to

    .. math::

        g \, a_{1}^{\dagger} a_{1}

    would be represented as::

        Term(
            scalar_expr = Symbol("g"),
            ops = [
                LocalOperatorRef(kind="boson", label="a", index=1,
                                 op_name="adag", power=1),
                LocalOperatorRef(kind="boson", label="a", index=1,
                                 op_name="a", power=1),
            ],
        )
    """

    scalar_expr: sp.Expr
    ops: List[OperatorFactor]


@dataclass
class HamiltonianIR:
    r"""
    Symbolic intermediate representation of a Hamiltonian :math:`H(t)`.

    The IR stores :math:`H(t)` as a sum of :class:`Term` instances:

    .. math::

        H(t) = \sum_k \text{terms}[k].\text{scalar\_expr} \,
                \prod_j \hat{O}_{k,j}.

    Attributes
    ----------
    terms : list of Term
        List of monomial terms that describe the Hamiltonian or collapse
        operator in factorized form.
    has_time_dep : bool
        Flag indicating whether any scalar coefficient depends on the
        time symbol or on any of the additional time-like symbols.

    Examples
    --------
    For a driven qubit Hamiltonian,

    .. math::

        H(t) = \frac{\omega_0}{2} \sigma_{z,1}
                + A \cos(\omega t) \sigma_{x,1},

    the IR contains two terms: one static and one time dependent.
    """

    terms: List[Term]
    has_time_dep: bool


def parse_latex_expr(latex_str: str) -> sp.Expr:
    r"""
    Canonicalize physics-style LaTeX and parse it into a SymPy expression.

    This function first applies :func:`dsl.canonicalize_physics_latex`
    to normalize physics-style LaTeX, and then calls
    :func:`sympy.parsing.latex.parse_latex` to obtain a SymPy
    expression.

    The canonicalization step rewrites common operators, for example,

    * ``\sigma_{x,1} \rightarrow \sx_{1}``
    * ``J_{z,1} \rightarrow \Jz_{1}``
    * ``a_{1}^{\dagger} \rightarrow \adag_{1}``
    * ``\hat{n}_{1} \rightarrow n_{1}``

    and wraps bare multi-letter scalar identifiers (such as ``nbar``,
    ``kappa``) so that SymPy treats them as single scalar symbols.

    Parameters
    ----------
    latex_str : str
        Physics-style LaTeX string representing a scalar or
        operator-valued expression.

    Returns
    -------
    sympy.Expr
        SymPy expression corresponding to the canonicalized LaTeX.

    Examples
    --------
    >>> expr = parse_latex_expr(r"\\omega_c \\hat{n}_{1}")
    >>> sorted(str(s) for s in expr.free_symbols)
    ['n_{1}', 'omega_c']
    """
    logger.debug("parse_latex_expr: canonicalizing LaTeX: %s", latex_str)
    canonical = canonicalize_physics_latex(latex_str)
    logger.debug("parse_latex_expr: canonical form: %s", canonical)
    expr = parse_latex(canonical)
    return expr


def _rescue_implicit_scalar_funcs(expr: sp.Expr, config: HilbertConfig) -> sp.Expr:
    r"""
    Rewrite implicit scalar functions applied to operator expressions.

    SymPy's LaTeX parser can interpret expressions such as

    .. math::

        g \left( a_{1}^{\dagger} + a_{1} \right)

    as an undefined function :math:`g(\cdot)` instead of scalar
    multiplication. This helper rewrites undefined functions that have
    operator arguments into scalar multiplication:

    .. math::

        g\big(a_{1}^{\dagger} + a_{1}\big)
        \;\longrightarrow\;
        g \, \big(a_{1}^{\dagger} + a_{1}\big).

    Rules
    -----
    * Only undefined functions (:class:`sympy.core.function.AppliedUndef`)
      are considered.
    * A function is rewritten only if at least one of its arguments
      contains a DSL-recognized operator symbol under the supplied
      :class:`dsl.HilbertConfig`.
    * Known scalar functions such as ``\sin``, ``\cos``, or ``\exp``
      are not affected, because they do not appear as
      :class:`AppliedUndef` nodes.

    Parameters
    ----------
    expr : sympy.Expr
        SymPy expression to transform.
    config : HilbertConfig
        Hilbert-space configuration used to recognize operator symbols.

    Returns
    -------
    sympy.Expr
        Expression in which undefined functions with operator arguments
        are converted into scalar symbols multiplied by their arguments,
        while purely scalar undefined functions are left intact.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[],
    ...     bosons=[BosonSpec(label="a", index=1, cutoff=5)],
    ...     customs=[],
    ... )
    >>> a1 = sp.Symbol("a_{1}")
    >>> adag1 = sp.Symbol("adag_{1}")
    >>> g = sp.Function("g")
    >>> expr = g(a1 + adag1)
    >>> fixed = _rescue_implicit_scalar_funcs(expr, cfg)
    >>> isinstance(fixed, sp.Mul)
    True
    """

    def _rec(node: sp.Expr) -> sp.Expr:
        r"""Recursively rescue implicit scalar functions applied to operators."""
        if isinstance(node, AppliedUndef):
            func = node.func
            func_name = func.name

            has_op_arg = False
            for arg in node.args:
                for s in arg.free_symbols:
                    if try_parse_operator_symbol(s, config) is not None:
                        has_op_arg = True
                        break
                if has_op_arg:
                    break

            new_args = [_rec(a) for a in node.args]

            if has_op_arg:
                func_sym = sp.Symbol(func_name)
                if not new_args:
                    logger.debug(
                        "Implicit scalar func rescue: %s() with operator "
                        "argument but no explicit args; treating as scalar symbol.",
                        func_name,
                    )
                    return func_sym
                factors = [func_sym] + new_args
                return sp.Mul(*factors)

            return node.func(*new_args)

        if node.args:
            new_args = [_rec(a) for a in node.args]
            return node.func(*new_args)
        return node

    return _rec(expr)


def _expand_noncommutative_powers(expr: sp.Expr) -> sp.Expr:
    r"""
    Expand powers of sums when operands are non-commutative.

    SymPy may leave ``(A + B)**n`` unexpanded when ``A``/``B`` are
    non-commutative. This helper explicitly expands such powers for
    integer ``n >= 2`` to surface cross terms in the IR.
    """

    def _expand_pow(node: sp.Expr) -> sp.Expr:
        r"""Expand ``(A+B)**n`` for integer ``n`` when operands are non-commutative."""
        if (
            isinstance(node, sp.Pow)
            and isinstance(node.exp, sp.Integer)
            and node.exp >= 2
            and isinstance(node.base, sp.Add)
        ):
            return sp.expand(node, mul=True)
        return node

    return expr.replace(lambda x: isinstance(x, sp.Pow), _expand_pow)


def _make_operators_noncommutative(
    expr: sp.Expr, config: HilbertConfig
) -> Tuple[sp.Expr, Set[sp.Symbol]]:
    r"""
    Mark operator symbols as non-commutative in a SymPy expression.

    This helper scans the free symbols in ``expr`` and identifies those
    that correspond to operators under the supplied
    :class:`dsl.HilbertConfig`. Each such symbol is replaced by a
    non-commutative counterpart created as

    .. code-block:: python

        sp.Symbol(name, commutative=False)

    The returned expression preserves operator order from this point
    onward and enables detection of operator-valued scalar functions.

    Parameters
    ----------
    expr : sympy.Expr
        SymPy expression containing scalar and operator symbols.
    config : HilbertConfig
        Hilbert-space configuration used to recognize operator symbols.

    Returns
    -------
    expr_nc : sympy.Expr
        Copy of ``expr`` where each operator symbol has been replaced by
        a non-commutative symbol.
    op_syms_nc : set of sympy.Symbol
        Set of non-commutative symbols that were introduced.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> expr = sp.Symbol("sx_{1}") * sp.Symbol("sx_{1}")
    >>> expr_nc, op_syms_nc = _make_operators_noncommutative(expr, cfg)
    >>> all(not s.is_commutative for s in op_syms_nc)
    True
    """
    op_syms: Set[sp.Symbol] = set()

    for s in expr.free_symbols:
        ref = try_parse_operator_symbol(s, config)
        if ref is not None:
            op_syms.add(s)

    if not op_syms:
        return expr, set()

    repl = {s: sp.Symbol(s.name, commutative=False) for s in op_syms}
    expr_nc = expr.xreplace(repl)
    op_syms_nc = set(repl.values())

    logger.debug(
        "_make_operators_noncommutative: converted %d operator symbols.",
        len(op_syms_nc),
    )

    return expr_nc, op_syms_nc


def _rescue_merged_time_scalars(
    expr: sp.Expr, t_name: str = "t", extra_time_names: tuple[str, ...] = ("phi",)
) -> sp.Expr:
    r"""
    Rescue merged time scalars such as ``\omega_{dt}`` to ``\omega_{d} t``.

    SymPy's LaTeX parser sometimes converts products like
    ``\cos(\omega_d t)`` into a single symbol ``omega_{dt}`` instead of
    a product ``omega_{d} * t``. This helper detects such merged
    symbols and rewrites them as products of a base parameter and a
    time-like symbol.

    If a symbol named ``t_name`` or any name in ``extra_time_names``
    already appears explicitly in ``expr.free_symbols``, then no
    modification is performed.

    Parameters
    ----------
    expr : sympy.Expr
        SymPy expression to scan for merged time-like symbols.
    t_name : str, optional
        Name of the primary time symbol, by default ``"t"``.
    extra_time_names : tuple of str, optional
        Additional names to treat as time-like symbols (for example
        ``("phi",)``). These are used only for detecting whether an
        explicit time-like symbol already appears and for splitting
        merged subscripts.

    Returns
    -------
    sympy.Expr
        Expression where merged symbols such as ``omega_{dt}`` have
        been rewritten as ``omega_{d} * t`` whenever no explicit time
        symbol was already present.

    Examples
    --------
    >>> expr = sp.Symbol("omega_{dt}") * sp.Symbol("sx_{1}")
    >>> rescued = _rescue_merged_time_scalars(expr, t_name="t")
    >>> sorted(str(s) for s in rescued.free_symbols)
    ['omega_{d}', 'sx_{1}', 't']
    """
    time_names = {t_name} | set(extra_time_names)
    if any(s.name in time_names for s in expr.free_symbols):
        return expr

    t_sym = sp.Symbol(t_name)
    replacements: dict[sp.Symbol, sp.Expr] = {}

    for s in expr.free_symbols:
        name = s.name
        if "_" not in name:
            continue
        base, subpart = name.split("_", 1)
        if not (subpart.startswith("{") and subpart.endswith("}")):
            continue
        sub = subpart[1:-1]
        if "*" in sub or not sub.isalnum():
            continue
        match_time = [
            tn for tn in time_names if sub.endswith(tn) and len(sub) > len(tn)
        ]
        if not match_time:
            continue

        tn = match_time[0]
        new_sub = sub[: -len(tn)]
        if not new_sub:
            continue

        new_name = f"{base}_{{{new_sub}}}"
        new_sym = sp.Symbol(new_name)

        replacements[s] = new_sym * t_sym

    if not replacements:
        return expr

    logger.debug(
        "_rescue_merged_time_scalars: rewrote %d merged time symbols.",
        len(replacements),
    )

    return expr.xreplace(replacements)


def _is_operator_function_allowed(
    func: sp.Function, op_syms_nc: Set[sp.Symbol]
) -> bool:
    r"""
    Return True if ``func`` is an operator-valued function we accept.

    Rules:
    * Function name must be in ALLOWED_OPERATOR_FUNCTIONS.
    * Must have exactly one argument.
    * Argument must reduce to exactly one operator symbol (optionally
      raised to a non-negative integer power) times a scalar factor
      containing no operators. Sums of operators are rejected.
    """
    func_name = func.func.__name__
    if func_name not in ALLOWED_OPERATOR_FUNCTIONS:
        return False
    if len(func.args) != 1:
        return False

    arg = func.args[0]
    if isinstance(arg, sp.Add):
        return False

    def _is_op_pow(node: sp.Expr) -> bool:
        r"""Return True if node is an operator symbol or its non-negative power."""
        if isinstance(node, sp.Symbol):
            return node in op_syms_nc
        if isinstance(node, sp.Pow) and isinstance(node.base, sp.Symbol):
            return (
                node.base in op_syms_nc
                and isinstance(node.exp, sp.Integer)
                and node.exp >= 0
            )
        return False

    if _is_op_pow(arg):
        return True

    if isinstance(arg, sp.Mul):
        op_terms = [fac for fac in arg.args if _is_op_pow(fac)]
        if len(op_terms) != 1:
            return False
        for fac in arg.args:
            if fac is op_terms[0]:
                continue
            if any(s in op_syms_nc for s in fac.free_symbols):
                return False
        return True

    return False


def _parse_operator_function_ref(
    func: sp.Function, config: HilbertConfig
) -> OperatorFunctionRef:
    r"""
    Convert an allowed operator function f(O) into OperatorFunctionRef.

    Assumes validity was pre-checked by _is_operator_function_allowed.
    """
    func_name = func.func.__name__
    arg = func.args[0]
    scalar_factor: sp.Expr = sp.Integer(1)
    op_expr: sp.Expr

    if isinstance(arg, sp.Add):
        raise DSLValidationError(
            "Operator function argument cannot be a sum; only a single operator "
            "optionally multiplied by a scalar is allowed."
        )

    if isinstance(arg, sp.Mul):
        op_part = None
        scalar_parts: list[sp.Expr] = []
        for fac in arg.args:
            if isinstance(fac, sp.Symbol) and try_parse_operator_symbol(fac, config):
                if op_part is not None:
                    raise DSLValidationError(
                        "Operator function argument must contain exactly one operator."
                    )
                op_part = fac
                continue
            if isinstance(fac, sp.Pow) and isinstance(fac.base, sp.Symbol):
                if try_parse_operator_symbol(fac.base, config):
                    if op_part is not None:
                        raise DSLValidationError(
                            "Operator function argument must contain exactly one "
                            "operator."
                        )
                    op_part = fac
                    continue
            scalar_parts.append(fac)

        if op_part is None:
            raise DSLValidationError(
                "Operator function argument must contain exactly one operator."
            )
        op_expr = op_part
        scalar_factor = sp.Mul(*scalar_parts) if scalar_parts else sp.Integer(1)

    else:
        op_expr = arg

    if isinstance(op_expr, sp.Pow):
        base = op_expr.base
        power = int(op_expr.exp)
        if not isinstance(base, sp.Symbol):
            raise DSLValidationError(
                "Operator function argument must be a single operator symbol "
                "optionally raised to a positive integer power."
            )
        ref = try_parse_operator_symbol(base, config)
        if ref is None:
            raise DSLValidationError(
                f"Unrecognized operator symbol '{base}' inside {func_name}()."
            )
        ref = LocalOperatorRef(
            kind=ref.kind,
            label=ref.label,
            index=ref.index,
            op_name=ref.op_name,
            power=power,
        )
    elif isinstance(op_expr, sp.Symbol):
        ref = try_parse_operator_symbol(op_expr, config)
        if ref is None:
            raise DSLValidationError(
                f"Unrecognized operator symbol '{op_expr}' inside {func_name}()."
            )
    else:
        raise DSLValidationError(
            "Operator function argument must be a single operator symbol "
            "optionally raised to a positive integer power."
        )

    return OperatorFunctionRef(
        func_name=func_name, arg=ref, scalar_factor=scalar_factor
    )


def expr_to_ir(
    expr: sp.Expr,
    config: HilbertConfig,
    t_name: str = "t",
    time_symbols: tuple[str, ...] | None = None,
) -> HamiltonianIR:
    r"""
    Convert a SymPy expression into :class:`HamiltonianIR`.

    The conversion pipeline is:

    0. Rescue merged time scalars such as ``omega_{dt}`` by rewriting
       them as ``omega_{d} * t`` (and respect any extra
       ``time_symbols``).
    1. Rewrite implicit scalar functions with operator arguments, for
       example

       .. math::

           g\big(a_{1}^{\dagger} + a_{1}\big)
           \;\longrightarrow\;
           g \, \big(a_{1}^{\dagger} + a_{1}\big).

    2. Mark operator symbols as non-commutative, preserving operator
       order from this step onward.
    3. Expand powers of sums for non-commutative operands so that
       cross terms appear explicitly.
    4. Reject operator-valued scalar functions such as
       ``exp(\sigma_{z,1})`` or ``\sin(\sigma_{x,1})``. Only
       scalar-valued functions of scalar arguments are allowed in
       coefficient expressions.
    5. Expand the expression with :func:`sympy.expand` to split it into
       an additive sum of monomials. A guard aborts if the number of
       additive terms exceeds :data:`MAX_EXPANDED_TERMS`.
    6. For each monomial term:

       * Split the product into factors.
       * Classify each factor as a scalar factor or an operator factor.
       * For operator powers such as ``a_{1}^{3}`` or ``n_{1}^{2}``,
         only non-negative integer powers are accepted; negative or
         non-integer powers raise a :class:`DSLValidationError`.
       * Collect scalar factors into a single ``scalar_expr``.
       * Compress consecutive identical operators into a single
         :class:`LocalOperatorRef` with higher ``power``.

    7. Mark the IR as time-dependent if any ``scalar_expr`` contains a
       free symbol whose name is ``t_name`` or one of the names in
       ``time_symbols``.

    Parameters
    ----------
    expr : sympy.Expr
        SymPy expression built from canonicalized LaTeX that may
        contain both scalar and operator symbols.
    config : HilbertConfig
        Hilbert-space configuration describing all qubits, bosons, and
        custom subsystems. This is used to recognize operator symbols
        and to distinguish them from scalar parameters.
    t_name : str, optional
        Name of the primary time symbol, by default ``"t"``.
    time_symbols : tuple of str, optional
        Additional names to treat as time-like symbols when detecting
        time dependence. If ``None``, a default of ``("phi",)`` is
        used only in the merged-time rescue step.

    Returns
    -------
    HamiltonianIR
        Intermediate representation of the expression, in which each
        additive term has been factorized into a scalar coefficient and
        an ordered product of local operators.

    Raises
    ------
    DSLValidationError
        If an operator-valued scalar function is detected (for example
        ``\exp(\sigma_{z,1})``), if an operator appears with a negative
        or non-integer power, or if expansion would produce more than
        :data:`MAX_EXPANDED_TERMS` additive monomials.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> expr = parse_latex_expr(
    ...     r"\\frac{\\omega_0}{2} \\sigma_{z,1} + A \\cos(\\omega t) \\sigma_{x,1}"
    ... )
    >>> ir = expr_to_ir(expr, cfg, t_name="t")
    >>> len(ir.terms) >= 2
    True
    """
    expr = _rescue_merged_time_scalars(
        expr, t_name=t_name, extra_time_names=time_symbols or ("phi",)
    )

    expr_fixed = _rescue_implicit_scalar_funcs(expr, config)

    expr_nc, op_syms_nc = _make_operators_noncommutative(expr_fixed, config)
    expr_nc = _expand_noncommutative_powers(expr_nc)
    expr_nc = sp.expand(expr_nc, mul=True, power_exp=True, power_base=True)

    for func in expr_nc.atoms(sp.Function):
        if any(s in op_syms_nc for s in func.free_symbols):
            if not _is_operator_function_allowed(func, op_syms_nc):
                raise DSLValidationError(ERROR_HINT_OPERATOR_FUNC)

    if isinstance(expr_nc, sp.Add) and len(expr_nc.args) > MAX_EXPANDED_TERMS:
        raise DSLValidationError(
            f"Expression expands to {len(expr_nc.args)} additive terms, "
            f"which exceeds the configured limit of {MAX_EXPANDED_TERMS}. "
            "Rewrite your Hamiltonian explicitly instead of using large "
            "powers of operator sums such as (a_1 + a_2)^k with large k."
        )

    if isinstance(expr_nc, sp.Add):
        term_exprs = list(expr_nc.args)
    else:
        term_exprs = [expr_nc]

    terms: List[Term] = []

    for term_expr in term_exprs:
        if isinstance(term_expr, sp.Mul):
            factors = list(term_expr.args)
        else:
            factors = [term_expr]

        scalar_factors: List[sp.Expr] = []
        op_refs_flat: List[OperatorFactor] = []

        for fac in factors:
            if isinstance(fac, sp.Pow):
                base, exp = fac.base, fac.exp

                if isinstance(base, sp.Symbol):
                    ref = try_parse_operator_symbol(base, config)
                    if ref is not None:
                        if not isinstance(exp, sp.Integer):
                            raise DSLValidationError(
                                f"Operator symbol '{base}' has non-integer "
                                f"power '{exp}'. Non-integer powers of "
                                "operators are not supported by this DSL."
                            )
                        if exp < 0:
                            raise DSLValidationError(
                                f"Operator symbol '{base}' has negative power "
                                f"'{exp}'. Negative powers of operators are "
                                "not supported by this DSL."
                            )

                        n = int(exp)
                        for _ in range(n):
                            op_refs_flat.append(ref)
                        continue

                scalar_factors.append(fac)
                continue

            if isinstance(fac, sp.Symbol):
                ref = try_parse_operator_symbol(fac, config)
                if ref is None:
                    scalar_factors.append(fac)
                else:
                    op_refs_flat.append(ref)
                continue

            if isinstance(fac, sp.Function):
                op_syms_here = {
                    s for s in fac.free_symbols if try_parse_operator_symbol(s, config)
                }
                if op_syms_here:
                    if not _is_operator_function_allowed(fac, op_syms_nc):
                        raise DSLValidationError(ERROR_HINT_OPERATOR_FUNC)
                    op_refs_flat.append(_parse_operator_function_ref(fac, config))
                    continue

            scalar_factors.append(fac)

        if scalar_factors:
            scalar_expr = sp.Mul(*scalar_factors)
        else:
            scalar_expr = sp.Integer(1)

        ops: List[OperatorFactor] = []
        for ref in op_refs_flat:
            if isinstance(ref, OperatorFunctionRef):
                ops.append(ref)
                continue

            if ops and isinstance(ops[-1], LocalOperatorRef):
                last = ops[-1]
                if (
                    last.kind == ref.kind
                    and last.label == ref.label
                    and last.index == ref.index
                    and last.op_name == ref.op_name
                ):
                    ops[-1] = LocalOperatorRef(
                        kind=last.kind,
                        label=last.label,
                        index=last.index,
                        op_name=last.op_name,
                        power=last.power + 1,
                    )
                    continue
            ops.append(
                LocalOperatorRef(
                    kind=ref.kind,
                    label=ref.label,
                    index=ref.index,
                    op_name=ref.op_name,
                    power=ref.power,
                )
            )

        terms.append(Term(scalar_expr=scalar_expr, ops=ops))

    time_names = set(time_symbols) if time_symbols else set()
    time_names.add(t_name)

    def _term_has_time(term: Term) -> bool:
        r"""Detect time dependence in a single IR term."""
        if any(s.name in time_names for s in term.scalar_expr.free_symbols):
            return True
        for op in term.ops:
            if isinstance(op, OperatorFunctionRef):
                if any(s.name in time_names for s in op.scalar_factor.free_symbols):
                    return True
        return False

    has_time_dep = any(_term_has_time(term) for term in terms)

    logger.debug(
        "expr_to_ir: produced HamiltonianIR with %d terms " "(time-dependent=%s).",
        len(terms),
        has_time_dep,
    )

    return HamiltonianIR(terms=terms, has_time_dep=has_time_dep)


def latex_to_ir(
    latex_str: str,
    config: HilbertConfig,
    t_name: str = "t",
    time_symbols: tuple[str, ...] | None = None,
) -> HamiltonianIR:
    r"""
    Parse a LaTeX Hamiltonian and convert it into :class:`HamiltonianIR`.

    This is a convenience wrapper around :func:`parse_latex_expr` and
    :func:`expr_to_ir`. It accepts physics-style LaTeX, canonicalizes
    it using :func:`dsl.canonicalize_physics_latex`, parses it into a
    SymPy expression, and finally builds the IR representation.

    Parameters
    ----------
    latex_str : str
        Physics-style LaTeX string describing a Hamiltonian or a single
        collapse operator.
    config : HilbertConfig
        Hilbert-space configuration describing all subsystems.
    t_name : str, optional
        Name of the primary time symbol, by default ``"t"``.
    time_symbols : tuple of str, optional
        Additional names to treat as time-like symbols when detecting
        time dependence.

    Returns
    -------
    HamiltonianIR
        Intermediate representation of the parsed expression.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> H_latex = r"\\frac{\\omega_0}{2} \\sigma_{z,1}"
    >>> ir = latex_to_ir(H_latex, cfg, t_name="t")
    >>> len(ir.terms)
    1
    """
    expr = parse_latex_expr(latex_str)
    return expr_to_ir(expr, config, t_name=t_name, time_symbols=time_symbols)


def map_ir_terms(ir: HamiltonianIR, fn) -> HamiltonianIR:
    r"""
    Apply a transformation function to each Term and return a new IR.
    """
    new_terms = [fn(term) for term in ir.terms]
    return HamiltonianIR(terms=new_terms, has_time_dep=ir.has_time_dep)


def _term_ops_signature(terms: List[Term]) -> set[tuple[str, str, int, int]]:
    r"""
    Build a compact signature describing the operators in a list of terms.

    Parameters
    ----------
    terms : list of Term
        Terms whose operator content should be summarized.

    Returns
    -------
    set of tuple
        Set of quadruples ``(kind, op_name, index, power)`` summarizing
        all operator references across all terms.
    """
    sig: set[tuple[str, str, int, int]] = set()
    for term in terms:
        for r in term.ops:
            if isinstance(r, OperatorFunctionRef):
                sig.add(
                    (f"func:{r.func_name}", r.arg.op_name, r.arg.index, r.arg.power)
                )
            else:
                sig.add((r.kind, r.op_name, r.index, r.power))
    return sig


if __name__ == "__main__":  # pragma: no cover
    print("Run pytest to execute the test suite.")
