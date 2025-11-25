# backend_qutip.py
#
# QuTiP backend for the LaTeX → DSL → IR pipeline.
#
# Responsibilities:
#   - Take HamiltonianIR (from ir.py) plus a HilbertConfig (from dsl.py).
#   - Build QuTiP Qobj operators for each IR term using tensor products.
#   - Evaluate scalar SymPy coefficients with a user-supplied parameter dict.
#   - Return:
#       * a single static Hamiltonian Qobj for time-independent problems, OR
#       * a QuTiP time-dependent Hamiltonian list [H0, [H1, f1], ...]
#         for time-dependent problems.
#
# Limitations / assumptions:
#   - Parameters dict keys may be given in user-friendly form like "omega_c";
#     internally SymPy may call the symbol "omega_{c}". We resolve both via
#     aliasing rules (_param_aliases).
#   - Time-dependence detection uses t_name plus any provided time_symbols;
#     envelopes are lambdified as f(t, args) with extra time-like symbols treated
#     as parameters.
#   - Collapse operators and Hamiltonians both support time-dependent scalar
#     envelopes f(t, args) multiplying fixed operator monomials.

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import sympy as sp
from qutip import (  # type: ignore
    Qobj,
    destroy,
    qeye,
    sigmam,
    sigmap,
    sigmax,
    sigmay,
    sigmaz,
    tensor,
)

from latex_parser.backend_base import (
    BackendBase,
    BackendOptions,
    CompiledHamiltonianBase,
    CompiledOpenSystemBase,
)
from latex_parser.backend_cache import BaseOperatorCache, SubsystemInfo
from latex_parser.backend_utils import (
    expr_has_time as _expr_has_time_shared,
)
from latex_parser.backend_utils import (
    param_aliases,
    resolve_param,
)
from latex_parser.dsl import (  # type: ignore
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    LocalOperatorRef,
)
from latex_parser.dsl_constants import (
    ALLOWED_OPERATOR_FUNCTIONS,
    ERROR_HINT_TIME_DEP_COLLAPSE,
)
from latex_parser.ir import (
    HamiltonianIR,
    OperatorFactor,
    OperatorFunctionRef,
    Term,
    latex_to_ir,
)  # type: ignore
from latex_parser.operator_functions import apply_operator_function

logger = logging.getLogger(__name__)


class QutipOperatorCache(BaseOperatorCache[Qobj]):
    r"""
    Cache QuTiP operators for a fixed Hilbert configuration.

    This class fixes a tensor-product ordering of subsystems derived
    from a :class:`HilbertConfig` and caches both:

    * local operators (for example ``\\sigma_{x,1}``, ``a_{1}``,
      ``J_{z,1}``), and
    * embedded full-system operators obtained via tensor products.

    Parameters
    ----------
    config : HilbertConfig
        Configuration describing all qubits, bosons, and custom
        subsystems.

    Attributes
    ----------
    config : HilbertConfig
        Original configuration used to build the cache.
    subsystems : list of SubsystemInfo
        Ordered list of subsystem descriptors. The order is all qubits,
        then bosons, then custom subsystems.
    subsystem_index : dict
        Mapping from ``(kind, label, index)`` to the corresponding
        position in :attr:`subsystems`.
    identities : list of qutip.Qobj
        Local identity operators ``qeye(dim)`` for each subsystem.
    local_ops : dict
        Cache of local operators keyed by
        ``(kind, label, index, op_name)``.
    full_ops : dict
        Cache of tensor-embedded operators keyed by
        ``(kind, label, index, op_name, power)``.

    Examples
    --------
    Build a cache and retrieve local and full operators:

    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[BosonSpec(label="a", index=1, cutoff=3)],
    ...     customs=[],
    ... )
    >>> cache = QutipOperatorCache(cfg)
    >>> ref = LocalOperatorRef(kind="qubit", label="q", index=1,
    ...                        op_name="sx", power=1)
    >>> local = cache.local_operator(ref)
    >>> full = cache.full_operator(ref)
    >>> local.dims, full.dims
    ([[2], [2]], [[2, 3], [2, 3]])
    """

    def __init__(self, config: HilbertConfig):
        r"""Initialize the operator cache for a given Hilbert configuration."""
        self.local_ops: Dict[Tuple[str, str, int, str], Qobj] = {}
        self.full_ops: Dict[Tuple[str, str, int, str, int], Qobj] = {}
        super().__init__(config)

    # BaseOperatorCache hooks ------------------------------------------------
    def _local_identity(self, dim: int) -> Qobj:
        """Return a QuTiP identity operator of dimension ``dim``."""
        return qeye(dim)

    def _kron(self, a: Qobj, b: Qobj) -> Qobj:
        """Kronecker product for QuTiP operators."""
        return tensor(a, b)

    def _find_subsystem(self, kind: str, label: str, index: int) -> SubsystemInfo:
        r"""
        Locate a subsystem by kind, label and index.

        Parameters
        ----------
        kind : {"qubit", "boson", "custom"}
            Subsystem type to search for.
        label : str
            Subsystem label, for example ``"q"``.
        index : int
            Subsystem index, for example ``1``.

        Returns
        -------
        SubsystemInfo
            Subsystem descriptor matching the query.

        Raises
        ------
        DSLValidationError
            If the requested subsystem is not found in the current
            :class:`HilbertConfig`.

        Examples
        --------
        >>> cfg = HilbertConfig(
        ...     qubits=[QubitSpec(label="q", index=1)],
        ...     bosons=[],
        ...     customs=[],
        ... )
        >>> cache = QutipOperatorCache(cfg)
        >>> ss = cache._find_subsystem("qubit", "q", 1)
        >>> (ss.kind, ss.label, ss.index)
        ('qubit', 'q', 1)
        """
        key = (kind, label, index)
        if key not in self.subsystem_index:
            raise DSLValidationError(
                "No subsystem found for operator "
                f"kind={kind}, label={label}, index={index} in the current "
                "HilbertConfig."
            )
        pos = self.subsystem_index[key]
        return self.subsystems[pos]

    def _make_local_qubit_op(self, op_name: str, ss: SubsystemInfo) -> Qobj:
        r"""
        Build a local qubit operator.

        Parameters
        ----------
        op_name : {"sx", "sy", "sz", "sp", "sm"}
            Operator name in the DSL. Respectively maps to
            :math:`\\sigma_x`, :math:`\\sigma_y`, :math:`\\sigma_z`,
            :math:`\\sigma_+`, and :math:`\\sigma_-`.
        ss : SubsystemInfo
            Subsystem descriptor for the target qubit.

        Returns
        -------
        qutip.Qobj
            Local qubit operator acting on a two-dimensional space.

        Raises
        ------
        DSLValidationError
            If ``op_name`` is not a supported qubit operator.

        Examples
        --------
        >>> cfg = HilbertConfig(
        ...     qubits=[QubitSpec(label="q", index=1)],
        ...     bosons=[],
        ...     customs=[],
        ... )
        >>> cache = QutipOperatorCache(cfg)
        >>> ss = cache._find_subsystem("qubit", "q", 1)
        >>> op = cache._make_local_qubit_op("sx", ss)
        >>> op.shape
        (2, 2)
        """
        if op_name == "sx":
            return sigmax()
        if op_name == "sy":
            return sigmay()
        if op_name == "sz":
            return sigmaz()
        if op_name == "sp":
            return sigmap()
        if op_name == "sm":
            return sigmam()

        raise DSLValidationError(
            "Unsupported qubit operator "
            f"'{op_name}' for subsystem {ss.label}_{ss.index}."
        )

    def _make_local_boson_op(self, op_name: str, ss: SubsystemInfo) -> Qobj:
        r"""
        Build a local bosonic operator.

        Parameters
        ----------
        op_name : {"a", "adag", "af", "adagf", "n"}
            Operator name in the DSL. Respectively maps to the
            annihilation operator :math:`a`, the creation operator
            :math:`a^{\\dagger}`, and the number operator
            :math:`\\hat{n} = a^{\\dagger} a`. The ``af`` and ``adagf``
            variants apply an f-deformation supplied on the
            :class:`BosonSpec` as ``a f(n)`` and ``f(n) a^{\\dagger}``.
        ss : SubsystemInfo
            Subsystem descriptor for the target bosonic mode.

        Returns
        -------
        qutip.Qobj
            Local bosonic operator acting on a Fock space of
            dimension ``ss.dim``.

        Raises
        ------
        DSLValidationError
            If ``op_name`` is not a supported bosonic operator.

        Examples
        --------
        >>> cfg = HilbertConfig(
        ...     qubits=[],
        ...     bosons=[BosonSpec(label="a", index=1, cutoff=3)],
        ...     customs=[],
        ... )
        >>> cache = QutipOperatorCache(cfg)
        >>> ss = cache._find_subsystem("boson", "a", 1)
        >>> a = cache._make_local_boson_op("a", ss)
        >>> a.shape
        (3, 3)
        """
        dim = ss.dim
        # Safe guard for type checking in deformation handling.
        if not isinstance(ss.spec, BosonSpec):
            spec = ss.spec
            required_attrs = ["label", "index", "cutoff"]
            if not all(hasattr(spec, attr) for attr in required_attrs):
                raise DSLValidationError(
                    "Internal error: expected BosonSpec for boson subsystem "
                    f"{ss.label}_{ss.index}."
                )
            # Minimal validation of required fields to avoid silently
            # accepting wrong objects.
            try:
                if getattr(spec, "cutoff") <= 0 or getattr(spec, "index") <= 0:
                    raise DSLValidationError(
                        "Invalid boson spec for "
                        f"{ss.label}_{ss.index}: cutoff and index must be positive."
                    )
            except Exception as exc:
                if isinstance(exc, DSLValidationError):
                    raise
                raise DSLValidationError(
                    "Internal error: invalid boson spec for " f"{ss.label}_{ss.index}."
                ) from exc

        def _deformation_diag() -> Qobj:
            r"""
            Build diagonal Qobj containing f(n) for n=0,...,dim-1.

            Uses the callable attached to BosonSpec.deformation and
            validates shape/return type.
            """
            deform_fn = ss.spec.deformation
            if deform_fn is None:
                raise DSLValidationError(
                    f"Boson {ss.label}_{ss.index} has no deformation "
                    f"function, but deformed operator '{op_name}' was "
                    "requested."
                )
            n_vals = np.arange(dim, dtype=float)
            try:
                f_vals = np.asarray(deform_fn(n_vals))
            except Exception as exc:  # pragma: no cover - defensive
                raise DSLValidationError(
                    "Deformation callable for "
                    f"{ss.label}_{ss.index} raised an exception: {exc}"
                ) from exc
            if f_vals.shape != (dim,):
                raise DSLValidationError(
                    "Deformation callable for "
                    f"{ss.label}_{ss.index} must return shape ({dim},), "
                    f"got {f_vals.shape}."
                )
            return Qobj(np.diag(f_vals), dims=[[dim], [dim]])

        if op_name == "a":
            return destroy(dim)
        if op_name == "adag":
            a = destroy(dim)
            return a.dag()
        if op_name == "af":
            a = destroy(dim)
            return a * _deformation_diag()
        if op_name == "adagf":
            a = destroy(dim)
            return _deformation_diag() * a.dag()
        if op_name == "n":
            a = destroy(dim)
            return a.dag() * a

        raise DSLValidationError(
            "Unsupported boson operator "
            f"'{op_name}' for subsystem {ss.label}_{ss.index}."
        )

    def _make_local_custom_op(self, op_name: str, ss: SubsystemInfo) -> Qobj:
        r"""
        Build a local operator on a custom subsystem.

        Operators for custom subsystems are taken from the
        ``operators`` mapping on the corresponding
        :class:`CustomSpec`.

        Parameters
        ----------
        op_name : str
            Name of the operator defined in
            :attr:`CustomSpec.operators`.
        ss : SubsystemInfo
            Subsystem descriptor for the custom subsystem.

        Returns
        -------
        qutip.Qobj
            Local operator acting on the custom subsystem.

        Raises
        ------
        DSLValidationError
            If the subsystem does not have a :class:`CustomSpec`,
            if the operator is missing, if it is not a
            :class:`qutip.Qobj`, or if its dimensions do not match
            ``ss.dim``.

        Examples
        --------
        >>> import numpy as np
        >>> Jz = Qobj(np.diag([1.0, 0.0, -1.0]))
        >>> cfg = HilbertConfig(
        ...     qubits=[],
        ...     bosons=[],
        ...     customs=[CustomSpec(label="c", index=1, dim=3,
        ...                          operators={"Jz": Jz})],
        ... )
        >>> cache = QutipOperatorCache(cfg)
        >>> ss = cache._find_subsystem("custom", "c", 1)
        >>> op = cache._make_local_custom_op("Jz", ss)
        >>> op.shape
        (3, 3)
        """
        spec = ss.spec
        if not isinstance(spec, CustomSpec):
            raise DSLValidationError(
                "Internal error: expected CustomSpec for custom subsystem "
                f"{ss.label}_{ss.index}."
            )

        if op_name not in spec.operators:
            raise DSLValidationError(
                f"Custom subsystem {ss.label}_{ss.index} has no operator named "
                f"'{op_name}'."
            )

        op = spec.operators[op_name]

        if not isinstance(op, Qobj):
            raise DSLValidationError(
                "Custom operator "
                f"'{op_name}' for subsystem {ss.label}_{ss.index} is not a Qobj."
            )

        if op.dims[0][0] != ss.dim or op.dims[1][0] != ss.dim:
            raise DSLValidationError(
                "Custom operator "
                f"'{op_name}' for subsystem {ss.label}_{ss.index} "
                f"has dimension {op.dims}, expected ({ss.dim}x{ss.dim})."
            )

        return op

    def local_operator(self, ref: LocalOperatorRef) -> Qobj:
        r"""
        Return a local operator for a given :class:`LocalOperatorRef`.

        This method constructs, caches, and returns a local operator
        acting only on a single subsystem, ignoring the ``power`` field
        on the reference. Powers are applied later in
        :meth:`full_operator`.

        Parameters
        ----------
        ref : LocalOperatorRef
            Reference describing the operator kind, label, index, and
            operator name on a single subsystem.

        Returns
        -------
        qutip.Qobj
            Local operator acting on a Hilbert space of dimension
            determined by the target subsystem.

        Raises
        ------
        DSLValidationError
            If the referenced subsystem or operator is invalid.

        Examples
        --------
        >>> cfg = HilbertConfig(
        ...     qubits=[QubitSpec(label="q", index=1)],
        ...     bosons=[],
        ...     customs=[],
        ... )
        >>> cache = QutipOperatorCache(cfg)
        >>> ref = LocalOperatorRef(kind="qubit", label="q", index=1,
        ...                        op_name="sz", power=1)
        >>> op = cache.local_operator(ref)
        >>> op.shape
        (2, 2)
        """
        key = (ref.kind, ref.label, ref.index, ref.op_name)
        if key in self.local_ops:
            return self.local_ops[key]

        ss = self._find_subsystem(ref.kind, ref.label, ref.index)

        if ss.kind == "qubit":
            op = self._make_local_qubit_op(ref.op_name, ss)
        elif ss.kind == "boson":
            op = self._make_local_boson_op(ref.op_name, ss)
        elif ss.kind == "custom":
            op = self._make_local_custom_op(ref.op_name, ss)
        else:
            raise DSLValidationError(
                f"Unknown subsystem kind '{ss.kind}' for {ss.label}_{ss.index}."
            )

        self.local_ops[key] = op
        return op

    def full_operator(self, ref: LocalOperatorRef) -> Qobj:
        r"""
        Embed a local operator into the full Hilbert space.

        The local operator described by ``ref`` is raised to the power
        ``ref.power`` and embedded as a tensor product with identities
        on all other subsystems.

        Parameters
        ----------
        ref : LocalOperatorRef
            Reference describing the local operator and its power.

        Returns
        -------
        qutip.Qobj
            Operator acting on the full tensor-product space.

        Raises
        ------
        DSLValidationError
            If a negative power is requested or the subsystem cannot be
            found.

        Examples
        --------
        >>> cfg = HilbertConfig(
        ...     qubits=[QubitSpec(label="q", index=1)],
        ...     bosons=[BosonSpec(label="a", index=1, cutoff=3)],
        ...     customs=[],
        ... )
        >>> cache = QutipOperatorCache(cfg)
        >>> ref = LocalOperatorRef(kind="boson", label="a", index=1,
        ...                        op_name="a", power=2)
        >>> op = cache.full_operator(ref)
        >>> op.dims
        [[2, 3], [2, 3]]
        """
        key = (ref.kind, ref.label, ref.index, ref.op_name, ref.power)
        if key in self.full_ops:
            return self.full_ops[key]

        local_op = self.local_operator(ref)

        if ref.power == 1:
            local_pow = local_op
        else:
            if ref.power < 0:
                raise DSLValidationError(
                    f"Negative operator power {ref.power} is not supported "
                    f"for {ref.op_name}_{ref.index}."
                )
            local_pow = local_op**ref.power

        pos = self.subsystem_index[(ref.kind, ref.label, ref.index)]

        if not self.subsystems:
            full_op = local_pow
        else:
            factors = list(self._identity_factors or self.identities)
            factors[pos] = local_pow
            full_op = tensor(*factors)

        self.full_ops[key] = full_op
        return full_op


def _validate_custom_specs_for_qutip(config: HilbertConfig) -> None:
    r"""
    Validate that custom subsystem operators are QuTiP objects with correct dims.
    """
    for spec in config.customs:
        if not isinstance(spec, CustomSpec):
            raise DSLValidationError(
                "Custom subsystem entries must be CustomSpec instances, got "
                f"{type(spec)!r}."
            )
        for op_name, op in spec.operators.items():
            if not isinstance(op, Qobj):
                raise DSLValidationError(
                    "Custom operator "
                    f"'{op_name}' for {spec.label}_{spec.index} must be a "
                    f"qutip.Qobj; got {type(op).__name__}."
                )
            if op.dims[0][0] != spec.dim or op.dims[1][0] != spec.dim:
                raise DSLValidationError(
                    "Custom operator "
                    f"'{op_name}' for {spec.label}_{spec.index} has dims "
                    f"{op.dims}, expected ({spec.dim}x{spec.dim})."
                )


def term_to_qobj(term: Term, cache: QutipOperatorCache) -> Qobj:
    r"""
    Convert an IR term to a full-system operator.

    The operator part of a :class:`Term` is constructed as the product
    of embedded operators in the order they appear in
    ``term.ops``. The scalar coefficient ``term.scalar_expr`` is not
    applied here and is handled separately at the Hamiltonian level.

    Parameters
    ----------
    term : Term
        IR term containing a scalar expression and a list of operator
        references.
    cache : QutipOperatorCache
        Operator cache used to embed local operators into the full
        Hilbert space.

    Returns
    -------
    qutip.Qobj
        Operator acting on the full Hilbert space. If
        ``term.ops`` is empty, the global identity is returned.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> cache = QutipOperatorCache(cfg)
    >>> term = Term(
    ...     scalar_expr=sp.Symbol("g"),
    ...     ops=[LocalOperatorRef(kind="qubit", label="q", index=1,
    ...                           op_name="sx", power=1)],
    ... )
    >>> op = term_to_qobj(term, cache)
    >>> isinstance(op, Qobj)
    True
    """
    if not term.ops:
        return cache.global_identity
    return _operator_product(term.ops, cache)


def _term_to_qobj_static(
    term: Term, cache: QutipOperatorCache, params: Dict[str, complex]
) -> Qobj:
    r"""
    Static version that evaluates operator-function scalar factors using params.
    """
    if not term.ops:
        return cache.global_identity

    op_acc: Qobj | None = None
    for fac in term.ops:
        if isinstance(fac, LocalOperatorRef):
            op_q = cache.full_operator(fac)
        else:
            scale = _evaluate_scalar_static(fac.scalar_factor, params)
            op_q = _apply_operator_function_scaled(fac.func_name, fac.arg, scale, cache)
        op_acc = op_q if op_acc is None else op_acc * op_q
    return op_acc if op_acc is not None else cache.global_identity


def _operator_product(ops: List[OperatorFactor], cache: QutipOperatorCache) -> Qobj:
    r"""Multiply a sequence of operator factors."""
    op = _operator_factor_to_qobj(ops[0], cache)
    for ref in ops[1:]:
        op = op * _operator_factor_to_qobj(ref, cache)
    return op


def _operator_factor_to_qobj(factor: OperatorFactor, cache: QutipOperatorCache) -> Qobj:
    r"""
    Convert LocalOperatorRef or OperatorFunctionRef to a full-system Qobj.

    Note: this static version assumes any scalar_factor on an operator
    function is time independent and has already been numerically
    substituted if it involved parameters.
    """
    if isinstance(factor, OperatorFunctionRef):
        if factor.scalar_factor.free_symbols:
            raise DSLValidationError(
                "Internal error: operator function with non-numeric scalar factor "
                "passed to static operator conversion."
            )
        scale = complex(factor.scalar_factor)
        return _apply_operator_function_scaled(
            factor.func_name, factor.arg, scale, cache
        )
    return cache.full_operator(factor)


def _apply_operator_function_scaled(
    func_name: str,
    arg_ref: OperatorFunctionRef | LocalOperatorRef,
    scale: complex,
    cache: QutipOperatorCache,
) -> Qobj:
    r"""
    Evaluate an operator-valued function (exp, cos, sin) on ``scale * O``.
    """
    op = cache.full_operator(
        arg_ref if isinstance(arg_ref, LocalOperatorRef) else arg_ref  # type: ignore
    )
    arr = scale * op.full()
    try:
        res = apply_operator_function(arr, func_name, backend="numpy")
    except ValueError as exc:
        raise DSLValidationError(
            "Unsupported operator function "
            f"'{func_name}'. Allowed: {sorted(ALLOWED_OPERATOR_FUNCTIONS)}."
        ) from exc
    return Qobj(res, dims=op.dims)


def _build_time_dep_operator_function_callable(
    term: Term,
    cache: QutipOperatorCache,
    params: Dict[str, complex],
    t_name: str,
    time_symbols: tuple[str, ...] | None = None,
) -> Callable[[float, Dict[str, Any]], Qobj]:
    r"""
    Build a callable returning a Qobj for terms where an operator function
    carries time dependence in its scalar factor.

    Currently supports general products; operator functions with static
    scalar factors are pre-evaluated, and those with time dependence are
    evaluated on each call.
    """
    time_names = {t_name} | (set(time_symbols) if time_symbols else set())

    # Scalar coefficient callable
    if _expr_has_time(term.scalar_expr, time_names):
        scalar_fn, _, _ = _build_time_dep_term_callable(
            term.scalar_expr, t_name, time_symbols
        )
        scalar_const: complex | None = None
    else:
        scalar_const = _evaluate_scalar_static(term.scalar_expr, params)
        scalar_fn = None

    # Precompute factors
    factors: list[tuple[str, Any]] = []
    for fac in term.ops:
        if isinstance(fac, LocalOperatorRef):
            op_q = cache.full_operator(fac)
            factors.append(("static", op_q))
            continue

        # OperatorFunctionRef
        if _expr_has_time(fac.scalar_factor, time_names):
            scale_fn, _, _ = _build_time_dep_term_callable(
                fac.scalar_factor, t_name, time_symbols
            )
            factors.append(("opfunc_time", fac.func_name, fac.arg, scale_fn))
        else:
            scale_val = _evaluate_scalar_static(fac.scalar_factor, params)
            op_q = _apply_operator_function_scaled(
                fac.func_name, fac.arg, scale_val, cache
            )
            factors.append(("opfunc_static", op_q))

    def _call(t: float, args: Dict[str, Any]) -> Qobj:
        r"""
        Evaluate the operator product at time ``t`` with scalar envelope applied.
        """
        op_acc: Qobj | None = None
        for item in factors:
            tag = item[0]
            if tag == "static":
                op_q = item[1]
            elif tag == "opfunc_static":
                op_q = item[1]
            elif tag == "opfunc_time":
                _, func_name, arg_ref, scale_fn = item
                scale_val = scale_fn(t, args)
                op_q = _apply_operator_function_scaled(
                    func_name, arg_ref, scale_val, cache
                )
            else:
                raise DSLValidationError("Internal error: unknown factor tag.")
            op_acc = op_q if op_acc is None else op_acc * op_q

        if op_acc is None:
            op_acc = cache.global_identity

        scalar_val = scalar_const if scalar_fn is None else scalar_fn(t, args)
        return scalar_val * op_acc

    return _call


def _param_aliases(name: str) -> List[str]:
    r"""Generate possible parameter aliases for LaTeX-escaped symbols."""
    return param_aliases(name)


def _lookup_param_name(name: str, params: Dict[str, complex]) -> tuple[str, complex]:
    r"""Resolve a parameter key using alias rules and return the matched value."""
    return resolve_param(
        name, params, warn_on_multiple=True, logger=logger  # preserves warning UX
    )


def _evaluate_scalar_static(expr: sp.Expr, params: Dict[str, complex]) -> complex:
    r"""
    Evaluate a time-independent scalar SymPy expression.

    The expression is assumed to contain only scalar symbols and
    scalar-valued functions. Parameters are substituted from ``params``
    using :func:`_lookup_param_name`, and the result is evaluated to a
    complex number.

    Parameters
    ----------
    expr : sympy.Expr
        Scalar SymPy expression with no operator-valued symbols.
    params : dict
        Mapping from user-facing parameter keys to numeric values
        (float or complex). Keys may be given in forms such as
        ``"omega_c"`` or ``"omega_{c}"``.

    Returns
    -------
    complex
        Numeric value obtained after substitution and evaluation.

    Raises
    ------
    DSLValidationError
        If the expression still contains free symbols after parameter
        substitution.

    Examples
    --------
    >>> g = sp.Symbol("g")
    >>> A = sp.Symbol("A")
    >>> expr = g * A + 1
    >>> val = _evaluate_scalar_static(expr, {"g": 0.5, "A": 2.0})
    >>> float(val)
    2.0
    """
    # If there are AppliedUndef heads whose name matches a parameter, treat
    # them as implicit scalar multiplication of their arguments, e.g.
    #   kappa(x) -> kappa * x
    # so that parameter substitution works as expected.
    replacements = {}
    for f in expr.atoms(sp.Function):
        fname = f.func.__name__
        if fname in params:
            if f.args:
                arg_prod = sp.Mul(*f.args)
            else:
                arg_prod = 1
            replacements[f] = sp.Symbol(fname) * arg_prod
    if replacements:
        expr = expr.xreplace(replacements)

    subs_map = {}
    # Use all Symbol atoms, not just free_symbols, because SymPy sometimes
    # omits well-known constants (e.g., Greek letters) from free_symbols.
    for s in expr.atoms(sp.Symbol):
        name = s.name
        _, value = _lookup_param_name(name, params)
        subs_map[s] = value

    # Also catch non-Symbol atoms whose string representation matches a param
    # (e.g., SymPy's built-in kappa constant).
    for atom in expr.atoms():
        key = str(atom)
        if key in params and atom not in subs_map:
            subs_map[atom] = params[key]

    expr_sub = expr.subs(subs_map)

    if expr_sub.free_symbols:
        missing = {s.name for s in expr_sub.free_symbols}
        raise DSLValidationError(
            f"Scalar expression still has free symbols {missing} after substitution; "
            "check your parameter dict."
        )

    return (
        complex(expr_sub.evalf()) if hasattr(expr_sub, "evalf") else complex(expr_sub)
    )


def compile_static_hamiltonian_ir(
    ir: HamiltonianIR,
    config: HilbertConfig,
    params: Dict[str, complex],
    cache: QutipOperatorCache | None = None,
) -> Qobj:
    r"""
    Compile a time-independent Hamiltonian IR into a QuTiP ``Qobj``.
    """
    backend = QutipBackend()
    cache = cache or backend._make_cache(config, options=None)
    return backend._compile_static(ir, cache, params, options=None)


def compile_static_hamiltonian_from_latex(
    H_latex: str,
    config: HilbertConfig,
    params: Dict[str, complex],
) -> Qobj:
    r"""
    Compile a static Hamiltonian directly from LaTeX.

    This is a convenience wrapper for the static path

    .. math::

        H(\\text{LaTeX}) \\rightarrow \\text{IR} \\rightarrow H \\text{ (Qobj)}.

    Parameters
    ----------
    H_latex : str
        Physics-style LaTeX expression for the Hamiltonian
        :math:`H`.
    config : HilbertConfig
        Hilbert-space configuration describing all subsystems.
    params : dict
        Parameter dictionary used to evaluate scalar coefficients.

    Returns
    -------
    qutip.Qobj
        Static Hamiltonian operator.

    Raises
    ------
    DSLValidationError
        If the LaTeX expression produces a time-dependent IR.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> H_latex = r"\\frac{\\omega_0}{2} \\sigma_{z,1}"
    >>> H = compile_static_hamiltonian_from_latex(H_latex, cfg, {"omega_0": 1.0})
    >>> isinstance(H, Qobj)
    True
    """
    ir = latex_to_ir(H_latex, config, t_name="t")
    return compile_static_hamiltonian_ir(ir, config, params)


@dataclass
class CompiledHamiltonianQutip(CompiledHamiltonianBase):
    r"""Compiled Hamiltonian components in QuTiP format."""

    H: Any
    H0: Qobj
    time_terms: List[Tuple[Any, Callable[[float, Dict[str, Any]], Any]]]
    args: Dict[str, Any]
    time_dependent: bool


@dataclass
class CompiledOpenSystemQutip(CompiledOpenSystemBase):
    r"""Compiled open-system container for QuTiP."""


class QutipBackend(BackendBase):
    r"""QuTiP backend for compiling IR to QuTiP ``Qobj`` structures."""

    def _make_cache(
        self, config: HilbertConfig, options: BackendOptions | None
    ) -> QutipOperatorCache:
        r"""Prepare a cache of QuTiP operators for the given configuration."""
        _validate_custom_specs_for_qutip(config)
        return QutipOperatorCache(config)

    def _compile_static(
        self,
        ir: HamiltonianIR,
        cache: QutipOperatorCache,
        params: Dict[str, complex],
        options: BackendOptions | None = None,
    ) -> Qobj:
        r"""Compile a static Hamiltonian IR into a QuTiP ``Qobj``."""
        if ir.has_time_dep:
            raise DSLValidationError(
                "compile_static called on time-dependent IR; use time-dependent path."
            )
        H: Qobj | None = None
        for term in ir.terms:
            coeff = _evaluate_scalar_static(term.scalar_expr, params)
            if abs(coeff) == 0:
                continue
            op = _term_to_qobj_static(term, cache, params)
            contrib = coeff * op
            H = contrib if H is None else H + contrib
        return H if H is not None else 0 * cache.global_identity

    def _compile_time_dependent(
        self,
        ir: HamiltonianIR,
        cache: QutipOperatorCache,
        params: Dict[str, complex],
        *,
        t_name: str,
        time_symbols: tuple[str, ...] | None,
        options: BackendOptions | None = None,
    ) -> CompiledHamiltonianQutip:
        r"""
        Compile a time-dependent Hamiltonian IR into QuTiP list/list callable form.
        """
        if not ir.has_time_dep:
            H_static = self._compile_static(ir, cache, params, options=options)
            return CompiledHamiltonianQutip(
                H=H_static,
                H0=H_static,
                time_terms=[],
                args=dict(params),
                time_dependent=False,
            )

        H0: Qobj | None = None
        time_terms: List[Tuple[Any, Callable[[float, Dict[str, Any]], Any]]] = []
        time_names = {t_name} | (set(time_symbols) if time_symbols else set())

        for term in ir.terms:
            scalar = term.scalar_expr
            has_t_scalar = _expr_has_time(scalar, time_names)
            has_t_opfunc = any(
                isinstance(opf, OperatorFunctionRef)
                and _expr_has_time(opf.scalar_factor, time_names)
                for opf in term.ops
            )

            if not has_t_scalar and not has_t_opfunc:
                coeff = _evaluate_scalar_static(scalar, params)
                if abs(coeff) == 0:
                    continue
                op = _term_to_qobj_static(term, cache, params)
                contrib = coeff * op
                H0 = contrib if H0 is None else H0 + contrib
                continue

            if has_t_opfunc:
                term_callable = _build_time_dep_operator_function_callable(
                    term, cache, params, t_name, time_symbols
                )
                time_terms.append((None, term_callable))
            else:
                op = _term_to_qobj_static(term, cache, params)
                f_k, _, _ = _build_time_dep_term_callable(scalar, t_name, time_symbols)
                time_terms.append((op, f_k))

        if H0 is None:
            H0 = 0 * cache.global_identity

        H_list: List[Any] = [H0]
        for Hk, fk in time_terms:
            H_list.append(fk if Hk is None else [Hk, fk])

        return CompiledHamiltonianQutip(
            H=H_list,
            H0=H0,
            time_terms=time_terms,
            args=dict(params),
            time_dependent=True,
        )

    def compile_open_system_from_latex(
        self,
        H_latex: str,
        c_ops_latex: List[str],
        config: HilbertConfig,
        params: Dict[str, complex],
        t_name: str = "t",
        time_symbols: tuple[str, ...] | None = None,
    ) -> CompiledOpenSystemQutip:
        r"""
        Compile Hamiltonian and collapse operators from LaTeX into QuTiP objects.
        """
        cache = self._make_cache(config, options=None)
        H_compiled = self._compile_time_dependent(
            latex_to_ir(H_latex, config, t_name=t_name, time_symbols=time_symbols),
            cache,
            params,
            t_name=t_name,
            time_symbols=time_symbols,
            options=None,
        )

        if c_ops_latex:
            c_ops, _, c_td = compile_collapse_ops_from_latex(
                c_ops_latex,
                config,
                params,
                t_name=t_name,
                cache=cache,
                time_symbols=time_symbols,
            )
        else:
            c_ops, _, c_td = [], dict(params), False

        args = dict(params)
        return CompiledOpenSystemQutip(
            H=H_compiled.H,
            c_ops=c_ops,
            args=args,
            config=config,
            time_dependent=(H_compiled.time_dependent or c_td),
        )


def _expr_has_time(expr: sp.Expr, time_names: set[str]) -> bool:
    r"""Return True if the expression depends on any of the given time symbols."""
    return _expr_has_time_shared(expr, time_names)


@lru_cache(maxsize=256)
def _build_time_dep_term_callable_cached(
    scalar_expr: sp.Expr,
    t_name: str,
    time_symbols: tuple[str, ...],
) -> Tuple[
    Callable[[float, Dict[str, Any]], complex], List[sp.Symbol], List[List[str]]
]:
    r"""
    Build a QuTiP-compatible envelope function for a time-dependent term.

    The input scalar expression is assumed to depend on a time symbol
    ``t_name`` and on zero or more scalar parameters. A callable of the
    form

    .. math::

        f_k(t, \\text{args}) \\rightarrow \\mathbb{C}

    is returned together with information about which parameter symbols
    it uses and which aliases are tried for each parameter.

    Parameters
    ----------
    scalar_expr : sympy.Expr
        Scalar expression that depends on time and scalar parameters.
    t_name : str
        Name of the primary time symbol, for example ``"t"``.
    time_symbols : tuple of str
        Additional symbol names that should be treated as time-like for
        the purpose of detecting time dependence.

    Returns
    -------
    f_k : callable
        Callable ``f_k(t, args)`` used by QuTiP for time-dependent
        coefficients.
    param_syms : list of sympy.Symbol
        List of parameter symbols (excluding the time symbol) that
        appear in ``scalar_expr``.
    param_aliases : list of list of str
        For each parameter symbol, the list of alias keys to try when
        looking up a value in the ``args`` dictionary.

    Raises
    ------
    DSLValidationError
        If no time symbol corresponding to ``t_name`` or
        ``time_symbols`` can be found in ``scalar_expr``.

    Examples
    --------
    >>> t = sp.Symbol("t")
    >>> omega_d = sp.Symbol("omega_{d}")
    >>> expr = omega_d * sp.cos(t)
    >>> f_k, param_syms, _ = _build_time_dep_term_callable(expr, t_name="t")
    >>> float(f_k(0.5, {"omega_d": 2.0}))
    1.7551651...
    """
    free_syms = list(scalar_expr.free_symbols)
    # Normalize built-in symbols (I, E, pi) to SymPy constants.
    builtin_map = {"I": sp.I, "E": sp.E, "pi": sp.pi}
    for sym in list(free_syms):
        repl = builtin_map.get(sym.name)
        if repl is not None and sym != repl:
            scalar_expr = scalar_expr.subs(sym, repl)
    free_syms = list(scalar_expr.free_symbols)
    time_names = {t_name} | set(time_symbols)
    t_syms = [s for s in free_syms if s.name in time_names]
    if not t_syms:
        raise DSLValidationError(
            "Internal error: _build_time_dep_term_callable called on "
            "a scalar_expr that does not depend on time."
        )
    t_sym = t_syms[0]

    builtin_names = {"I", "E", "pi"}
    builtin_syms = {sp.I, sp.E, sp.pi}
    param_syms = [
        s
        for s in free_syms
        if s is not t_sym and s not in builtin_syms and s.name not in builtin_names
    ]
    param_syms_sorted = sorted(param_syms, key=lambda s: s.name)
    param_aliases = [_param_aliases(s.name) for s in param_syms_sorted]

    # Build L(t, *params)
    # Lambdify using NumPy; if scalar_expr contains unsupported functions,
    # this will raise (desired for unsupported envelopes).
    if param_syms_sorted:
        lmbd = sp.lambdify((t_sym, *param_syms_sorted), scalar_expr, modules="numpy")
    else:
        lmbd = sp.lambdify((t_sym,), scalar_expr, modules="numpy")

    def f_k(t: float, args: Dict[str, Any]) -> complex:
        r"""
        Evaluate the time-dependent scalar envelope at time ``t``.
        """
        vals: List[float | complex] = []
        for aliases in param_aliases:
            value = None
            for key in aliases:
                if isinstance(args, dict) and key in args:
                    value = args[key]
                    break
            if value is None:
                raise KeyError(
                    f"Missing value for parameter; tried keys {aliases}. "
                    f"Available keys: {sorted(args.keys())}"
                )
            vals.append(value)

        if vals:
            val = lmbd(t, *vals)
        else:
            val = lmbd(t)
        return complex(val)

    return f_k, param_syms_sorted, param_aliases


def _build_time_dep_term_callable(
    scalar_expr: sp.Expr,
    t_name: str,
    time_symbols: tuple[str, ...] | None = None,
) -> Tuple[
    Callable[[float, Dict[str, Any]], complex], List[sp.Symbol], List[List[str]]
]:
    r"""
    Wrapper that normalizes time symbols and uses a cached builder.
    """
    norm_time_symbols = tuple(time_symbols) if time_symbols else tuple()
    return _build_time_dep_term_callable_cached(scalar_expr, t_name, norm_time_symbols)


def compile_time_dependent_hamiltonian_ir(
    ir: HamiltonianIR,
    config: HilbertConfig,
    params: Dict[str, complex],
    t_name: str = "t",
    cache: QutipOperatorCache | None = None,
    time_symbols: tuple[str, ...] | None = None,
) -> CompiledHamiltonianQutip:
    r"""
    Compile a Hamiltonian IR (static or time dependent) into QuTiP form.
    """
    backend = QutipBackend()
    cache = cache or backend._make_cache(config, options=None)
    if not ir.has_time_dep:
        t_name = ""
    return backend._compile_time_dependent(
        ir,
        cache,
        params,
        t_name=t_name,
        time_symbols=time_symbols,
        options=None,
    )


def compile_hamiltonian_from_latex(
    H_latex: str,
    config: HilbertConfig,
    params: Dict[str, complex],
    t_name: str = "t",
) -> CompiledHamiltonianQutip:
    r"""
    Compile a Hamiltonian directly from LaTeX into QuTiP format.

    This is a high-level convenience wrapper that runs the full
    pipeline

    .. math::

        H(\\text{LaTeX}) \\rightarrow \\text{IR} \\rightarrow \\text{QuTiP},

    and returns a :class:`CompiledHamiltonianQutip` object.

    Parameters
    ----------
    H_latex : str
        Physics-style LaTeX expression for the Hamiltonian
        :math:`H(t)`.
    config : HilbertConfig
        Hilbert-space configuration describing all subsystems.
    params : dict
        Parameter dictionary used to evaluate scalar coefficients and
        envelopes.
    t_name : str, optional
        Primary time symbol name, by default ``"t"``.

    Returns
    -------
    CompiledHamiltonianQutip
        Compiled Hamiltonian in QuTiP format.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> H_latex = r"A \\cos(\\omega t) \\sigma_{x,1}"
    >>> compiled = compile_hamiltonian_from_latex(
    ...     H_latex, cfg, {"A": 0.5, "omega": 1.0}, t_name="t"
    ... )
    >>> isinstance(compiled.H, list)
    True
    """
    ir = latex_to_ir(H_latex, config, t_name=t_name)
    return compile_time_dependent_hamiltonian_ir(ir, config, params, t_name=t_name)


def compile_collapse_ops_from_latex(
    c_ops_latex: List[str],
    config: HilbertConfig,
    params: Dict[str, complex],
    t_name: str = "t",
    cache: QutipOperatorCache | None = None,
    time_symbols: tuple[str, ...] | None = None,
) -> tuple[List[Any], Dict[str, Any], bool]:
    r"""
    Compile LaTeX collapse operators into QuTiP ``c_ops`` format.

    Each string in ``c_ops_latex`` is interpreted as one collapse
    channel. Static collapse operators are summed into a single
    :class:`qutip.Qobj`. Time-dependent collapse operators are
    restricted to single monomials and compiled as ``[C0, f(t, args)]``
    where :math:`C_0` is the operator part and :math:`f` is a scalar
    envelope.

    Parameters
    ----------
    c_ops_latex : list of str
        LaTeX strings describing collapse channels.
    config : HilbertConfig
        Hilbert-space configuration used to construct operators.
    params : dict
        Parameter dictionary used in scalar coefficients and envelopes.
    t_name : str, optional
        Name of the primary time symbol, by default ``"t"``.
    cache : QutipOperatorCache or None, optional
        Optional operator cache. If ``None``, a new cache is built from
        ``config``.
    time_symbols : tuple of str or None, optional
        Additional time-like symbol names considered when detecting
        time dependence.

    Returns
    -------
    c_ops : list
        List suitable as QuTiP ``c_ops`` argument. Elements are either
        static :class:`qutip.Qobj` instances or time-dependent pairs
        ``[Qobj, f(t, args)]``.
    args : dict
        Parameter dictionary to be passed to QuTiP together with
        ``c_ops``.
    any_time_dep : bool
        Flag indicating whether at least one collapse operator is
        time dependent.

    Raises
    ------
    DSLValidationError
        If a time-dependent collapse operator produces more than one
        IR term or if a collapse term has no operator part.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> c_ops_latex = [r"\\sqrt{\\gamma} \\sigma_{-,1}"]
    >>> c_ops, args, td = compile_collapse_ops_from_latex(
    ...     c_ops_latex, cfg, {"gamma": 0.1}, t_name="t"
    ... )
    >>> len(c_ops), td
    (1, False)
    """
    cache = cache or QutipOperatorCache(config)
    c_ops: List[Any] = []
    any_time_dep = False
    _validate_custom_specs_for_qutip(config)
    logger.debug(
        "Compiling %d collapse ops, subsystems=%s", len(c_ops_latex), cache.subsystems
    )

    for c_latex in c_ops_latex:
        ir = latex_to_ir(c_latex, config, t_name=t_name, time_symbols=time_symbols)

        if not ir.terms:
            raise DSLValidationError(
                f"Collapse operator LaTeX {c_latex!r} contains no operator terms."
            )

        # Check if any term is explicitly time-dependent
        time_names = set(time_symbols) if time_symbols else set()
        time_names.add(t_name)
        has_t_any = any(
            any(s.name in time_names for s in term.scalar_expr.free_symbols)
            for term in ir.terms
        )

        if not has_t_any:
            # Purely static collapse operator; we can safely sum all terms.
            C: Qobj | None = None
            for term in ir.terms:
                if not term.ops:
                    raise DSLValidationError(
                        "Collapse operator "
                        f"{c_latex!r} has a term with no operator part."
                    )
                coeff = _evaluate_scalar_static(term.scalar_expr, params)
                if abs(coeff) == 0:
                    continue
                op_mat = term_to_qobj(term, cache)
                contrib = coeff * op_mat
                C = contrib if C is None else C + contrib

            if C is None:
                # All coefficients vanished numerically; skip.
                continue

            c_ops.append(C)
        else:
            # Time-dependent collapse operator. For correctness with QuTiP's
            # [C0, f(t,args)] representation, we only support a single
            # monomial.
            if len(ir.terms) != 1:
                raise DSLValidationError(ERROR_HINT_TIME_DEP_COLLAPSE)

            term = ir.terms[0]
            if not term.ops:
                raise DSLValidationError(
                    "Time-dependent collapse operator "
                    f"{c_latex!r} has no operator part."
                )

            scalar = term.scalar_expr
            free_syms = scalar.free_symbols
            has_t = any(s.name in time_names for s in free_syms)

            # If scalar simplifies to zero after substituting params (time → 0),
            # skip quietly.
            subs_all: Dict[sp.Symbol, Any] = {}
            for s in list(free_syms):
                if s.name in time_names:
                    subs_all[s] = 0
                    continue
                try:
                    _, val = _lookup_param_name(s.name, params)
                except DSLValidationError:
                    continue
                subs_all[s] = val
            scalar_eval = sp.simplify(scalar.subs(subs_all))
            if scalar_eval.is_number and complex(scalar_eval) == 0:
                continue

            if not has_t:
                # Should not happen here, but fall back gracefully.
                coeff = _evaluate_scalar_static(scalar, params)
                op_mat = term_to_qobj(term, cache)
                C = coeff * op_mat
                c_ops.append(C)
                continue

            # Operators: fixed O
            op_mat = term_to_qobj(term, cache)
            # Envelope: f(t, args) = scalar_expr(t, params)
            # Prefer the primary t_name for the callable; extra time_symbols
            # are treated as parameters.
            f_k, _, _ = _build_time_dep_term_callable(scalar, t_name, time_symbols)

            try:
                val0 = f_k(0.0, dict(params))
                if abs(val0) == 0:
                    continue
            except Exception:
                pass

            c_ops.append([op_mat, f_k])
            any_time_dep = True

    return c_ops, dict(params), any_time_dep


def compile_open_system_from_latex(
    H_latex: str,
    c_ops_latex: List[str] | None,
    config: HilbertConfig,
    params: Dict[str, complex],
    t_name: str = "t",
    time_symbols: tuple[str, ...] | None = None,
) -> CompiledOpenSystemQutip:
    r"""
    Compile an open quantum system from LaTeX into QuTiP format.

    This function compiles both the Hamiltonian and collapse operators
    for use with QuTiP solvers. It supports static and time-dependent
    Hamiltonians and static or time-dependent collapse operators,
    subject to the monomial restriction for time-dependent channels.

    Parameters
    ----------
    H_latex : str
        Physics-style LaTeX expression for the Hamiltonian
        :math:`H(t)`.
    c_ops_latex : list of str or None
        LaTeX expressions for collapse channels. If ``None`` or empty,
        the system is treated as closed.
    config : HilbertConfig
        Hilbert-space configuration describing all subsystems.
    params : dict
        Parameter dictionary used in both Hamiltonian and collapse
        operator coefficients and envelopes.
    t_name : str, optional
        Primary time symbol name, by default ``"t"``.
    time_symbols : tuple of str or None, optional
        Additional time-like symbols considered when detecting time
        dependence in both Hamiltonian and collapse operators.

    Returns
    -------
    CompiledOpenSystemQutip
        Compiled open quantum system with Hamiltonian, collapse
        operators, parameter dictionary, and configuration.

    Examples
    --------
    >>> cfg = HilbertConfig(
    ...     qubits=[QubitSpec(label="q", index=1)],
    ...     bosons=[],
    ...     customs=[],
    ... )
    >>> H_latex = r"\\frac{\\omega_0}{2} \\sigma_{z,1}"
    >>> c_ops_latex = [r"\\sqrt{\\gamma} \\sigma_{-,1}"]
    >>> params = {"omega_0": 1.0, "gamma": 0.1}
    >>> model = compile_open_system_from_latex(
    ...     H_latex, c_ops_latex, cfg, params, t_name="t"
    ... )
    >>> isinstance(model.H, Qobj)
    True
    """
    backend = QutipBackend()
    return backend.compile_open_system_from_latex(
        H_latex=H_latex,
        c_ops_latex=c_ops_latex or [],
        config=config,
        params=params,
        t_name=t_name,
        time_symbols=time_symbols,
    )


def _term_ops_signature(terms: List[Term]) -> set[tuple[str, str, int, int]]:
    r"""Return a set of (kind, op_name, index, power) tuples found in IR terms."""
    sig: set[tuple[str, str, int, int]] = set()
    for term in terms:
        for r in term.ops:
            sig.add((r.kind, r.op_name, r.index, r.power))
    return sig


if __name__ == "__main__":  # pragma: no cover
    print("Run pytest to execute the test suite.")
