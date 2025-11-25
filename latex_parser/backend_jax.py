# backend_jax.py
#
# Pure JAX backend for the LaTeX → DSL → IR pipeline.
#
# Responsibilities:
#   - Take HamiltonianIR (from ir.py) plus a HilbertConfig (from dsl.py).
#   - Build dense JAX arrays for each IR term using tensor products.
#   - Evaluate scalar SymPy coefficients with a user-supplied parameter dict.
#   - Return:
#       * a single static Hamiltonian matrix for time-independent problems.
#
# Limitations (current step):
#   - Only static (time-independent) scalar coefficients are supported.
#   - Time-dependent Hamiltonians or collapse operators raise an error.
#   - No built-in time-envelopes or Liouvillian builders yet.
#
# This backend is intended for differentiable workflows in JAX:
#
#   H(LaTeX) -> IR -> dense jax.numpy arrays
#
# so that you can plug the resulting matrices into your own JAX-based
# solvers / optimizers.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Tuple

import sympy as sp

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
    lookup_param_name,
    param_aliases,
)
from latex_parser.dsl import (  # type: ignore
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    LocalOperatorRef,
)
from latex_parser.dsl_constants import (  # type: ignore
    ALLOWED_OPERATOR_FUNCTIONS,
    ERROR_HINT_TIME_DEP_COLLAPSE,
)
from latex_parser.errors import BackendUnavailableError
from latex_parser.ir import (  # type: ignore
    HamiltonianIR,
    OperatorFunctionRef,
    Term,
    latex_to_ir,
)
from latex_parser.operator_functions import apply_operator_function

# JAX is optional; defer hard failure until the backend is actually used.
_JAX_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - optional dependency
    import jax as jax  # type: ignore
except Exception as exc:  # pragma: no cover - defensive import
    jax = None  # type: ignore
    _JAX_IMPORT_ERROR = exc

jnp = None
DEFAULT_DTYPE: Any = complex
_JAX_AVAILABLE = False

if jax is not None:
    try:  # pragma: no cover - optional dependency setup
        # Default to x64 for higher fidelity; default to CPU unless the user
        # requested a platform via environment variables.
        jax.config.update("jax_enable_x64", True)
        if not (os.environ.get("JAX_PLATFORM_NAME") or os.environ.get("JAX_PLATFORMS")):
            jax.config.update("jax_platform_name", "cpu")
        import jax.numpy as jnp  # type: ignore

        DEFAULT_DTYPE = jnp.complex128
        _JAX_AVAILABLE = True
    except Exception as exc:  # pragma: no cover - defensive import
        _JAX_IMPORT_ERROR = exc
        jax = None  # type: ignore
        jnp = None
        DEFAULT_DTYPE = complex
        _JAX_AVAILABLE = False


def _require_jax() -> tuple[Any, Any]:
    r"""
    Raise a clear error if JAX is unavailable. Returns (jax, jnp) for convenience.
    """
    if not _JAX_AVAILABLE or jax is None or jnp is None:
        raise BackendUnavailableError(
            "jax is required for backend_jax; install with `pip install jax jaxlib`."
        ) from _JAX_IMPORT_ERROR
    return jax, jnp


logger = logging.getLogger(__name__)


def _collect_param_names(ir: HamiltonianIR, time_names: set[str]) -> set[str]:
    r"""Collect parameter symbol names from scalar expressions, excluding
    time/builtins."""
    builtin_names = {"I", "E", "pi"}
    names: set[str] = set()
    for term in ir.terms:
        for sym in term.scalar_expr.free_symbols:
            if sym.name not in time_names and sym.name not in builtin_names:
                names.add(sym.name)
        for op in term.ops:
            if isinstance(op, OperatorFunctionRef):
                for sym in op.scalar_factor.free_symbols:
                    if sym.name not in time_names and sym.name not in builtin_names:
                        names.add(sym.name)
    return names


@dataclass(frozen=True)
class JaxBackendOptions:
    r"""
    Backend configuration for the JAX compiler.

    Attributes
    ----------
    dtype : jax.numpy.dtype
        Complex dtype used for all constructed arrays.
    platform : str or None
        Optional JAX platform override (e.g., "cpu", "gpu").
    """

    dtype: Any = DEFAULT_DTYPE
    platform: str | None = None


def _merge_options(
    options: JaxBackendOptions | None,
    *,
    dtype: Any | None = None,
    platform: str | None = None,
) -> JaxBackendOptions:
    r"""
    Combine explicit dtype/platform overrides with an optional options object.
    """
    base = options or JaxBackendOptions()
    final_dtype = dtype if dtype is not None else base.dtype
    final_platform = platform if platform is not None else base.platform
    return JaxBackendOptions(dtype=final_dtype, platform=final_platform)


def _apply_platform(platform: str | None) -> None:
    r"""
    Optionally set the JAX platform. This is a no-op if platform is None.
    """
    _require_jax()
    if platform:
        jax.config.update("jax_platform_name", platform)


def _jax_kron(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    r"""
    Kronecker product of two matrices using ``jax.numpy`` only.

    Parameters
    ----------
    a : jax.numpy.ndarray
        Left factor of shape ``(m, n)``.
    b : jax.numpy.ndarray
        Right factor of shape ``(p, q)``.

    Returns
    -------
    jax.numpy.ndarray
        Kronecker product of shape ``(m p, n q)``.
    """
    _require_jax()
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    a_m, a_n = a.shape
    b_p, b_q = b.shape

    # Arrange axes so that row index = i * b_p + k and column index = j * b_q + l.
    a_exp = a[:, None, :, None]
    b_exp = b[None, :, None, :]
    return (a_exp * b_exp).reshape(a_m * b_p, a_n * b_q)


class JaxOperatorCache(BaseOperatorCache[Any]):
    r"""
    Cache JAX operators for a fixed Hilbert configuration.

    This class mirrors :class:`QutipOperatorCache` but builds dense
    ``jax.numpy`` arrays instead of QuTiP ``Qobj`` instances.

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
        Mapping ``(kind, label, index) -> position`` in the tensor
        product.
    identities : list of jax.numpy.ndarray
        Local identity matrices for each subsystem.
    global_identity : jax.numpy.ndarray
        Identity on the full tensor-product Hilbert space.
    """

    def __init__(self, config: HilbertConfig, *, dtype: Any = DEFAULT_DTYPE):
        r"""
        Initialize the cache and precompute local/global operators.
        """
        _require_jax()
        self.dtype = jnp.dtype(dtype)
        self.local_ops: Dict[Tuple[str, str, int, str], jnp.ndarray] = {}
        self.full_ops: Dict[Tuple[str, str, int, str, int], jnp.ndarray] = {}
        super().__init__(config)

    # BaseOperatorCache hooks ------------------------------------------------
    def _local_identity(self, dim: int) -> jnp.ndarray:
        """Return a dense JAX identity matrix with the configured dtype."""
        _require_jax()
        return jnp.eye(dim, dtype=self.dtype)

    def _kron(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Kronecker product for dense JAX arrays."""
        return _jax_kron(a, b)

    def _find_subsystem(self, kind: str, label: str, index: int) -> SubsystemInfo:
        r"""
        Look up a subsystem descriptor by kind/label/index.
        """
        key = (kind, label, index)
        pos = self.subsystem_index.get(key)
        if pos is None:
            raise DSLValidationError(f"Unknown subsystem {key} in HilbertConfig.")
        return self.subsystems[pos]

    def local_operator(self, ref: LocalOperatorRef) -> jnp.ndarray:
        r"""
        Return a local operator for a given :class:`LocalOperatorRef`.

        The operator acts only on a single subsystem and has shape
        ``(dim, dim)`` where ``dim`` is the local Hilbert-space dimension.
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

    def full_operator(self, ref: LocalOperatorRef) -> jnp.ndarray:
        r"""
        Embed a local operator into the full Hilbert space.

        The local operator described by ``ref`` is raised to the power
        ``ref.power`` and embedded as a tensor product with identities
        on all other subsystems.
        """
        key = (ref.kind, ref.label, ref.index, ref.op_name, ref.power)
        if key in self.full_ops:
            return self.full_ops[key]

        if ref.power < 0:
            raise DSLValidationError(
                f"Negative powers are not supported for operator "
                f"{ref.op_name}_{ref.index}."
            )

        local = self.local_operator(
            LocalOperatorRef(
                kind=ref.kind,
                label=ref.label,
                index=ref.index,
                op_name=ref.op_name,
                power=1,
            )
        )

        if ref.power == 0:
            local_pow = jnp.eye(local.shape[0], dtype=self.dtype)
        elif ref.power == 1:
            local_pow = local
        else:
            local_pow = local
            for _ in range(ref.power - 1):
                local_pow = local_pow @ local

        if not self.subsystems:
            full_op = local_pow
        else:
            pos = self.subsystem_index[(ref.kind, ref.label, ref.index)]
            factors = list(self._identity_factors or self.identities)
            factors[pos] = local_pow
            full_op = factors[0]
            for f in factors[1:]:
                full_op = _jax_kron(full_op, f)

        self.full_ops[key] = full_op
        return full_op

    def _make_local_qubit_op(self, op_name: str, ss: SubsystemInfo) -> jnp.ndarray:
        r"""
        Build a local qubit operator.

        ``op_name`` ∈ {``"sx"``, ``"sy"``, ``"sz"``, ``"sp"``, ``"sm"``}
        and maps to :math:`\\sigma_x`, :math:`\\sigma_y`, :math:`\\sigma_z`,
        :math:`\\sigma_+`, :math:`\\sigma_-`.
        """
        dim = ss.dim
        if dim != 2:
            raise DSLValidationError(
                f"Qubit subsystem {ss.label}_{ss.index} must have dim=2, got {dim}."
            )

        zero = 0.0
        one = 1.0
        i = 1.0j

        if op_name == "sx":
            return jnp.array([[zero, one], [one, zero]], dtype=self.dtype)
        if op_name == "sy":
            return jnp.array([[zero, -i], [i, zero]], dtype=self.dtype)
        if op_name == "sz":
            return jnp.array([[one, zero], [zero, -one]], dtype=self.dtype)
        if op_name == "sp":
            return jnp.array([[zero, one], [zero, zero]], dtype=self.dtype)
        if op_name == "sm":
            return jnp.array([[zero, zero], [one, zero]], dtype=self.dtype)

        raise DSLValidationError(
            "Unsupported qubit operator "
            f"'{op_name}' for subsystem {ss.label}_{ss.index}."
        )

    def _make_local_boson_op(self, op_name: str, ss: SubsystemInfo) -> jnp.ndarray:
        r"""
        Build a local bosonic operator.

        ``op_name`` ∈ {``"a"``, ``"adag"``, ``"af"``, ``"adagf"``, ``"n"``}
        and maps to the annihilation operator :math:`a`, the creation
        operator :math:`a^{\\dagger}`, the number operator
        :math:`\\hat{n} = a^{\\dagger} a`, and the f-deformed combinations
        :math:`a f(\\hat{n})`, :math:`f(\\hat{n}) a^{\\dagger}`.
        """
        if not isinstance(ss.spec, BosonSpec):
            raise DSLValidationError(
                "Internal error: expected BosonSpec for boson subsystem "
                f"{ss.label}_{ss.index}."
            )

        dim = ss.dim

        # Base annihilation operator a
        idx = jnp.arange(1, dim, dtype=jnp.int32)
        data = jnp.sqrt(idx.astype(jnp.float64))
        a = jnp.zeros((dim, dim), dtype=self.dtype)
        a = a.at[idx - 1, idx].set(data.astype(self.dtype))

        def deformation_diag() -> jnp.ndarray:
            r"""
            Build diagonal JAX array containing :math:`f(n)` for ``n=0,...,dim-1``.
            """
            deform_fn = ss.spec.deformation
            if deform_fn is None:
                raise DSLValidationError(
                    f"Boson {ss.label}_{ss.index} has no deformation function, "
                    f"but deformed operator '{op_name}' was requested."
                )
            n_vals = jnp.arange(dim, dtype=jnp.float64)
            try:
                f_vals = deform_fn(n_vals)  # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover - defensive
                raise DSLValidationError(
                    "Deformation callable for "
                    f"{ss.label}_{ss.index} raised an exception: {exc}"
                ) from exc
            f_arr = jnp.asarray(f_vals)
            if f_arr.shape != (dim,):
                raise DSLValidationError(
                    f"Deformation callable for {ss.label}_{ss.index} must return "
                    f"shape ({dim},), got {f_arr.shape}."
                )
            return jnp.diag(f_arr.astype(self.dtype))

        if op_name == "a":
            return a
        if op_name == "adag":
            return a.conj().T
        if op_name == "af":
            diag_f = deformation_diag()
            return a @ diag_f
        if op_name == "adagf":
            diag_f = deformation_diag()
            return diag_f @ a.conj().T
        if op_name == "n":
            n_vals = jnp.arange(dim, dtype=jnp.float64)
            return jnp.diag(n_vals.astype(self.dtype))

        raise DSLValidationError(
            "Unsupported boson operator "
            f"'{op_name}' for subsystem {ss.label}_{ss.index}."
        )

    def _make_local_custom_op(self, op_name: str, ss: SubsystemInfo) -> jnp.ndarray:
        r"""
        Build a local operator for a custom subsystem.

        Operators must be array-like and convertible to a
        :math:`\\mathrm{dim} \\times \\mathrm{dim}` dense matrix.
        """
        if not isinstance(ss.spec, CustomSpec):
            raise DSLValidationError(
                "Internal error: expected CustomSpec for subsystem "
                f"{ss.label}_{ss.index}."
            )

        ops: Mapping[str, object] = ss.spec.operators
        if op_name not in ops:
            raise DSLValidationError(
                f"Custom subsystem {ss.label}_{ss.index} has no operator '{op_name}'."
            )

        op_val = ops[op_name]
        if hasattr(op_val, "full"):
            op_arr = op_val.full()  # type: ignore[attr-defined]
        else:
            op_arr = op_val

        arr = jnp.asarray(op_arr, dtype=self.dtype)
        if arr.shape != (ss.dim, ss.dim):
            raise DSLValidationError(
                f"Custom operator '{op_name}' for {ss.label}_{ss.index} must "
                f"have shape ({ss.dim}, {ss.dim}), got {arr.shape}."
            )
        return arr


def _param_aliases(name: str) -> List[str]:
    r"""Compatibility alias for shared param_aliases helper."""
    return param_aliases(name)


def _lookup_param_name(name: str, params: Dict[str, complex]) -> tuple[str, complex]:
    r"""Compatibility alias for shared lookup_param_name helper."""
    return lookup_param_name(name, params)


def _evaluate_scalar_static(
    expr: sp.Expr, params: Dict[str, complex], *, dtype: Any = DEFAULT_DTYPE
) -> jnp.ndarray:
    r"""
    Evaluate a time-independent scalar SymPy expression and return a JAX scalar.
    """
    # Fast path: single symbol mapped directly to a JAX value (keeps
    # AD-friendly tracer).
    if isinstance(expr, sp.Symbol) and expr.name in params:
        return jnp.asarray(params[expr.name], dtype=dtype)

    _require_jax()
    # Rewrite AppliedUndef heads that match params as implicit scalar multiplication.
    replacements = {}
    for f in expr.atoms(sp.Function):
        fname = f.func.__name__
        if fname in params:
            arg_prod = sp.Mul(*f.args) if f.args else 1
            replacements[f] = sp.Symbol(fname) * arg_prod
    if replacements:
        expr = expr.xreplace(replacements)

    subs_map: Dict[sp.Basic, complex] = {}
    for s in expr.atoms(sp.Symbol):
        name = s.name
        _, value = lookup_param_name(name, params)
        subs_map[s] = value

    # Also catch non-Symbol atoms whose string representation matches a param.
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

    val = expr_sub.evalf() if hasattr(expr_sub, "evalf") else expr_sub
    return jnp.asarray(val, dtype=dtype)


def _expr_has_time(expr: sp.Expr, time_names: set[str]) -> bool:
    r"""Return ``True`` if ``expr`` depends on any of the given time-like symbols."""
    return _expr_has_time_shared(expr, time_names)


@lru_cache(maxsize=256)
def _build_time_dep_term_callable_cached(
    scalar_expr: sp.Expr,
    t_name: str,
    time_symbols: tuple[str, ...],
    *,
    dtype: Any = DEFAULT_DTYPE,
) -> tuple[
    Callable[[float, Dict[str, Any]], jnp.ndarray],
    List[sp.Symbol],
    List[List[str]],
]:
    r"""
    Build a JAX-friendly envelope function f(t, args) for a scalar expression.
    """
    _require_jax()
    free_syms = list(scalar_expr.free_symbols)
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
    param_aliases_list = [param_aliases(s.name) for s in param_syms_sorted]

    if param_syms_sorted:
        lmbd = sp.lambdify((t_sym, *param_syms_sorted), scalar_expr, modules="jax")
    else:
        lmbd = sp.lambdify((t_sym,), scalar_expr, modules="jax")

    def f_k(t: float, args: Dict[str, Any]) -> jnp.ndarray:
        r"""
        Evaluate the time-dependent scalar envelope at time ``t``.
        """
        vals: List[float | complex | jnp.ndarray] = []
        for aliases in param_aliases_list:
            value = None
            for key in aliases:
                if isinstance(args, Mapping) and key in args:
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
        return jnp.asarray(val, dtype=dtype)

    return f_k, param_syms_sorted, param_aliases_list


def _build_time_dep_term_callable(
    scalar_expr: sp.Expr,
    t_name: str,
    time_symbols: tuple[str, ...] | None = None,
    *,
    dtype: Any = DEFAULT_DTYPE,
) -> tuple[
    Callable[[float, Dict[str, Any]], jnp.ndarray],
    List[sp.Symbol],
    List[List[str]],
]:
    r"""
    Wrapper that normalizes time symbols and uses a cached builder.
    """
    norm_time_symbols = tuple(time_symbols) if time_symbols else tuple()
    return _build_time_dep_term_callable_cached(
        scalar_expr, t_name, norm_time_symbols, dtype=dtype
    )


def _build_time_dep_operator_function_callable_jax(
    term: Term,
    cache: JaxOperatorCache,
    params: Dict[str, complex],
    t_name: str,
    time_symbols: tuple[str, ...] | None = None,
    *,
    dtype: Any = DEFAULT_DTYPE,
) -> Callable[[float, Dict[str, Any]], jnp.ndarray]:
    r"""
    Build a callable returning a full-system operator for terms where an
    operator function carries time dependence in its scalar factor.
    """
    _require_jax()
    time_names = {t_name} | (set(time_symbols) if time_symbols else set())

    if _expr_has_time(term.scalar_expr, time_names):
        scalar_fn, _, _ = _build_time_dep_term_callable(
            term.scalar_expr, t_name, time_symbols, dtype=dtype
        )
        scalar_const: jnp.ndarray | None = None
    else:
        scalar_const = _evaluate_scalar_static(term.scalar_expr, params, dtype=dtype)
        scalar_fn = None

    factors: List[tuple[str, Any]] = []
    for fac in term.ops:
        if isinstance(fac, LocalOperatorRef):
            op_mat = cache.full_operator(fac)
            factors.append(("static", op_mat))
            continue

        if isinstance(fac, OperatorFunctionRef):
            if _expr_has_time(fac.scalar_factor, time_names):
                scale_fn, _, _ = _build_time_dep_term_callable(
                    fac.scalar_factor, t_name, time_symbols, dtype=dtype
                )
                factors.append(("opfunc_time", fac.func_name, fac.arg, scale_fn))
            else:
                scale_val = _evaluate_scalar_static(
                    fac.scalar_factor, params, dtype=dtype
                )
                op_mat = _apply_operator_function_scaled_jax(
                    fac.func_name, fac.arg, scale_val, cache, dtype=dtype
                )
                factors.append(("opfunc_static", op_mat))
            continue

        raise DSLValidationError("Internal error: unknown operator factor type.")

    def _call(t: float, args: Dict[str, Any]) -> jnp.ndarray:
        r"""
        Evaluate the full operator product at time ``t`` with parameters ``args``.
        """
        op_acc: jnp.ndarray | None = None
        for item in factors:
            tag = item[0]
            if tag in {"static", "opfunc_static"}:
                op_mat = item[1]
            elif tag == "opfunc_time":
                _, func_name, arg_ref, scale_fn = item
                scale_val = scale_fn(t, args)
                op_mat = _apply_operator_function_scaled_jax(
                    func_name, arg_ref, scale_val, cache, dtype=dtype
                )
            else:
                raise DSLValidationError("Internal error: unknown factor tag.")
            op_acc = op_mat if op_acc is None else op_acc @ op_mat

        if op_acc is None:
            assert cache.global_identity is not None
            op_acc = cache.global_identity

        scalar_val = scalar_const if scalar_fn is None else scalar_fn(t, args)
        return jnp.asarray(scalar_val, dtype=dtype) * op_acc

    return _call


def _apply_operator_function_scaled_jax(
    func_name: str,
    arg_ref: OperatorFunctionRef | LocalOperatorRef,
    scale: complex,
    cache: JaxOperatorCache,
    *,
    dtype: Any = DEFAULT_DTYPE,
) -> jnp.ndarray:
    r"""
    Evaluate an operator-valued function :math:`f(O)` on ``scale * O``.

    Supported functions are those in ``ALLOWED_OPERATOR_FUNCTIONS``,
    currently ``{"exp", "cos", "sin"}``.
    """
    _require_jax()
    # Note: OperatorFunctionRef.arg is a LocalOperatorRef.
    if isinstance(arg_ref, LocalOperatorRef):
        op = cache.full_operator(arg_ref)
    elif isinstance(arg_ref, OperatorFunctionRef):
        # Nested functions are not expected at the IR level.
        op = cache.full_operator(arg_ref.arg)
    else:
        raise DSLValidationError(
            "Internal error: unsupported argument type in operator function."
        )

    mat = jnp.asarray(scale, dtype=dtype) * op

    fname = func_name
    if fname not in ALLOWED_OPERATOR_FUNCTIONS:
        raise DSLValidationError(
            f"Unsupported operator function '{fname}'. "
            f"Allowed: {sorted(ALLOWED_OPERATOR_FUNCTIONS)}."
        )

    try:
        return apply_operator_function(mat, fname, backend="jax")
    except Exception as exc:
        raise DSLValidationError(
            f"Unsupported operator function '{fname}'. "
            f"Allowed: {sorted(ALLOWED_OPERATOR_FUNCTIONS)}."
        ) from exc


def _term_to_jax_static(
    term: Term,
    cache: JaxOperatorCache,
    params: Dict[str, complex],
    *,
    dtype: Any = DEFAULT_DTYPE,
) -> jnp.ndarray:
    r"""
    Convert a :class:`Term` to a full-system JAX operator (static case).

    Operator-function scalar factors are evaluated using ``params``.
    """
    _require_jax()
    if not term.ops:
        assert cache.global_identity is not None
        return cache.global_identity

    op_acc: jnp.ndarray | None = None
    for fac in term.ops:
        if isinstance(fac, LocalOperatorRef):
            op_mat = cache.full_operator(fac)
        elif isinstance(fac, OperatorFunctionRef):
            scale = _evaluate_scalar_static(fac.scalar_factor, params, dtype=dtype)
            op_mat = _apply_operator_function_scaled_jax(
                fac.func_name, fac.arg, scale, cache, dtype=dtype
            )
        else:
            raise DSLValidationError(
                "Internal error: unknown operator factor type in Term."
            )

        op_acc = op_mat if op_acc is None else op_acc @ op_mat

    assert op_acc is not None
    return op_acc


def _compile_static_ir_to_jax(
    ir: HamiltonianIR,
    config: HilbertConfig,
    params: Dict[str, complex],
    cache: JaxOperatorCache | None = None,
    *,
    dtype: Any = DEFAULT_DTYPE,
) -> jnp.ndarray:
    r"""
    Compile a static :class:`HamiltonianIR` into a single JAX matrix.

    Each IR term is converted to a full-system operator via
    :func:`_term_to_jax_static`, multiplied by its scalar coefficient,
    and summed.
    """
    _require_jax()
    cache = cache or JaxOperatorCache(config, dtype=dtype)

    H: jnp.ndarray | None = None
    logger.debug(
        "Compiling static Hamiltonian to JAX: %d terms, subsystems=%s",
        len(ir.terms),
        cache.subsystems,
    )

    for term in ir.terms:
        coeff = _evaluate_scalar_static(term.scalar_expr, params, dtype=dtype)
        if jnp.all(coeff == 0):
            continue

        op = _term_to_jax_static(term, cache, params, dtype=dtype)
        coeff_arr = jnp.asarray(coeff, dtype=dtype)
        contrib = coeff_arr * op

        if H is None:
            H = contrib
        else:
            H = H + contrib

    if H is None:
        assert cache.global_identity is not None
        H = jnp.zeros_like(cache.global_identity)

    return H


@dataclass
class CompiledHamiltonianJax(CompiledHamiltonianBase):
    r"""
    Container for a compiled JAX Hamiltonian.

    Attributes
    ----------
    H : jax.numpy.ndarray or list
        Static Hamiltonian matrix for time-independent problems, or a
        list ``[H0, [H1, f1], ...]`` / ``[H0, fk_callable, ...]`` for
        time-dependent cases (mirroring the QuTiP backend layout).
    H0 : jax.numpy.ndarray
        Static part of the Hamiltonian.
    time_terms : list of tuple
        List of pairs ``(Hk, fk)`` where ``Hk`` is a JAX matrix (or
        ``None`` for operator-returning callables) and ``fk`` is a
        callable ``fk(t, args)`` producing either a scalar envelope or
        a full operator.
    args : dict
        Parameter dictionary used when evaluating scalar coefficients.
    time_dependent : bool
        ``True`` if any time dependence was detected.
    parameters : set[str]
        Names of parameters referenced by the Hamiltonian (excluding
        time symbols).
    """

    H: Any
    H0: jnp.ndarray
    time_terms: List[Tuple[Any, Callable[[float, Dict[str, Any]], Any]]]
    args: Dict[str, Any]
    time_dependent: bool
    parameters: set[str]


def compile_static_hamiltonian_ir(
    ir: HamiltonianIR,
    config: HilbertConfig,
    params: Dict[str, complex],
    cache: JaxOperatorCache | None = None,
    *,
    dtype: Any | None = None,
    platform: str | None = None,
    options: JaxBackendOptions | None = None,
) -> jnp.ndarray:
    r"""
    Compile a static :class:`HamiltonianIR` into a single JAX matrix.
    """
    opts = _merge_options(options, dtype=dtype, platform=platform)
    backend = JaxBackend()
    cache = cache or backend._make_cache(config, options=opts)
    return backend._compile_static(ir, cache, params, options=opts)


def compile_static_hamiltonian_from_latex(
    H_latex: str,
    config: HilbertConfig,
    params: Dict[str, complex],
    *,
    dtype: Any | None = None,
    platform: str | None = None,
    options: JaxBackendOptions | None = None,
) -> jnp.ndarray:
    r"""
    Compile a static Hamiltonian directly from LaTeX into a JAX matrix.
    """
    ir = latex_to_ir(H_latex, config, t_name="t")
    return compile_static_hamiltonian_ir(
        ir, config, params, dtype=dtype, platform=platform, options=options
    )


def compile_time_dependent_hamiltonian_ir(
    ir: HamiltonianIR,
    config: HilbertConfig,
    params: Dict[str, complex],
    t_name: str = "t",
    cache: JaxOperatorCache | None = None,
    time_symbols: tuple[str, ...] | None = None,
    *,
    dtype: Any | None = None,
    platform: str | None = None,
    options: JaxBackendOptions | None = None,
) -> CompiledHamiltonianJax:
    r"""
    Compile a (possibly) time-dependent :class:`HamiltonianIR` into JAX format.
    """
    opts = _merge_options(options, dtype=dtype, platform=platform)
    backend = JaxBackend()
    cache = cache or backend._make_cache(config, options=opts)
    # Drop t_name if IR is static to simplify downstream signatures.
    if not ir.has_time_dep:
        t_name = ""
    return backend._compile_time_dependent(
        ir,
        cache,
        params,
        t_name=t_name,
        time_symbols=time_symbols,
        options=opts,
    )


def compile_hamiltonian_from_latex(
    H_latex: str,
    config: HilbertConfig,
    params: Dict[str, complex],
    t_name: str = "t",
    *,
    dtype: Any | None = None,
    platform: str | None = None,
    options: JaxBackendOptions | None = None,
) -> CompiledHamiltonianJax:
    r"""
    Compile a Hamiltonian directly from LaTeX into JAX format.
    """
    ir = latex_to_ir(H_latex, config, t_name=t_name)
    return compile_time_dependent_hamiltonian_ir(
        ir,
        config,
        params,
        t_name=t_name,
        dtype=dtype,
        platform=platform,
        options=options,
    )


def _compile_collapse_ops_from_latex_jax(
    c_ops_latex: List[str],
    config: HilbertConfig,
    params: Dict[str, complex],
    t_name: str = "t",
    cache: JaxOperatorCache | None = None,
    time_symbols: tuple[str, ...] | None = None,
    *,
    dtype: Any | None = None,
    platform: str | None = None,
    options: JaxBackendOptions | None = None,
    param_names: set[str] | None = None,
) -> tuple[List[Any], Dict[str, Any], bool]:
    r"""
    Compile LaTeX collapse operators into JAX ``c_ops`` format.

    Static collapse channels return dense matrices; time-dependent
    channels are compiled as ``(C0, f(t, args))`` pairs, mirroring the
    QuTiP backend contract.
    """
    _require_jax()
    opts = _merge_options(options, dtype=dtype, platform=platform)
    _apply_platform(opts.platform)
    cache = cache or JaxOperatorCache(config, dtype=opts.dtype)
    c_ops: List[Any] = []
    any_time_dep = False
    logger.debug(
        "Compiling %d collapse ops, subsystems=%s", len(c_ops_latex), cache.subsystems
    )

    for c_latex in c_ops_latex:
        ir = latex_to_ir(c_latex, config, t_name=t_name, time_symbols=time_symbols)

        if not ir.terms:
            raise DSLValidationError(
                f"Collapse operator LaTeX {c_latex!r} contains no operator terms."
            )

        time_names = set(time_symbols) if time_symbols else set()
        time_names.add(t_name)
        if param_names is not None:
            param_names |= _collect_param_names(ir, time_names)
        has_t_any = any(
            _expr_has_time(term.scalar_expr, time_names)
            or any(
                isinstance(opf, OperatorFunctionRef)
                and _expr_has_time(opf.scalar_factor, time_names)
                for opf in term.ops
            )
            for term in ir.terms
        )

        if not has_t_any:
            # Purely static collapse operator; sum all terms.
            C: jnp.ndarray | None = None
            for term in ir.terms:
                if not term.ops:
                    raise DSLValidationError(
                        "Collapse operator "
                        f"{c_latex!r} has a term with no operator part."
                    )
                coeff = _evaluate_scalar_static(
                    term.scalar_expr, params, dtype=opts.dtype
                )
                if jnp.all(coeff == 0):
                    continue
                op_mat = _term_to_jax_static(term, cache, params, dtype=opts.dtype)
                contrib = jnp.asarray(coeff, dtype=opts.dtype) * op_mat
                C = contrib if C is None else C + contrib

            if C is None:
                continue

            c_ops.append(C)
        else:
            # Time-dependent collapse operator restricted to a single monomial.
            if len(ir.terms) != 1:
                raise DSLValidationError(ERROR_HINT_TIME_DEP_COLLAPSE)

            term = ir.terms[0]
            if not term.ops:
                raise DSLValidationError(
                    "Time-dependent collapse operator "
                    f"{c_latex!r} has no operator part."
                )

            scalar = term.scalar_expr

            # Skip channels that vanish at t=0 after substituting known params.
            subs_all: Dict[sp.Symbol, Any] = {}
            for s in list(scalar.free_symbols):
                if s.name in time_names:
                    subs_all[s] = 0
                    continue
                try:
                    _, val = lookup_param_name(s.name, params)
                except DSLValidationError:
                    continue
                subs_all[s] = val
            scalar_eval = sp.simplify(scalar.subs(subs_all))
            if scalar_eval.is_number and complex(scalar_eval) == 0:
                continue

            op_mat = _term_to_jax_static(term, cache, params, dtype=opts.dtype)
            f_k, _, _ = _build_time_dep_term_callable(
                scalar, t_name, time_symbols, dtype=opts.dtype
            )
            c_ops.append([op_mat, f_k])
            any_time_dep = True

    return c_ops, dict(params), any_time_dep


# Public alias preserved for compatibility with tests and callers.
compile_collapse_ops_from_latex = _compile_collapse_ops_from_latex_jax


@dataclass
class CompiledOpenSystemJax(CompiledOpenSystemBase):
    r"""
    Compiled open quantum system in JAX format.

    Attributes
    ----------
    H : jax.numpy.ndarray or list
        Static Hamiltonian matrix or time-dependent H-list as produced
        by :class:`CompiledHamiltonianJax`.
    c_ops : list
        Static collapse operators as matrices or time-dependent pairs
        ``[C0, f(t, args)]``.
    args : dict
        Parameter dictionary used when evaluating scalar coefficients.
    config : HilbertConfig
        Hilbert-space configuration used to construct operators.
    time_dependent : bool
        Flag indicating whether any time dependence is present in the
        Hamiltonian or in the collapse channels.
    parameters : set[str]
        Names of parameters referenced by Hamiltonian or collapse ops.
    """

    H: Any
    c_ops: List[Any]
    args: Dict[str, Any]
    config: HilbertConfig
    time_dependent: bool
    parameters: set[str]


class JaxBackend(BackendBase):
    r"""JAX backend for compiling IR to dense JAX arrays."""

    def _make_cache(
        self, config: HilbertConfig, options: BackendOptions | None
    ) -> JaxOperatorCache:
        r"""
        Build a JAX operator cache for the provided Hilbert configuration.
        """
        _require_jax()
        opts = (
            options if isinstance(options, JaxBackendOptions) else JaxBackendOptions()
        )
        _apply_platform(opts.platform)
        return JaxOperatorCache(config, dtype=opts.dtype)

    def _compile_static(
        self,
        ir: HamiltonianIR,
        cache: JaxOperatorCache,
        params: Dict[str, complex],
        options: BackendOptions | None = None,
    ):
        r"""
        Compile a static Hamiltonian IR into dense JAX matrices.
        """
        if ir.has_time_dep:
            raise DSLValidationError(
                "compile_static called on time-dependent IR; use time-dependent path."
            )
        opts = (
            options if isinstance(options, JaxBackendOptions) else JaxBackendOptions()
        )
        return _compile_static_ir_to_jax(
            ir, cache.config, params, cache=cache, dtype=opts.dtype
        )

    def _compile_time_dependent(
        self,
        ir: HamiltonianIR,
        cache: JaxOperatorCache,
        params: Dict[str, complex],
        *,
        t_name: str,
        time_symbols: tuple[str, ...] | None,
        options: BackendOptions | None = None,
    ):
        r"""
        Compile a time-dependent Hamiltonian IR into JAX time-dependent form.
        """
        opts = (
            options if isinstance(options, JaxBackendOptions) else JaxBackendOptions()
        )
        _apply_platform(opts.platform)

        time_names = {t_name} | (set(time_symbols) if time_symbols else set())
        param_names = _collect_param_names(ir, time_names)

        if not ir.has_time_dep:
            H_static = self._compile_static(ir, cache, params, options=opts)
            return CompiledHamiltonianJax(
                H=H_static,
                H0=H_static,
                time_terms=[],
                args=dict(params),
                time_dependent=False,
                parameters=param_names,
            )

        H0 = None
        time_terms: List[Tuple[Any, Any]] = []

        for term in ir.terms:
            scalar = term.scalar_expr
            has_t_scalar = _expr_has_time(scalar, time_names)
            has_t_opfunc = any(
                isinstance(opf, OperatorFunctionRef)
                and _expr_has_time(opf.scalar_factor, time_names)
                for opf in term.ops
            )

            if not has_t_scalar and not has_t_opfunc:
                coeff = _evaluate_scalar_static(scalar, params, dtype=opts.dtype)
                if jnp.all(coeff == 0):
                    continue
                op = _term_to_jax_static(term, cache, params, dtype=opts.dtype)
                contrib = jnp.asarray(coeff, dtype=opts.dtype) * op
                H0 = contrib if H0 is None else H0 + contrib
                continue

            if has_t_opfunc:
                term_callable = _build_time_dep_operator_function_callable_jax(
                    term, cache, params, t_name, time_symbols, dtype=opts.dtype
                )
                time_terms.append((None, term_callable))
            else:
                op = _term_to_jax_static(term, cache, params, dtype=opts.dtype)
                f_k, _, _ = _build_time_dep_term_callable(
                    scalar, t_name, time_symbols, dtype=opts.dtype
                )
                time_terms.append((op, f_k))

        if H0 is None:
            assert cache.global_identity is not None
            H0 = jnp.zeros_like(cache.global_identity)

        H_list: List[Any] = [H0]
        for Hk, fk in time_terms:
            if Hk is None:
                H_list.append(fk)
            else:
                H_list.append([Hk, fk])

        return CompiledHamiltonianJax(
            H=H_list,
            H0=H0,
            time_terms=time_terms,
            args=dict(params),
            time_dependent=True,
            parameters=param_names,
        )

    def compile_collapse_ops_from_latex(
        self,
        c_ops_latex: List[str],
        config: HilbertConfig,
        params: Dict[str, complex],
        t_name: str = "t",
        cache: JaxOperatorCache | None = None,
        time_symbols: tuple[str, ...] | None = None,
        options: BackendOptions | None = None,
        param_names: set[str] | None = None,
        dtype: Any | None = None,
        platform: str | None = None,
    ) -> tuple[List[Any], Dict[str, Any], bool]:
        r"""
        Compile collapse operators from LaTeX into JAX-friendly objects.
        """
        return _compile_collapse_ops_from_latex_jax(
            c_ops_latex,
            config,
            params,
            t_name=t_name,
            cache=cache,
            time_symbols=time_symbols,
            options=options,  # type: ignore[arg-type]
            param_names=param_names,
            dtype=dtype,
            platform=platform,
        )

    def compile_open_system_from_latex(
        self,
        H_latex: str,
        params: Dict[str, complex],
        *,
        config: HilbertConfig,
        c_ops_latex: List[str] | None = None,
        t_name: str = "t",
        time_symbols: tuple[str, ...] | None = None,
        options: JaxBackendOptions | None = None,
    ) -> CompiledOpenSystemJax:
        r"""
        Compile Hamiltonian and collapse operators from LaTeX into JAX objects.
        """
        opts = options or JaxBackendOptions()
        cache = self._make_cache(config, options=opts)
        time_names = {t_name} | (set(time_symbols) if time_symbols else set())
        H_ir = latex_to_ir(H_latex, config, t_name=t_name, time_symbols=time_symbols)
        param_names = _collect_param_names(H_ir, time_names)
        H_compiled = self._compile_time_dependent(
            H_ir, cache, params, t_name=t_name, time_symbols=time_symbols, options=opts
        )

        if c_ops_latex:
            param_names_acc = set(param_names)
            c_ops, _, c_td = self.compile_collapse_ops_from_latex(
                c_ops_latex,
                config,
                params,
                t_name=t_name,
                cache=cache,
                time_symbols=time_symbols,
                dtype=opts.dtype,
                platform=opts.platform,
                options=opts,
                param_names=param_names_acc,
            )
            param_names |= param_names_acc
        else:
            c_ops, _, c_td = [], dict(params), False

        args = dict(params)
        return CompiledOpenSystemJax(
            H=H_compiled.H,
            c_ops=c_ops,
            args=args,
            config=config,
            time_dependent=(H_compiled.time_dependent or c_td),
            parameters=param_names,
        )


def compile_open_system_from_latex(
    H_latex: str,
    params: Dict[str, complex],
    *,
    config: HilbertConfig,
    c_ops_latex: List[str] | None = None,
    t_name: str = "t",
    time_symbols: tuple[str, ...] | None = None,
    dtype: Any | None = None,
    platform: str | None = None,
    options: JaxBackendOptions | None = None,
) -> CompiledOpenSystemJax:
    r"""
    Compile an open quantum system from LaTeX into JAX format.
    """
    opts = _merge_options(options, dtype=dtype, platform=platform)
    backend = JaxBackend()
    return backend.compile_open_system_from_latex(
        H_latex=H_latex,
        params=params,
        config=config,
        c_ops_latex=c_ops_latex or [],
        t_name=t_name,
        time_symbols=time_symbols,
        options=opts,
    )
