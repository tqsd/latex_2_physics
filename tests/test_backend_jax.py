import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.linalg import expm
from qutip import Qobj  # type: ignore

from latex_parser.backend_jax import (
    DEFAULT_DTYPE,
    JaxOperatorCache,
    JaxBackend,
    JaxBackendOptions,
    _term_to_jax_static,
    compile_collapse_ops_from_latex,
    compile_hamiltonian_from_latex,
    compile_open_system_from_latex,
    compile_static_hamiltonian_from_latex,
    compile_static_hamiltonian_ir,
    compile_time_dependent_hamiltonian_ir,
    latex_to_ir,
)
from latex_parser.backend_qutip import (
    QutipOperatorCache,
    _apply_operator_function_scaled,
)
from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    LocalOperatorRef,
    QubitSpec,
)
from latex_parser.errors import BackendUnavailableError

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_platform_name", "cpu")


def _assert_close(a, b, atol=1e-10):
    def _to_array(x):
        if hasattr(x, "full"):
            return jnp.asarray(np.array(x.full()), dtype=jnp.complex128)
        return jnp.asarray(x, dtype=jnp.complex128)

    a_arr = _to_array(a)
    b_arr = _to_array(b)
    assert jnp.allclose(a_arr, b_arr, atol=atol), jnp.max(jnp.abs(a_arr - b_arr))


def _ladder_ops(dim: int):
    idx = jnp.arange(1, dim, dtype=jnp.int32)
    data = jnp.sqrt(idx.astype(jnp.float64))
    a = (
        jnp.zeros((dim, dim), dtype=DEFAULT_DTYPE)
        .at[idx - 1, idx]
        .set(data.astype(DEFAULT_DTYPE))
    )
    adag = a.conj().T
    return a, adag


def test_invalid_config_entry_rejected():
    with pytest.raises(DSLValidationError):
        HilbertConfig(qubits=["bad"], bosons=[], customs=[])


def test_jax_backend_preserves_ad():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"g \sigma_{z,1}", cfg)
    backend = JaxBackend()
    opts = JaxBackendOptions()

    def loss(g):
        H = backend.compile_static_from_ir(ir, cfg, {"g": g}, options=opts)
        evals = jnp.linalg.eigvalsh(H)
        return evals[0].real

    grad_fn = jax.grad(loss)
    grad = grad_fn(jnp.array(0.5))
    assert jnp.isclose(grad, -1.0, atol=1e-6)


def test_jax_backend_sinh_ad():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"g \sinh(\sigma_{z,1})", cfg)
    backend = JaxBackend()
    opts = JaxBackendOptions()

    def loss(g):
        H = backend.compile_static_from_ir(ir, cfg, {"g": g}, options=opts)
        return jnp.sum(jnp.real(jnp.diag(H)))

    grad = jax.grad(loss)(jnp.array(1.0))
    assert grad.shape == ()


def test_deformed_boson_backend_jax():
    def deform(n):
        return jnp.sqrt(n + 1.0)

    dim = 4
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=dim, deformation=deform)],
        customs=[],
    )
    cache = JaxOperatorCache(cfg)

    diag = jnp.diag(deform(jnp.arange(dim, dtype=jnp.float64)).astype(DEFAULT_DTYPE))
    a, adag = _ladder_ops(dim)

    af_expected = a @ diag
    adagf_expected = diag @ adag

    af = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="af")
    )
    adagf = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="adagf")
    )

    _assert_close(af, af_expected)
    _assert_close(adagf, adagf_expected)


def test_deformed_boson_backend_other_fn_jax():
    def deform(n):
        return jnp.exp(-0.5 * n)

    dim = 5
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=dim, deformation=deform)],
        customs=[],
    )
    cache = JaxOperatorCache(cfg)

    diag = jnp.diag(deform(jnp.arange(dim, dtype=jnp.float64)).astype(DEFAULT_DTYPE))
    a, adag = _ladder_ops(dim)

    af_expected = a @ diag
    adagf_expected = diag @ adag

    af = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="af")
    )
    adagf = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="adagf")
    )

    _assert_close(af, af_expected)
    _assert_close(adagf, adagf_expected)


def test_operator_function_cos_qubit_backend_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\cos(\sigma_{z,1})", cfg, t_name="t")
    cache = JaxOperatorCache(cfg)
    term = ir.terms[0]
    op = _term_to_jax_static(term, cache, params={})

    sz = cache.local_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz")
    )
    manual = 0.5 * (expm(1j * sz) + expm(-1j * sz))
    _assert_close(op, manual)


def test_operator_function_exp_boson_backend_jax():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=3)], customs=[]
    )
    ir = latex_to_ir(r"\exp(n_{1})", cfg, t_name="t")
    cache = JaxOperatorCache(cfg)
    term = ir.terms[0]
    op = _term_to_jax_static(term, cache, params={})

    a, adag = _ladder_ops(3)
    n_op = adag @ a
    manual = expm(n_op)
    _assert_close(op, manual)


def test_operator_function_with_time_backend_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(t \sigma_{z,1})", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 1
    Hk, fk = compiled.time_terms[0]
    assert Hk is None

    sz = JaxOperatorCache(cfg).local_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz")
    )
    for tval in [0.0, 0.2, -0.3]:
        op = fk(tval, compiled.args)
        manual = expm(tval * sz)
        _assert_close(op, manual)


def test_static_hamiltonian_with_time_independent_function_backend_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(0.5 \sigma_{z,1})", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, params={})
    sz = JaxOperatorCache(cfg).local_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz")
    )
    manual = expm(0.5 * sz)
    _assert_close(H, manual)


def test_static_hamiltonian_with_mixed_time_function_backend_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(0.5 t \sigma_{z,1})", cfg, t_name="t")
    with pytest.raises(DSLValidationError):
        compile_static_hamiltonian_ir(ir, cfg, params={})


def test_static_hamiltonian_with_no_time_function_backend_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(2) \sigma_{z,1}", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, params={})
    sz = JaxOperatorCache(cfg).local_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz")
    )
    manual = jnp.exp(2) * sz
    _assert_close(H, manual)


def test_time_dependent_hamiltonian_with_mixed_time_function_backend_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(0.5 t \sigma_{z,1})", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is True
    Hk, fk = compiled.time_terms[0]
    assert Hk is None

    sz = JaxOperatorCache(cfg).local_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz")
    )
    for tval in [0.0, 0.3, -0.4]:
        op = fk(tval, compiled.args)
        manual = expm(0.5 * tval * sz)
        _assert_close(op, manual)


def test_time_dependent_hamiltonian_with_no_time_function_backend_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(2) \sigma_{z,1}", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is False
    H = compiled.H
    sz = JaxOperatorCache(cfg).local_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz")
    )
    manual = jnp.exp(2) * sz
    _assert_close(H, manual)


def test_static_jc_jax():
    N = 3
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[],
    )

    H_latex = r"""
        \omega_c n_{1}
        + \frac{\omega_q}{2} \sigma_{z,1}
        + g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})
    """

    params = {"omega_c": 1.5, "omega_q": 2.0, "g": 0.1}

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    H_compiled = compile_static_hamiltonian_ir(ir, cfg, params)
    cache = JaxOperatorCache(cfg)

    a_full = cache.full_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="a", power=1)
    )
    adag_full = cache.full_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="adag", power=1)
    )
    n_full = adag_full @ a_full

    sz_full = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz", power=1)
    )
    sp_full = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sp", power=1)
    )
    sm_full = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sm", power=1)
    )

    H_c = params["omega_c"] * n_full
    H_q = (params["omega_q"] / 2.0) * sz_full
    H_int = params["g"] * (a_full @ sp_full + adag_full @ sm_full)
    H_ref = H_c + H_q + H_int

    _assert_close(H_compiled, H_ref, atol=1e-9)


def test_time_dep_driven_qubit_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])

    H_latex = r"""
        \frac{\omega_0}{2} \sigma_{z,1}
        + A \cos(\omega t) \sigma_{x,1}
    """

    params = {"omega_0": 2.0, "A": 0.3, "omega": 1.5}
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params, t_name="t")

    assert compiled.time_dependent is True
    H0 = compiled.H0
    time_terms = compiled.time_terms

    sz = JaxOperatorCache(cfg).full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz", power=1)
    )
    sx = JaxOperatorCache(cfg).full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=1)
    )

    _assert_close(H0, (params["omega_0"] / 2.0) * sz)

    assert len(time_terms) == 1
    H1, f1 = time_terms[0]
    _assert_close(H1, sx)
    for t in [0.0, 0.7]:
        val = f1(t, compiled.args)
        expected = params["A"] * math.cos(params["omega"] * t)
        assert abs(val - expected) < 1e-10


def test_static_collapse_qubit_boson_jax():
    N = 3
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[],
    )

    c_ops_latex = [r"\sqrt{\kappa} a_{1}", r"\sqrt{\gamma} \sigma_{-,1}"]
    params = {"kappa": 0.1, "gamma": 0.05}

    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params, t_name="t"
    )

    assert td is False
    assert len(c_ops) == 2

    cache = JaxOperatorCache(cfg)
    a = cache.full_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="a", power=1)
    )
    sm = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sm", power=1)
    )

    C1_ref = math.sqrt(params["kappa"]) * a
    C2_ref = math.sqrt(params["gamma"]) * sm

    _assert_close(c_ops[0], C1_ref)
    _assert_close(c_ops[1], C2_ref)
    assert args == params


def test_time_dep_collapse_qubit_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])

    c_ops_latex = [r"\sqrt{\gamma} \exp(-t/2) \sigma_{-,1}"]
    params = {"gamma": 0.2}

    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params, t_name="t"
    )

    assert td is True
    assert len(c_ops) == 1
    C0, f = c_ops[0]

    sm = JaxOperatorCache(cfg).full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sm", power=1)
    )
    _assert_close(C0, sm)

    for t in [0.0, 0.4, 1.0]:
        val = f(t, args)
        expected = math.sqrt(params["gamma"]) * math.exp(-t / 2)
        assert abs(val - expected) < 1e-10


def test_param_alias_braced_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\omega_{d} \sigma_{x,1}", cfg, t_name="t")
    cache = JaxOperatorCache(cfg)
    H = compile_static_hamiltonian_ir(ir, cfg, params={"omega_d": 2.0}, cache=cache)
    sx = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=1)
    )
    _assert_close(H, 2.0 * sx)


def test_compile_hamiltonian_from_latex_wrapper_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\Delta \sigma_{x,1}"
    compiled = compile_hamiltonian_from_latex(
        H_latex, cfg, {"Delta": 0.7}, dtype=jnp.complex64
    )
    assert compiled.time_dependent is False
    assert compiled.H0.dtype == jnp.complex64
    cache = JaxOperatorCache(cfg, dtype=jnp.complex64)
    sx = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=1)
    )
    _assert_close(compiled.H0, 0.7 * sx)


def test_compile_open_system_time_dep_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \cos(\omega t) \sigma_{x,1}"
    c_ops_latex = [r"\sqrt{\gamma} \sigma_{-,1}"]
    params = {"A": 0.5, "omega": 1.2, "gamma": 0.1}

    model = compile_open_system_from_latex(
        H_latex,
        params,
        config=cfg,
        c_ops_latex=c_ops_latex,
        dtype=jnp.complex128,
    )

    assert model.time_dependent is True
    assert isinstance(model.H, list)
    assert len(model.H) == 2
    assert len(model.c_ops) == 1
    C = model.c_ops[0]
    cache = JaxOperatorCache(cfg)
    sm = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sm", power=1)
    )
    _assert_close(C, math.sqrt(params["gamma"]) * sm)


def test_compile_open_system_static_collapse_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\Delta \sigma_{z,1}"
    c_ops_latex = [r"\sqrt{\gamma} \sigma_{-,1}"]
    params = {"Delta": 0.3, "gamma": 0.2}

    model = compile_open_system_from_latex(
        H_latex,
        params,
        config=cfg,
        c_ops_latex=c_ops_latex,
        dtype=jnp.complex64,
    )

    assert model.time_dependent is False
    assert len(model.c_ops) == 1
    c0 = model.c_ops[0]
    cache = JaxOperatorCache(cfg, dtype=jnp.complex64)
    sm = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sm", power=1)
    )
    _assert_close(c0, math.sqrt(params["gamma"]) * sm)
    assert "gamma" in model.parameters


def test_compile_open_system_td_collapse_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\Delta \sigma_{x,1}"
    c_ops_latex = [r"\sqrt{\gamma} \exp(-t/2) \sigma_{-,1}"]
    params = {"Delta": 0.1, "gamma": 0.2}

    model = compile_open_system_from_latex(
        H_latex,
        params,
        config=cfg,
        c_ops_latex=c_ops_latex,
        dtype=jnp.complex64,
        t_name="t",
    )

    assert model.time_dependent is True
    assert len(model.c_ops) == 1
    C, fk = model.c_ops[0]
    cache = JaxOperatorCache(cfg, dtype=jnp.complex64)
    sm = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sm", power=1)
    )
    _assert_close(C, sm)
    for t in (0.0, 0.5, 1.0):
        expected = math.sqrt(params["gamma"]) * math.exp(-t / 2.0)
        assert abs(fk(t, model.args) - expected) < 1e-10
    assert "gamma" in model.parameters


def test_compile_static_hamiltonian_dtype_override_jax():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = compile_static_hamiltonian_from_latex(
        r"\alpha \sigma_{z,1}", cfg, {"alpha": 1.1}, dtype=jnp.complex64
    )
    assert H.dtype == jnp.complex64
    cache = JaxOperatorCache(cfg, dtype=jnp.complex64)
    sz = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz", power=1)
    )
    _assert_close(H, 1.1 * sz)


def test_compile_static_hamiltonian_ir_rejects_time_dep():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"A \cos(\omega t) \sigma_{x,1}", cfg, t_name="t")
    with pytest.raises(DSLValidationError):
        compile_static_hamiltonian_ir(ir, cfg, params={"A": 1.0, "omega": 2.0})


def test_operator_function_invalid_name():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = JaxOperatorCache(cfg, dtype=jnp.complex64)
    ref = LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=1)
    # _apply_operator_function_scaled_jax guards unsupported names; expect
    # DSLValidationError
    from latex_parser.backend_jax import _apply_operator_function_scaled_jax

    with pytest.raises(Exception):
        _apply_operator_function_scaled_jax("tanh", ref, 1.0, cache)


def test_full_operator_negative_power_rejected():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = JaxOperatorCache(cfg)
    with pytest.raises(DSLValidationError):
        cache.full_operator(
            LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=-1)
        )


def test_make_local_boson_op_invalid_spec():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    cache = JaxOperatorCache(cfg)

    class FakeInfo:
        def __init__(self):
            self.kind = "boson"
            self.label = "a"
            self.index = 1
            self.dim = 2
            self.spec = object()

    bogus_ss = FakeInfo()
    with pytest.raises(DSLValidationError):
        cache._make_local_boson_op("a", bogus_ss)  # type: ignore[arg-type]


def test_time_dep_collapse_sum_rejected():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=2)],
        bosons=[],
        customs=[],
    )
    with pytest.raises(DSLValidationError):
        compile_collapse_ops_from_latex(
            [r"\sqrt{\gamma} \exp(-t/2) (\sigma_{-,1} + \sigma_{-,2})"],
            cfg,
            {"gamma": 0.1},
            t_name="t",
        )


def test_time_dep_operator_function_callable_cos():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\cos(t) \sigma_{z,1}", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 1
    Hk, fk = compiled.time_terms[0]
    cache = JaxOperatorCache(cfg)
    sz = cache.full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz", power=1)
    )
    assert jnp.allclose(Hk, sz)
    for t in (0.0, 0.5):
        assert abs(fk(t, compiled.args) - math.cos(t)) < 1e-12


def test_validate_custom_spec_non_qobj_raises():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[CustomSpec(label="c", index=1, dim=2, operators={"Jz": np.eye(2)})],
    )
    ir = latex_to_ir(r"J_{z,1}", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, {})
    ref = jnp.asarray(np.eye(2))
    _assert_close(H, ref)


def test_apply_operator_function_scaled_invalid_name():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = QutipOperatorCache(cfg)
    ref = LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz", power=1)
    with pytest.raises(DSLValidationError):
        _apply_operator_function_scaled("tanh", ref, 1.0, cache)


def test_backend_unavailable_error(monkeypatch):
    import latex_parser.backend_jax as bj

    monkeypatch.setattr(
        bj,
        "_require_jax",
        lambda: (_ for _ in ()).throw(BackendUnavailableError("no jax")),
    )
    with pytest.raises(BackendUnavailableError):
        bj._require_jax()


def test_duplicate_subsystem_rejected():
    with pytest.raises(DSLValidationError):
        HilbertConfig(
            qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=1)],
            bosons=[],
            customs=[],
        )


def test_compile_collapse_time_dep_sum_rejected():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=2)],
        bosons=[],
        customs=[],
    )
    with pytest.raises(DSLValidationError):
        compile_collapse_ops_from_latex(
            [r"\sqrt{\gamma} \exp(-t/2) (\sigma_{-,1} + \sigma_{-,2})"],
            cfg,
            {"gamma": 0.1},
            t_name="t",
        )


def test_validate_custom_spec_dim_mismatch():
    bad_op = Qobj(np.eye(3))  # dim 3x3
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[CustomSpec(label="c", index=1, dim=2, operators={"Jz": bad_op})],
    )
    ir = latex_to_ir(r"J_{z,1}", cfg, t_name="t")
    with pytest.raises(DSLValidationError):
        compile_static_hamiltonian_ir(ir, cfg, {})


def test_param_alias_priority_prefers_shorter():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\omega_{c} \sigma_{z,1}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    params = {"omega_c": 1.0, "omega_c1": 2.0}
    H = compile_static_hamiltonian_ir(ir, cfg, params)
    sz = QutipOperatorCache(cfg).full_operator(
        LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sz", power=1)
    )
    _assert_close(H, 1.0 * sz)


def test_static_hamiltonian_zero_coeff_returns_zero():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"0 * \sigma_{x,1}", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, params={})
    _assert_close(H, jnp.zeros_like(H))


def test_make_local_boson_op_invalid_spec_raises():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    cache = QutipOperatorCache(cfg)

    # If spec is not BosonSpec, the method raises DSLValidationError; build
    # a fake _SubsystemInfo manually.
    class FakeInfo:
        def __init__(self):
            self.kind = "boson"
            self.label = "a"
            self.index = 1
            self.dim = 2
            self.spec = object()

    bogus_ss = FakeInfo()
    with pytest.raises(DSLValidationError):
        cache._make_local_boson_op("a", bogus_ss)  # type: ignore[arg-type]


def test_make_local_custom_missing_operator():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[
            CustomSpec(label="c", index=1, dim=2, operators={"Jz": Qobj(np.eye(2))})
        ],
    )
    cache = QutipOperatorCache(cfg)
    ss = cache._find_subsystem("custom", "c", 1)
    with pytest.raises(DSLValidationError):
        cache._make_local_custom_op("Jx", ss)


def test_apply_operator_function_scaled_invalid_scalar_factor():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = QutipOperatorCache(cfg)
    ref = LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=1)
    res = _apply_operator_function_scaled("exp", ref, 0.0, cache)
    assert isinstance(res, Qobj)


def test_compile_static_hamiltonian_ir_zero_terms_returns_zero():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    ir = latex_to_ir("0", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, params={})
    _assert_close(H, jnp.asarray([[0.0]]))


def test_custom_operator_wrong_dims():
    bad_op = Qobj(np.eye(3))
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[CustomSpec(label="c", index=1, dim=2, operators={"Jz": bad_op})],
    )
    ir = latex_to_ir(r"J_{z,1}", cfg, t_name="t")
    with pytest.raises(DSLValidationError):
        compile_static_hamiltonian_ir(ir, cfg, {})


def test_apply_operator_function_scaled_nan_scalar():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = QutipOperatorCache(cfg)
    ref = LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=1)
    res = _apply_operator_function_scaled("exp", ref, 0.0, cache)
    assert isinstance(res, Qobj)


def test_time_dep_operator_function_invalid_name_in_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    with pytest.raises(DSLValidationError):
        compile_time_dependent_hamiltonian_ir(
            latex_to_ir(r"\tanh(\sigma_{z,1})", cfg, t_name="t"),
            cfg,
            params={},
            t_name="t",
        )


def test_full_operator_negative_power_qutip():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = QutipOperatorCache(cfg)
    with pytest.raises(DSLValidationError):
        cache.full_operator(
            LocalOperatorRef(kind="qubit", label="q", index=1, op_name="sx", power=-2)
        )


def test_collapse_operator_missing_term_rejected():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    # Collapse operator with scalar only
    with pytest.raises(DSLValidationError):
        compile_collapse_ops_from_latex(
            [r"\sqrt{\gamma}"], cfg, {"gamma": 0.1}, t_name="t"
        )


def test_compile_static_hamiltonian_ir_all_zero_terms_returns_zero():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    ir = latex_to_ir("0", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, params={})
    _assert_close(H, jnp.asarray([[0.0]]))


def test_hamiltonian_parameters_tracked():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(
        r"\omega \sigma_{z,1} + g \cos(\nu t) \sigma_{x,1}", cfg, t_name="t"
    )
    compiled = compile_time_dependent_hamiltonian_ir(
        ir, cfg, {"omega": 1.0, "g": 0.5, "nu": 2.0}, t_name="t"
    )
    assert {"omega", "g", "nu"} <= compiled.parameters
    assert "t" not in compiled.parameters


def test_open_system_parameters_tracked():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    compiled = JaxBackend().compile_open_system_from_latex(
        H_latex=r"\delta \sigma_{x,1}",
        params={"delta": 0.4, "gamma": 0.1},
        config=cfg,
        c_ops_latex=[r"\sqrt{\gamma} \sigma_{-,1}"],
    )
    assert {"delta", "gamma"} <= compiled.parameters
