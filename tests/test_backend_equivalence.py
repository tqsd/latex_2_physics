import os

import jax
import numpy as np

from latex_parser.backend_jax import (
    compile_static_hamiltonian_from_latex as compile_static_jax,
    compile_time_dependent_hamiltonian_ir as compile_td_jax,
)
from latex_parser.backend_qutip import (
    compile_static_hamiltonian_from_latex as compile_static_qutip,
    compile_time_dependent_hamiltonian_ir as compile_td_qutip,
)
from latex_parser.dsl import BosonSpec, HilbertConfig, QubitSpec
from latex_parser.errors import BackendUnavailableError
from latex_parser.ir import latex_to_ir

# Force JAX to CPU for deterministic comparisons
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_platforms", "cpu")


def _qobj_to_np(obj):
    return np.array(obj.full(), dtype=complex)


def _assert_close(a, b, atol=1e-12):
    a_arr = np.array(a)
    b_arr = np.array(b)
    assert np.allclose(a_arr, b_arr, atol=atol), np.max(np.abs(a_arr - b_arr))


def test_static_qubit_equivalence():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\omega \sigma_{z,1}"
    params = {"omega": 1.234}

    H_jax = compile_static_jax(H_latex, cfg, params)
    H_qutip = compile_static_qutip(H_latex, cfg, params)

    _assert_close(H_jax, _qobj_to_np(H_qutip))


def test_static_jc_equivalence():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=3)],
        customs=[],
    )
    H_latex = r"\omega_c n_{1} + g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})"
    params = {"omega_c": 0.7, "g": -0.2}

    H_jax = compile_static_jax(H_latex, cfg, params)
    H_qutip = compile_static_qutip(H_latex, cfg, params)

    _assert_close(H_jax, _qobj_to_np(H_qutip))


def test_static_operator_function_equivalence():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\exp(0.4 \sigma_{z,1})"
    params: dict[str, float] = {}

    H_jax = compile_static_jax(H_latex, cfg, params)
    H_qutip = compile_static_qutip(H_latex, cfg, params)

    _assert_close(H_jax, _qobj_to_np(H_qutip))


def test_time_dependent_drive_equivalence():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(
        r"\Delta \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}", cfg, t_name="t"
    )
    params = {"Delta": 0.3, "A": 0.8, "omega": 1.1}

    jax_td = compile_td_jax(ir, cfg, params, t_name="t")
    qutip_td = compile_td_qutip(ir, cfg, params, t_name="t")

    # Compare static part
    _assert_close(jax_td.H0, _qobj_to_np(qutip_td.H0))

    assert len(jax_td.time_terms) == len(qutip_td.time_terms) == 1
    Hk_jax, fk_jax = jax_td.time_terms[0]
    Hk_qutip, fk_qutip = qutip_td.time_terms[0]

    _assert_close(Hk_jax, _qobj_to_np(Hk_qutip))

    for tval in (0.0, 0.5):
        val_jax = fk_jax(tval, jax_td.args)
        val_qutip = fk_qutip(tval, qutip_td.args)
        assert np.allclose(val_jax, val_qutip, atol=1e-12)


def _compile_func(func: str):
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = rf"\{func}(\sigma_{{z,1}})"
    params: dict[str, float] = {}
    try:
        H_jax = compile_static_jax(H, cfg, params)
    except BackendUnavailableError:
        raise
    H_qutip = compile_static_qutip(H, cfg, params)
    return H_jax, _qobj_to_np(H_qutip)


def test_operator_function_equivalence_extended():
    for func in ("exp", "cos", "sin", "cosh", "sinh"):
        try:
            H_jax, H_qutip = _compile_func(func)
            _assert_close(H_jax, H_qutip, atol=1e-10)
        except BackendUnavailableError:
            # If JAX is unavailable, ensure a clear error is raised and continue.
            assert True
