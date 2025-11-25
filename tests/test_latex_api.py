import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from qutip import Qobj, sigmam  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Ensure we import the local package, not any preinstalled version.
for mod in list(sys.modules):
    if mod.startswith("latex_parser"):
        del sys.modules[mod]

import latex_parser.latex_api as api  # noqa: E402


def test_make_config_two_bosons():
    cfg = api.make_config(qubits=[1], bosons=[5, 7])
    subs = cfg.all_subsystems()
    kinds_labels_indices = [(kind, spec.label, spec.index) for kind, spec in subs]
    assert ("qubit", "q", 1) in kinds_labels_indices
    assert ("boson", "a", 1) in kinds_labels_indices
    assert ("boson", "a", 2) in kinds_labels_indices


def test_compile_model_static_qubit():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    params = {"omega_0": 2.0}
    model = api.compile_model(
        H_latex=H_latex,
        params=params,
        qubits=[1],
        bosons=[],
        customs=[],
        c_ops_latex=None,
        t_name="t",
    )
    assert isinstance(model.H, Qobj)
    assert not model.time_dependent
    assert model.c_ops == [] or model.c_ops is None


def test_compile_model_static_collapse():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    c_ops_latex = [r"\sqrt{\gamma} \sigma_{-,1}"]
    params = {"omega_0": 2.0, "gamma": 0.3}
    model = api.compile_model(
        H_latex=H_latex,
        params=params,
        qubits=[1],
        bosons=[],
        customs=[],
        c_ops_latex=c_ops_latex,
        t_name="t",
    )
    assert isinstance(model.H, Qobj)
    assert len(model.c_ops) == 1
    C0 = model.c_ops[0]
    diff = (C0 - (params["gamma"] ** 0.5) * sigmam()).norm()
    assert diff < 1e-10
    assert not model.time_dependent


def test_make_config_deformation_latex():
    cfg = api.make_config(bosons=[(4, "a", r"\sqrt{n}")])
    boson = cfg.bosons[0]
    vals = boson.deformation([0, 1, 2, 3])
    assert np.allclose(vals, np.sqrt([0, 1, 2, 3]))
    assert boson.deformation_latex is not None


def test_make_config_deformation_complex():
    cfg = api.make_config(bosons=[(3, "a", r"\exp(i n)")])
    boson = cfg.bosons[0]
    vals = boson.deformation([0, 1, 2])
    # Expect a complex phase sequence
    assert np.allclose(vals, np.exp(1j * np.array([0, 1, 2])))


def test_compile_model_two_time_symbols():
    cfg = api.make_config(qubits=[1], bosons=[])
    H_latex = r"A \cos(\omega t) \sigma_{x,1} + B \sin(s) \sigma_{x,1}"
    params = {"A": 0.5, "omega": 1.0, "B": 0.2}
    model = api.compile_model(
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=None,
        t_name="t",
        time_symbols=("t", "s"),
    )
    assert model.time_dependent


def test_make_config_deformation_latex_invalid():
    with pytest.raises(api.DSLValidationError):
        api.make_config(bosons=[(4, "a", r"\sqrt{m}")])


def test_compile_model_unsupported_operator_base():
    H_latex = r"\omega_b \, b_{1}"
    params = {"omega_b": 1.0}
    with pytest.raises(api.DSLValidationError):
        api.compile_model(
            H_latex=H_latex,
            params=params,
            qubits=[],
            bosons=[5],
            customs=[],
            c_ops_latex=None,
            t_name="t",
        )


def test_compile_model_time_dependent():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega_d t) \sigma_{x,1}"
    params = {"omega_0": 2.0, "A": 0.5, "omega_d": 1.0}
    model = api.compile_model(
        H_latex=H_latex,
        params=params,
        qubits=[1],
        bosons=[],
        customs=[],
        c_ops_latex=None,
        t_name="t",
    )
    assert model.time_dependent
    assert isinstance(model.H, list)
    assert len(model.H) == 2
    assert isinstance(model.H[0], Qobj)
    assert isinstance(model.H[1], list)
    assert isinstance(model.H[1][0], Qobj)


def test_compile_model_time_dependent_collapse():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    c_ops_latex = [r"\sqrt{\gamma(t)} \sigma_{-,1}"]
    params = {"omega_0": 2.0, "gamma": 0.3}
    model = api.compile_model(
        H_latex=H_latex,
        params=params,
        qubits=[1],
        bosons=[],
        customs=[],
        c_ops_latex=c_ops_latex,
        t_name="t",
    )
    assert isinstance(model.H, Qobj)
    assert len(model.c_ops) == 1
    C0 = model.c_ops[0]
    assert isinstance(C0, list)
    assert len(C0) == 2
    assert isinstance(C0[0], Qobj)
    # time-dependent envelope callable
    assert callable(C0[1])
    assert model.time_dependent


def test_compile_model_invalid_operator():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + \omega_b b_{1}^\dagger b_{1}"
    params = {"omega_0": 2.0, "omega_b": 1.0}
    with pytest.raises(api.DSLValidationError):
        api.compile_model(
            H_latex=H_latex,
            params=params,
            qubits=[1],
            bosons=[5],
            customs=[],
            c_ops_latex=None,
            t_name="t",
        )


def test_compile_model_invalid_syntax():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + \omega_b b_{1}^\dagger b_{1"
    params = {"omega_0": 2.0, "omega_b": 1.0}
    with pytest.raises(api.DSLValidationError):
        api.compile_model(
            H_latex=H_latex,
            params=params,
            qubits=[1],
            bosons=[5],
            customs=[],
            c_ops_latex=None,
            t_name="t",
        )


def test_compile_model_undefined_parameter():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + \omega_b b_{1}^\dagger b_{1}"
    params = {"omega_0": 2.0}  # omega_b is missing
    with pytest.raises(api.DSLValidationError):
        api.compile_model(
            H_latex=H_latex,
            params=params,
            qubits=[1],
            bosons=[5],
            customs=[],
            c_ops_latex=None,
            t_name="t",
        )


def test_compile_model_invalid_collapse_operator():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    c_ops_latex = [
        r"\sqrt{\gamma} \sigma_{x,1}"
    ]  # sigma_x is not a valid collapse operator
    params = {"omega_0": 2.0, "gamma": 0.3}
    model = api.compile_model(
        H_latex=H_latex,
        params=params,
        qubits=[1],
        bosons=[],
        customs=[],
        c_ops_latex=c_ops_latex,
        t_name="t",
    )
    assert len(model.c_ops) == 1


def test_make_config_no_subsystems():
    cfg = api.make_config(qubits=[], bosons=[])
    subs = cfg.all_subsystems()
    assert subs == []


def test_make_config_invalid_boson_deformation():
    with pytest.raises(Exception):
        api.make_config(bosons=[(3, "a", r"\unknown_func(n)")])


def test_compile_model_numpy_backend_delta():
    cfg = api.make_config(qubits=[1], bosons=[])
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    params = {"omega_0": 1.0}
    if not hasattr(api, "compile_model"):
        pytest.skip("compile_model not available in installed package")
    model_np = api.compile_model(
        H_latex=H_latex,
        params=params,
        backend="numpy",
        config=cfg,
    )
    assert model_np.H is not None


def test_compile_model_empty_hamiltonian():
    H_latex = r""
    params = {}
    with pytest.raises(Exception):
        api.compile_model(
            H_latex=H_latex,
            params=params,
            qubits=[1],
            bosons=[],
            customs=[],
            c_ops_latex=None,
            t_name="t",
        )


def test_compile_model_invalid_time_variable():
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega_d x) \sigma_{x,1}"
    params = {"omega_0": 2.0, "A": 0.5, "omega_d": 1.0}
    with pytest.raises(api.DSLValidationError):
        api.compile_model(
            H_latex=H_latex,
            params=params,
            qubits=[1],
            bosons=[],
            customs=[],
            c_ops_latex=None,
            t_name="t",
        )


def test_compile_model_numpy_backend():
    cfg = api.make_config(qubits=[1], bosons=[])
    model = api.compile_model(
        H_latex=r"\delta \sigma_{x,1}",
        params={"delta": 0.4},
        backend="numpy",
        config=cfg,
    )
    assert isinstance(model.H, np.ndarray)
    assert model.time_dependent is False


def test_compile_model_jax():
    cfg = api.make_config(qubits=[1], bosons=[])
    H_latex = r"A \cos(\omega t) \sigma_{x,1}"
    params = {"A": 0.5, "omega": 1.0}
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_platforms", "cpu")
    model, diag = api.compile_model(
        H_latex=H_latex,
        params=params,
        backend="jax",
        config=cfg,
        diagnostics=True,
    )
    assert model.time_dependent is True
    assert diag["time_dependent"] is True
    assert diag["term_count"] == 1
    assert "t" in diag["time_symbols"]


def test_compile_model_invalid_backend():
    with pytest.raises(api.DSLValidationError):
        api.compile_model(
            H_latex=r"\omega \sigma_{z,1}",
            params={"omega": 1.0},
            backend="unknown",
            qubits=[1],
            bosons=[],
        )


def test_lint_latex_model_reports_signature():
    cfg = api.make_config(qubits=[1], bosons=[])
    diag = api.lint_latex_model(
        H_latex=r"g \sigma_{x,1} \sigma_{x,1}",
        config=cfg,
        t_name="t",
    )
    assert diag["term_count"] == 1
    sig = diag["operator_signature"]
    assert ("qubit", "sx", 1, 2) in sig
    assert diag["has_time_dep"] is False


def test_make_config_invalid_boson_tuple_length():
    with pytest.raises(TypeError):
        api.make_config(bosons=[(2, "a", "b", "extra")])
