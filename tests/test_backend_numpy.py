import numpy as np

from latex_parser.backend_numpy import compile_open_system_from_latex
from latex_parser.dsl import HilbertConfig, QubitSpec


def test_backend_numpy_static_and_collapse():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\Delta \sigma_{z,1}"
    c_ops_latex = [r"\sqrt{\gamma} \sigma_{-,1}"]
    params = {"Delta": 1.2, "gamma": 0.3}

    model = compile_open_system_from_latex(
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=c_ops_latex,
    )

    assert isinstance(model.H, np.ndarray)
    assert model.H.shape == (2, 2)
    assert len(model.c_ops) == 1
    C = model.c_ops[0]
    assert isinstance(C, np.ndarray)
    sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(C, (params["gamma"] ** 0.5) * sm)
    assert model.time_dependent is False


def test_backend_numpy_time_dep_and_callable_term():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\exp(t \sigma_{z,1})"
    params: dict[str, complex] = {}

    model = compile_open_system_from_latex(
        H_latex=H_latex,
        params=params,
        config=cfg,
    )

    # Expect H = [H0, fk_callable]; H0 should be 0 matrix, fk produces array.
    assert isinstance(model.H, list)
    H0 = model.H[0]
    fk = model.H[1]
    assert np.allclose(H0, np.zeros((2, 2)))
    mat = fk(0.5, model.args)
    assert isinstance(mat, np.ndarray)
    manual = np.array([[np.exp(0.5), 0.0], [0.0, np.exp(-0.5)]], dtype=complex)
    assert np.allclose(mat, manual, atol=1e-10)
    assert model.time_dependent is True


def test_backend_numpy_time_dep_collapse_conversion():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    c_ops_latex = [r"\sqrt{\gamma} \exp(-t/2) \sigma_{-,1}"]
    params = {"gamma": 0.4}

    model = compile_open_system_from_latex(
        H_latex=r"0",
        params=params,
        config=cfg,
        c_ops_latex=c_ops_latex,
    )

    assert model.time_dependent is True
    assert len(model.c_ops) == 1
    C0, f = model.c_ops[0]
    assert isinstance(C0, np.ndarray)
    for t in (0.0, 0.5):
        val = f(t, model.args)
        assert np.isclose(val, (params["gamma"] ** 0.5) * np.exp(-t / 2))


def test_backend_numpy_time_dep_callable_operator_function():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    model = compile_open_system_from_latex(
        H_latex=r"\exp(t \sigma_{z,1})",
        params={},
        config=cfg,
    )
    assert isinstance(model.H, list)
    fk = model.H[1]
    mat = fk(0.2, model.args)
    assert isinstance(mat, np.ndarray)
    manual = np.diag([np.exp(0.2), np.exp(-0.2)])
    assert np.allclose(mat, manual)


def test_backend_numpy_static_only_identity_zero():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    model = compile_open_system_from_latex(
        H_latex="0",
        params={},
        config=cfg,
    )
    assert isinstance(model.H, np.ndarray)
    assert model.H.shape == (1, 1)
    assert model.H[0, 0] == 0


def test_backend_numpy_args_passthrough():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    params = {"A": 1.0}
    model = compile_open_system_from_latex(
        H_latex=r"A \sigma_{x,1}",
        params=params,
        config=cfg,
    )
    assert model.args == params
