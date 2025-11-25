import pytest

from latex_parser.compile_core import (
    available_backends,
    backend_capabilities,
    compile_model_core,
    register_backend,
)
from latex_parser.dsl import DSLValidationError, HilbertConfig
from latex_parser.backend_utils import (
    collect_parameter_names,
    validate_required_params,
)
from latex_parser.ir import latex_to_ir


def test_available_backends_has_defaults():
    backends = available_backends()
    assert "qutip" in backends
    assert "jax" in backends
    assert "numpy" in backends


def test_backend_capabilities_exposed():
    caps = backend_capabilities("qutip")
    assert caps is not None
    assert caps.get("time_dependent") is True


def test_register_backend_and_dispatch_called():
    called = {}

    def _stub(**kwargs):
        called["kwargs"] = kwargs
        return "ok"

    register_backend("dummy", _stub)
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    result = compile_model_core(
        backend="dummy",
        H_latex=r"\omega",
        params={"omega": 1.0},
        config=cfg,
        c_ops_latex=None,
        t_name="t",
        time_symbols=None,
    )
    assert result == "ok"
    assert called["kwargs"]["params"]["omega"] == 1.0


def test_compile_model_core_missing_param_raises():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    with pytest.raises(DSLValidationError):
        compile_model_core(
            backend="qutip",
            H_latex=r"\omega",
            params={},
            config=cfg,
            c_ops_latex=None,
            t_name="t",
            time_symbols=None,
        )


def test_compile_model_core_alias_braces_allowed():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])

    def _stub(**kwargs):
        return kwargs["params"]

    register_backend("alias_backend", _stub)
    params_out = compile_model_core(
        backend="alias_backend",
        H_latex=r"\omega_{0}",
        params={"omega_0": 2.0},
        config=cfg,
        c_ops_latex=None,
        t_name="t",
        time_symbols=None,
    )
    assert params_out["omega_0"] == 2.0


def test_collect_parameter_names_skips_time_and_ops():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    ir = latex_to_ir(r"A \cos(\omega t) + \alpha", cfg, t_name="t")
    names = collect_parameter_names(ir, cfg, {"t"})
    assert names == {"A", "omega", "alpha"}
    assert "t" not in names


def test_validate_required_params_ignores_time():
    validate_required_params({"omega", "t"}, {"omega": 1.0}, {"t"})


def test_validate_required_params_missing():
    with pytest.raises(DSLValidationError):
        validate_required_params({"omega", "gamma"}, {"omega": 1.0}, set())
