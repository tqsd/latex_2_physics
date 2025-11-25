import logging

import numpy as np
import pytest

from latex_parser import auto_config
from latex_parser.backend_base import BackendBase
from latex_parser.backend_numpy import compile_open_system_from_latex as numpy_compile
from latex_parser.backend_utils import resolve_param
from latex_parser.backend_cache import BaseOperatorCache
from latex_parser.backend_qutip import QutipOperatorCache
from latex_parser.config_utils import make_config
from latex_parser.dsl import (
    DSLValidationError,
    HilbertConfig,
    QubitSpec,
)
from latex_parser.dsl import BosonSpec
from latex_parser.ir import HamiltonianIR
from latex_parser.operator_functions import apply_operator_function
import latex_parser.operator_functions as opfuncs
import sympy as sp
from latex_parser.ir import latex_to_ir


class _DummyBackend(BackendBase):
    def _make_cache(self, config, options=None):
        return None

    def _compile_static(self, ir, cache, params, options=None):
        return "static"

    def _compile_time_dependent(
        self, ir, cache, params, *, t_name, time_symbols, options=None
    ):
        return "td"


def test_backend_base_rejects_time_dependent_ir():
    ir = HamiltonianIR(terms=[], has_time_dep=True)
    backend = _DummyBackend()
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    with pytest.raises(DSLValidationError):
        backend.compile_static_from_ir(ir, cfg, params={})


def test_operator_functions_invalid_backend_and_func():
    mat = np.eye(2)
    with pytest.raises(ValueError):
        apply_operator_function(mat, "exp", backend="nope")
    with pytest.raises(ValueError):
        apply_operator_function(mat, "log", backend="numpy")


def test_backend_numpy_converts_time_dependent():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    compiled = numpy_compile(
        H_latex=r"A \cos(\omega t) \sigma_{x,1}",
        params={"A": 0.1, "omega": 1.0},
        config=cfg,
        c_ops_latex=[],
        t_name="t",
        time_symbols=None,
    )
    assert isinstance(compiled.H, list)
    assert len(compiled.H) > 1


def test_config_utils_invalid_boson_entry():
    with pytest.raises(TypeError):
        make_config(qubits=0, bosons=[(1, 2, 3, 4)], customs=None)


def test_backend_utils_resolve_param_warns(caplog):
    caplog.set_level(logging.WARNING)
    params = {"omega_c": 1.0, "omega_{c}": 2.0}
    key, val = resolve_param(
        "omega_{c}", params, warn_on_multiple=True, logger=logging.getLogger()
    )
    assert key in params
    assert any("matched multiple keys" in rec.message for rec in caplog.records)


def test_auto_config_requires_custom_template():
    with pytest.raises(DSLValidationError):
        auto_config.infer_config_from_latex(
            H_latex=r"Jx_{1}",
            c_ops_latex=None,
            default_boson_cutoff=3,
            custom_templates=None,
        )


def test_qubit_spec_invalid_index():
    with pytest.raises(DSLValidationError):
        QubitSpec(label="q", index=0)


def test_base_operator_cache_duplicate_detected():
    class _Cache(BaseOperatorCache[np.ndarray]):
        def _local_identity(self, dim: int) -> np.ndarray:
            return np.eye(dim)

        def _kron(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.kron(a, b)

    cfg = HilbertConfig.__new__(HilbertConfig)
    cfg.qubits = [QubitSpec(label="q", index=1), QubitSpec(label="q", index=1)]
    cfg.bosons = []
    cfg.customs = []
    with pytest.raises(DSLValidationError):
        _Cache(cfg)


def test_qutip_cache_missing_subsystem_and_bad_ops():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = QutipOperatorCache(cfg)
    with pytest.raises(DSLValidationError):
        cache._find_subsystem("qubit", "q", 2)
    with pytest.raises(DSLValidationError):
        cache._make_local_qubit_op("invalid", cache._find_subsystem("qubit", "q", 1))
    boson_cfg = HilbertConfig(
        bosons=[BosonSpec(label="a", index=1, cutoff=2)], qubits=[], customs=[]
    )
    boson_cache = QutipOperatorCache(boson_cfg)
    with pytest.raises(DSLValidationError):
        boson_cache._make_local_boson_op(
            "bad", boson_cache._find_subsystem("boson", "a", 1)
        )


def test_operator_functions_jax_unavailable(monkeypatch):
    monkeypatch.setattr(opfuncs, "_jnp", None)
    monkeypatch.setattr(opfuncs, "_jax_expm", None)
    with pytest.raises(RuntimeError):
        apply_operator_function(np.eye(1), "exp", backend="jax")


def test_backend_base_alias_helpers():
    assert BackendBase._param_aliases("omega_{c}")[:2] == ["omega_{c}", "omega_c"]
    key, val = BackendBase._lookup_param_name("omega_{c}", {"omega_c": 1.0})
    assert key == "omega_c" and val == 1.0
    assert BackendBase._expr_has_time(sp.Symbol("t"), {"t"})


def test_config_utils_deformation_string():
    cfg = make_config(qubits=0, bosons=[(2, "a", "n+1")], customs=None)
    assert cfg.bosons[0].deformation is not None


def test_ir_operator_function_rejects_sum():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    with pytest.raises(DSLValidationError):
        latex_to_ir(r"\exp(\sigma_{x,1} + \sigma_{z,1})", cfg, t_name="t")


def test_backend_jax_require_raises(monkeypatch):
    import latex_parser.backend_jax as bj

    monkeypatch.setattr(bj, "_JAX_AVAILABLE", False)
    monkeypatch.setattr(bj, "jax", None)
    monkeypatch.setattr(bj, "jnp", None)
    with pytest.raises(Exception):
        bj._require_jax()
