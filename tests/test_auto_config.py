import numpy as np
import pytest
from qutip import Qobj  # type: ignore

from latex_parser.auto_config import infer_config_from_latex
from latex_parser.dsl import CustomSpec, DSLValidationError


def test_infer_config_qubits_bosons():
    H = r"\omega a_{2} + g \sigma_{x,3}"
    c_ops = [r"\sqrt{\kappa} a_{1}"]

    cfg = infer_config_from_latex(H, c_ops_latex=c_ops, default_boson_cutoff=4)

    assert {q.index for q in cfg.qubits} == {1, 2, 3}
    assert {b.index for b in cfg.bosons} == {1, 2}
    assert all(b.cutoff == 4 for b in cfg.bosons)
    assert cfg.customs == []


def test_infer_config_custom_requires_template():
    H = r"J_{z,1}"
    with pytest.raises(DSLValidationError):
        infer_config_from_latex(H)

    template = CustomSpec(
        label="c",
        index=1,
        dim=2,
        operators={"Jz": Qobj(np.diag([1.0, -1.0]))},
    )
    cfg = infer_config_from_latex(
        H, custom_templates={"c": template}, default_boson_cutoff=3
    )
    assert len(cfg.customs) == 1
    spec = cfg.customs[0]
    assert spec.index == 1
    assert set(spec.operators.keys()) == {"Jz"}
