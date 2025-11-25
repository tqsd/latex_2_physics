import numpy as np

from latex_parser.config_utils import make_config, resolve_config
from latex_parser.dsl import HilbertConfig, QubitSpec


def test_resolve_config_returns_existing():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    out = resolve_config(
        config=cfg,
        auto_config=False,
        H_latex="",
        c_ops_latex=None,
        qubits=0,
        bosons=[],
        customs=None,
        default_boson_cutoff=2,
        custom_templates=None,
    )
    assert out is cfg


def test_make_config_deformation_from_latex():
    cfg = make_config(bosons=[(3, "a", r"\sqrt{n}")])
    assert len(cfg.bosons) == 1
    boson_spec = cfg.bosons[0]
    assert boson_spec.deformation is not None
    vals = boson_spec.deformation(np.arange(boson_spec.cutoff))
    assert vals.shape == (boson_spec.cutoff,)
