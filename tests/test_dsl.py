import pytest
import sympy as sp
from sympy.parsing.latex import parse_latex

from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    QubitSpec,
    canonicalize_physics_latex,
    extract_operator_refs_from_latex,
    make_finite_sum_pattern,
    parse_operator_symbol,
    try_parse_operator_symbol,
)


def test_jc_boson_qubit():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=10)],
        customs=[],
    )
    H_latex = r"""
    \omega_c \hat{n}_{1}
    + \frac{1}{2} \omega_q \sigma_{z,1}
    + g \left( a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1} \right)
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {
        ("boson", "n", 1),
        ("boson", "a", 1),
        ("boson", "adag", 1),
        ("qubit", "sz", 1),
        ("qubit", "sp", 1),
        ("qubit", "sm", 1),
    }
    assert expected.issubset(triplets)


def test_custom_subsystem():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[CustomSpec(label="c", index=1, dim=3, operators={"A": object()})],
    )
    expr_latex = r"\lambda A_{1}"
    refs = extract_operator_refs_from_latex(expr_latex, cfg)
    assert len(refs) == 1
    ref = refs[0]
    assert (ref.kind, ref.label, ref.index, ref.op_name) == ("custom", "c", 1, "A")


def test_two_mode_bosonic_beamsplitter():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[
            BosonSpec(label="a", index=1, cutoff=10),
            BosonSpec(label="a", index=2, cutoff=10),
        ],
        customs=[],
    )
    H_latex = r"""
    \omega_1 \hat{n}_{1}
    + \omega_2 \hat{n}_{2}
    + g \left( a_{1} a_{2}^{\dagger} + a_{1}^{\dagger} a_{2} \right)
    + \chi \hat{n}_{1} \hat{n}_{2}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {
        ("boson", "n", 1),
        ("boson", "n", 2),
        ("boson", "a", 1),
        ("boson", "a", 2),
        ("boson", "adag", 1),
        ("boson", "adag", 2),
    }
    assert expected.issubset(triplets)


def test_two_qubit_ising():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=2)],
        bosons=[],
        customs=[],
    )
    H_latex = r"""
    \frac{\omega_1}{2} \sigma_{z,1}
    + \frac{\omega_2}{2} \sigma_{z,2}
    + J \sigma_{z,1} \sigma_{z,2}
    + g \left( \sigma_{x,1} + \sigma_{x,2} \right)
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {
        ("qubit", "sz", 1),
        ("qubit", "sz", 2),
        ("qubit", "sx", 1),
        ("qubit", "sx", 2),
    }
    assert expected.issubset(triplets)


def test_qubit_boson_rabi():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=10)],
        customs=[],
    )
    H_latex = r"""
    \omega_c \hat{n}_{1}
    + \frac{\omega_q}{2} \sigma_{z,1}
    + g \left( a_{1} + a_{1}^{\dagger} \right) \sigma_{x,1}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {
        ("boson", "n", 1),
        ("boson", "a", 1),
        ("boson", "adag", 1),
        ("qubit", "sz", 1),
        ("qubit", "sx", 1),
    }
    assert expected.issubset(triplets)


def test_custom_plus_qubit():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[CustomSpec(label="c", index=1, dim=3, operators={"A": object()})],
    )
    H_latex = r"\lambda A_{1} \sigma_{z,1}"
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {("custom", "A", 1), ("qubit", "sz", 1)}
    assert expected.issubset(triplets)


def test_three_qubit_coupled():
    cfg = HilbertConfig(
        qubits=[
            QubitSpec(label="q", index=1),
            QubitSpec(label="q", index=2),
            QubitSpec(label="q", index=3),
        ],
        bosons=[],
        customs=[],
    )
    H_latex = r"""
    \sigma_{z,1} + \sigma_{z,2} + \sigma_{z,3}
    + 0.5 \sigma_{x,1} \sigma_{x,2}
    + 0.25 \sigma_{x,2} \sigma_{x,3}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {
        ("qubit", "sz", 1),
        ("qubit", "sz", 2),
        ("qubit", "sz", 3),
        ("qubit", "sx", 1),
        ("qubit", "sx", 2),
        ("qubit", "sx", 3),
    }
    assert expected.issubset(triplets)


def test_three_qubit_transverse_ising():
    cfg = HilbertConfig(
        qubits=[
            QubitSpec(label="q", index=1),
            QubitSpec(label="q", index=2),
            QubitSpec(label="q", index=3),
        ],
        bosons=[],
        customs=[],
    )
    H_latex = r"""
    -J \left( \sigma_{z,1} \sigma_{z,2} + \sigma_{z,2} \sigma_{z,3} \right)
    -h \left( \sigma_{x,1} + \sigma_{x,2} + \sigma_{x,3} \right)
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {
        ("qubit", "sz", 1),
        ("qubit", "sz", 2),
        ("qubit", "sz", 3),
        ("qubit", "sx", 1),
        ("qubit", "sx", 2),
        ("qubit", "sx", 3),
    }
    assert expected.issubset(triplets)


def test_driven_qubit_time_dependent():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"""
    \frac{\omega}{2} \sigma_{z,1}
    + A \exp\left( - (t / \sigma)^2 \right) \sigma_{x,1}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {("qubit", "sz", 1), ("qubit", "sx", 1)}
    assert expected.issubset(triplets)


def test_dispersive_jc():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=10)],
        customs=[],
    )
    H_latex = r"""
    \omega_r a_{1}^{\dagger} a_{1}
    + \frac{\omega_q}{2} \sigma_{z,1}
    + \chi a_{1}^{\dagger} a_{1} \sigma_{z,1}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {("boson", "a", 1), ("boson", "adag", 1), ("qubit", "sz", 1)}
    assert expected.issubset(triplets)


def test_two_qubit_heisenberg_like():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=2)],
        bosons=[],
        customs=[],
    )
    H_latex = r"""
    h_1 \sigma_{z,1} + h_2 \sigma_{z,2}
    + J_x \sigma_{x,1} \sigma_{x,2}
    + J_y \sigma_{y,1} \sigma_{y,2}
    + J_z \sigma_{z,1} \sigma_{z,2}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {
        ("qubit", "sz", 1),
        ("qubit", "sz", 2),
        ("qubit", "sx", 1),
        ("qubit", "sx", 2),
        ("qubit", "sy", 1),
        ("qubit", "sy", 2),
    }
    assert expected.issubset(triplets)


def test_invalid_boson_label_rejected():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="b", index=1, cutoff=5)],
        customs=[],
    )
    assert cfg.bosons[0].label == "b"


def test_pauli_without_comma():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    refs = extract_operator_refs_from_latex(r"\sigma_{x 1}", cfg)
    assert any(r.op_name == "sx" and r.index == 1 for r in refs)


def test_pauli_with_parenthesized_index():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    refs = extract_operator_refs_from_latex(r"\sigma_x^{(1)}", cfg)
    assert any(r.op_name == "sx" and r.index == 1 for r in refs)


def test_creation_with_spaced_dagger():
    cfg = HilbertConfig(
        bosons=[BosonSpec(label="a", index=1, cutoff=5)], qubits=[], customs=[]
    )
    refs = extract_operator_refs_from_latex(r"a_{1} ^ {\dagger}", cfg)
    assert any(r.op_name == "adag" and r.index == 1 for r in refs)


def test_deformed_boson_canonicalizes_and_parses():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=5, deformation=lambda n: n + 1)],
        customs=[],
    )
    refs = extract_operator_refs_from_latex(
        r"\tilde{a}_{1} + \tilde{a}_{1}^{\dagger}", cfg
    )
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    assert ("boson", "af", 1) in triplets
    assert ("boson", "adagf", 1) in triplets


def test_scalar_symbol_does_not_raise():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=5)],
        customs=[],
    )
    lam = sp.Symbol("\\lambda")
    ref = try_parse_operator_symbol(lam, cfg)
    assert ref is None


def test_unknown_custom_operator_rejected():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[CustomSpec(label="c", index=1, dim=3, operators={"A": object()})],
    )
    expr_latex = r"\lambda B_{1}"
    canonical = canonicalize_physics_latex(expr_latex)
    expr = parse_latex(canonical)
    sym = next(iter(expr.free_symbols))
    with pytest.raises(Exception):
        parse_operator_symbol(sym, cfg)


def test_missing_qubit_for_operator():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    expr_latex = r"\sigma_{x,2}"
    canonical = canonicalize_physics_latex(expr_latex)
    expr = parse_latex(canonical)
    sym = next(iter(expr.free_symbols))
    with pytest.raises(Exception):
        parse_operator_symbol(sym, cfg)


def test_spin_J_operators_custom():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[
            CustomSpec(
                label="c",
                index=1,
                dim=3,
                operators={"Jx": object(), "Jy": object(), "Jz": object()},
            ),
        ],
    )
    H_latex = r"""
    \lambda J_{x,1}
    + \mu J_x^{(1)}
    + \nu J_{y,1}
    + \eta J_{z,1}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {("custom", "Jx", 1), ("custom", "Jy", 1), ("custom", "Jz", 1)}
    assert expected.issubset(triplets)


def test_spin_Jpm_custom():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[
            CustomSpec(
                label="c", index=1, dim=3, operators={"Jp": object(), "Jm": object()}
            )
        ],
    )
    H_latex = r"""
    \gamma J_{+,1}
    + \delta J_{-,1}
    """
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    expected = {("custom", "Jp", 1), ("custom", "Jm", 1)}
    assert expected.issubset(triplets)


def test_n_th_is_scalar():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    sym = sp.Symbol("n_{th}")
    ref = try_parse_operator_symbol(sym, cfg)
    assert ref is None


def test_omega_c_star_1_is_scalar():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    sym = sp.Symbol("omega_{c*1}")
    ref = try_parse_operator_symbol(sym, cfg)
    assert ref is None


def test_hat_n_without_braces_canonicalizes():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    H_latex = r"\omega_c \hat n_{1}"
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    triplets = {(r.kind, r.op_name, r.index) for r in refs}
    assert ("boson", "n", 1) in triplets


def test_creation_without_braces_canonicalizes():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    refs = extract_operator_refs_from_latex(r"a_1^{\dagger}", cfg)
    assert any(r.kind == "boson" and r.op_name == "adag" and r.index == 1 for r in refs)


def test_canonicalize_whitespace_and_newlines():
    H_latex = r"""
        \omega_c   \hat{n}_{1}
        +  g   a_{1}^{\dagger}
    """
    canonical = canonicalize_physics_latex(H_latex)
    assert "\n" not in canonical
    assert "n_{1}" in canonical
    assert r"\adag_{1}" in canonical


def test_all_subsystems_ordering():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=2)],
        bosons=[BosonSpec(label="a", index=1, cutoff=5)],
        customs=[CustomSpec(label="c", index=1, dim=3, operators={"A": object()})],
    )
    subs = cfg.all_subsystems()
    kinds_labels_indices = [(kind, spec.label, spec.index) for kind, spec in subs]
    assert kinds_labels_indices == [
        ("qubit", "q", 1),
        ("qubit", "q", 2),
        ("boson", "a", 1),
        ("custom", "c", 1),
    ]


def test_bare_identifier_ab_becomes_single_scalar():
    H_latex = r"ab"
    text = canonicalize_physics_latex(H_latex)
    expr = parse_latex(text)
    syms = {s.name for s in expr.free_symbols}
    assert not ("a" in syms and "b" in syms)
    assert len(syms) == 1


def test_bare_drive_word_is_scalar_symbol():
    H_latex = r"drive \, t"
    text = canonicalize_physics_latex(H_latex)
    expr = parse_latex(text)
    syms = {s.name for s in expr.free_symbols}
    assert any("drive" in name for name in syms)
    assert "t" in syms
    assert len(syms) == 2


def test_cos_with_and_without_backslash_same_scalar_structure():
    H1 = r"\cos(\omega t)"
    H2 = r"cos(\omega t)"
    text1 = canonicalize_physics_latex(H1)
    text2 = canonicalize_physics_latex(H2)
    expr1 = parse_latex(text1)
    expr2 = parse_latex(text2)
    syms1 = {s.name for s in expr1.free_symbols}
    syms2 = {s.name for s in expr2.free_symbols}
    assert syms1 == syms2 == {"omega", "t"}


def test_a1_treated_as_scalar_when_no_bosons_dsl():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"a_{1} h_{0}"
    text = canonicalize_physics_latex(H_latex)
    expr = parse_latex(text)
    refs = extract_operator_refs_from_latex(H_latex, cfg)
    assert refs == []
    syms = {s.name for s in expr.free_symbols}
    assert "a_{1}" in syms
    assert "h_{0}" in syms


def test_normalize_symbol_name_strips_braces():
    from latex_parser.dsl import _normalize_symbol_name

    assert _normalize_symbol_name("omega_{c}") == "omega_c"
    assert _normalize_symbol_name("Jx_{1}") == "Jx_1"


def test_parse_operator_symbol_custom_suffix_name():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[],
        customs=[
            CustomSpec(label="c", index=1, dim=2, operators={"Jx_foo": object()}),
        ],
    )
    ref = parse_operator_symbol(sp.Symbol("Jx_foo_1"), cfg)
    assert ref.kind == "custom"
    assert ref.op_name == "Jx_foo"


def test_try_parse_operator_symbol_unknown_base_returns_none():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    sym = sp.Symbol("abc_1")
    assert try_parse_operator_symbol(sym, cfg) is None


def test_hilbertconfig_all_subsystems_order_mixed():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=2), QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[CustomSpec(label="c", index=1, dim=2, operators={"A": object()})],
    )
    kinds = [kind for kind, _ in cfg.all_subsystems()]
    assert kinds[:2] == ["qubit", "qubit"]
    assert kinds[-1] == "custom"


def test_try_parse_operator_symbol_rejects_negative_index():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    sym = sp.Symbol("sx_-1")
    assert try_parse_operator_symbol(sym, cfg) is None


def test_parse_operator_symbol_invalid_pattern():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    with pytest.raises(DSLValidationError):
        parse_operator_symbol(sp.Symbol("weird__name_1"), cfg)


def test_parse_operator_symbol_non_integer_index():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    with pytest.raises(DSLValidationError):
        parse_operator_symbol(sp.Symbol("sx_x"), cfg)


def test_lookup_subsystem_label_fallback():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="qcustom", index=2)],
        bosons=[BosonSpec(label="acustom", index=1, cutoff=2)],
        customs=[],
    )
    # Label "q" should still resolve via index fallback.
    ref_q = parse_operator_symbol(sp.Symbol("sx_2"), cfg)
    assert ref_q.label == "qcustom"
    # Label "a" should resolve to acustom via index.
    ref_a = parse_operator_symbol(sp.Symbol("a_1"), cfg)
    assert ref_a.label == "acustom"


def test_extract_operator_refs_from_latex_mixed_scalars():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=3)],
        customs=[],
    )
    refs = extract_operator_refs_from_latex(r"\sigma_{x,1} + \omega t + a_{1}", cfg)
    kinds = {(r.kind, r.op_name) for r in refs}
    assert ("qubit", "sx") in kinds
    assert ("boson", "a") in kinds


def test_try_parse_operator_symbol_invalid_kind():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    sym = sp.Symbol("B_{1}")
    ref = try_parse_operator_symbol(sym, cfg)
    assert ref is None


def test_duplicate_qubit_rejected_in_config():
    with pytest.raises(DSLValidationError):
        HilbertConfig(
            qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=1)],
            bosons=[],
            customs=[],
        )


def test_finite_sum_macro_expands():
    pattern, repl = make_finite_sum_pattern("j", "a", 1, 3)
    canon = canonicalize_physics_latex(
        r"\sum_{j=1}^{3} a_{j}", extra_patterns=[(pattern, repl)]
    )
    assert canon == r"a_{1} + a_{2} + a_{3}"
