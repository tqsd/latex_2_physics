import pytest
import sympy as sp

from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    QubitSpec,
)
from latex_parser.ir import (
    OperatorFunctionRef,
    Term,
    _expand_noncommutative_powers,
    _rescue_implicit_scalar_funcs,
    _rescue_merged_time_scalars,
    expr_to_ir,
    latex_to_ir,
    parse_latex_expr,
    map_ir_terms,
)


def _term_ops_signature(terms: list[Term]) -> set[tuple[str, str, int, int]]:
    sig: set[tuple[str, str, int, int]] = set()
    for term in terms:
        for r in term.ops:
            if isinstance(r, OperatorFunctionRef):
                sig.add(
                    (f"func:{r.func_name}", r.arg.op_name, r.arg.index, r.arg.power)
                )
            else:
                sig.add((r.kind, r.op_name, r.index, r.power))
    return sig


def test_driven_qubit_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"""
        \frac{\omega_0}{2} \sigma_{z,1}
        + A \cos(\omega t) \sigma_{x,1}
    """
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 2
    assert ir.has_time_dep is True
    sz_terms = [term for term in ir.terms if term.ops and term.ops[0].op_name == "sz"]
    sx_terms = [term for term in ir.terms if term.ops and term.ops[0].op_name == "sx"]
    assert len(sz_terms) == 1
    assert len(sx_terms) == 1
    sz_syms = {s.name for s in sz_terms[0].scalar_expr.free_symbols}
    sx_syms = {s.name for s in sx_terms[0].scalar_expr.free_symbols}
    assert sz_syms == {"omega_{0}"}
    assert {"A", "omega", "t"}.issubset(sx_syms)


def test_dicke_like_ir():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=5)],
        customs=[
            CustomSpec(
                label="c",
                index=1,
                dim=3,
                operators={"Jz": object(), "Jp": object(), "Jm": object()},
            )
        ],
    )
    H_latex = r"""
        \omega_0 J_{z,1}
        + \omega_c a_{1}^{\dagger} a_{1}
        + g (a_{1}^{\dagger} + a_{1}) (J_{+,1} + J_{-,1})
    """
    ir = latex_to_ir(H_latex, cfg)
    sig = _term_ops_signature(ir.terms)
    expected_subset = {
        ("custom", "Jz", 1, 1),
        ("custom", "Jp", 1, 1),
        ("custom", "Jm", 1, 1),
        ("boson", "a", 1, 1),
        ("boson", "adag", 1, 1),
    }
    assert expected_subset.issubset(sig)
    all_scalars = sp.simplify(sum(term.scalar_expr for term in ir.terms))
    names = {s.name for s in all_scalars.free_symbols}
    for needed in ("omega_{0}", "omega_{c}", "g"):
        assert needed in names


def test_deformed_boson_ir():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=4, deformation=lambda n: n + 1)],
        customs=[],
    )
    H_latex = r"g (\tilde{a}_{1} + \tilde{a}_{1}^{\dagger})"
    ir = latex_to_ir(H_latex, cfg)
    sig = _term_ops_signature(ir.terms)
    assert ("boson", "af", 1, 1) in sig
    assert ("boson", "adagf", 1, 1) in sig


def test_cos_phi_scalar_ir():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    H_latex = r"\cos(\phi) + \Omega"
    ir = latex_to_ir(H_latex, cfg, t_name="t", time_symbols=())
    assert ir.has_time_dep is False
    assert len(ir.terms) == 2
    for term in ir.terms:
        assert term.ops == []
        names = {s.name for s in term.scalar_expr.free_symbols}
        assert "phi" in names or "Omega" in names


def test_power_handling_ir():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    H_latex = r"k \hat{n}_{1}^2"
    ir = latex_to_ir(H_latex, cfg)
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert {s.name for s in term.scalar_expr.free_symbols} == {"k"}
    assert len(term.ops) == 1
    op = term.ops[0]
    assert (op.kind, op.op_name, op.index, op.power) == ("boson", "n", 1, 2)


def test_negative_power_rejected_ir():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    H_latex = r"\hat{n}_{1}^{-1}"
    expr = parse_latex_expr(H_latex)
    with pytest.raises(DSLValidationError):
        expr_to_ir(expr, cfg)


def test_trig_mix_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"""
        \sin(\omega t) \sigma_{x,1}
        + \tan(\omega t) \sigma_{y,1}
    """
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is True
    sig = _term_ops_signature(ir.terms)
    expected = {("qubit", "sx", 1, 1), ("qubit", "sy", 1, 1)}
    assert expected.issubset(sig)
    all_scalars = sp.simplify(sum(term.scalar_expr for term in ir.terms))
    names = {s.name for s in all_scalars.free_symbols}
    assert "omega" in names and "t" in names


def test_static_scalar_only_ir():
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[])
    H_latex = r"\omega_0 + A"
    ir = latex_to_ir(H_latex, cfg)
    assert len(ir.terms) == 2
    assert ir.has_time_dep is False
    for term in ir.terms:
        assert term.ops == []
        assert term.scalar_expr.free_symbols


def test_pure_operator_unit_coeff_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\sigma_{x,1}"
    ir = latex_to_ir(H_latex, cfg)
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert term.scalar_expr == 1
    assert len(term.ops) == 1
    op = term.ops[0]
    assert (op.kind, op.op_name, op.index, op.power) == ("qubit", "sx", 1, 1)


def test_two_qubit_coupling_ir():
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=2)],
        bosons=[],
        customs=[],
    )
    H_latex = r"J \sigma_{x,1} \sigma_{x,2}"
    ir = latex_to_ir(H_latex, cfg)
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert {s.name for s in term.scalar_expr.free_symbols} == {"J"}
    sig = _term_ops_signature(ir.terms)
    expected = {("qubit", "sx", 1, 1), ("qubit", "sx", 2, 1)}
    assert expected.issubset(sig)


def test_gaussian_pulse_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \exp(-(t / \tau)^2) \sigma_{x,1}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 1 and ir.has_time_dep
    term = ir.terms[0]
    names = {s.name for s in term.scalar_expr.free_symbols}
    assert "A" in names and "t" in names
    sig = _term_ops_signature(ir.terms)
    assert {("qubit", "sx", 1, 1)}.issubset(sig)


def test_time_dep_collapse_like_ir():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    C_latex = r"\exp(-a t) a_{1}"
    ir = latex_to_ir(C_latex, cfg, t_name="t")
    assert len(ir.terms) == 1 and ir.has_time_dep
    term = ir.terms[0]
    names = {s.name for s in term.scalar_expr.free_symbols}
    assert "a" in names and "t" in names
    sig = _term_ops_signature(ir.terms)
    assert {("boson", "a", 1, 1)}.issubset(sig)


def test_scalar_index_like_symbol_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"h_{1} \sigma_{x,1}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 1
    term = ir.terms[0]
    names = {s.name for s in term.scalar_expr.free_symbols}
    assert len(names) == 1 and "h" in next(iter(names))
    sig = _term_ops_signature(ir.terms)
    assert {("qubit", "sx", 1, 1)}.issubset(sig)


def test_two_mode_beamsplitter_ir():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[
            BosonSpec(label="a", index=1, cutoff=5),
            BosonSpec(label="a", index=2, cutoff=5),
        ],
        customs=[],
    )
    H_latex = r"k (a_{1}^{\dagger} a_{2} + a_{2}^{\dagger} a_{1})"
    ir = latex_to_ir(H_latex, cfg)
    assert len(ir.terms) == 2
    for term in ir.terms:
        assert {s.name for s in term.scalar_expr.free_symbols} == {"k"}
    sig = _term_ops_signature(ir.terms)
    expected_subset = {
        ("boson", "adag", 1, 1),
        ("boson", "a", 2, 1),
        ("boson", "adag", 2, 1),
        ("boson", "a", 1, 1),
    }
    assert expected_subset.issubset(sig)


def test_operator_inside_exp_allowed_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\exp(\sigma_{z,1})"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert len(term.ops) == 1 and isinstance(term.ops[0], OperatorFunctionRef)
    opref = term.ops[0]
    assert opref.func_name == "exp" and opref.arg.op_name == "sz"


def test_operator_inside_sin_allowed_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\sin(\sigma_{x,1})"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert len(term.ops) == 1 and isinstance(term.ops[0], OperatorFunctionRef)
    opref = term.ops[0]
    assert opref.func_name == "sin" and opref.arg.op_name == "sx"


def test_operator_function_rejects_sum_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\cos(\sigma_{x,1} + \sigma_{y,1})"
    expr = parse_latex_expr(H_latex)
    with pytest.raises(DSLValidationError):
        expr_to_ir(expr, cfg, t_name="t")


def test_operator_function_rejects_scalar_factor_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\exp(\sigma_{z,1} \sigma_{x,1})"
    expr = parse_latex_expr(H_latex)
    with pytest.raises(DSLValidationError):
        expr_to_ir(expr, cfg, t_name="t")


def test_operator_function_power_ir():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=3)], customs=[]
    )
    H_latex = r"\cos(n_{1}^{2})"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert len(term.ops) == 1 and isinstance(term.ops[0], OperatorFunctionRef)
    opref = term.ops[0]
    assert (
        opref.func_name == "cos" and opref.arg.op_name == "n" and opref.arg.power == 2
    )


def test_operator_function_with_scalar_factor_ir():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=3)], customs=[]
    )
    H_latex = r"\exp(t \, n_{1})"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is True
    term = ir.terms[0]
    assert len(term.ops) == 1 and isinstance(term.ops[0], OperatorFunctionRef)
    opref = term.ops[0]
    assert opref.func_name == "exp" and opref.arg.op_name == "n"
    assert "t" in {s.name for s in opref.scalar_factor.free_symbols}


def test_rescue_merged_time_scalars_simple_ir():
    expr = sp.Symbol("omega_{dt}") * sp.Symbol("sx_{1}")
    rescued = _rescue_merged_time_scalars(expr, t_name="t")
    names = {s.name for s in rescued.free_symbols}
    assert "omega_{d}" in names and "t" in names and "omega_{dt}" not in names


def test_rescue_merged_time_scalars_skips_if_t_present_ir():
    expr = sp.Symbol("omega_{dt}") + sp.Symbol("t")
    rescued = _rescue_merged_time_scalars(expr, t_name="t")
    assert rescued == expr


def test_implicit_scalar_funcs_scalar_only_args_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    x = sp.Symbol("x")
    g = sp.Function("g")
    expr = g(x)
    fixed = _rescue_implicit_scalar_funcs(expr, cfg)
    assert fixed == expr


def test_symbol_like_boson_index_without_boson_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"a_{1} h_{0}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert term.ops == []
    names = {s.name for s in term.scalar_expr.free_symbols}
    assert "a_{1}" in names and "h_{0}" in names


def test_implicit_scalar_func_with_operator_arg_ir():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    a1 = sp.Symbol("a_{1}")
    adag1 = sp.Symbol("adag_{1}")
    g = sp.Function("g")
    expr = g(a1 + adag1)
    fixed = _rescue_implicit_scalar_funcs(expr, cfg)
    assert isinstance(fixed, sp.Mul)
    syms = {s.name for s in fixed.free_symbols}
    assert "g" in syms and "a_{1}" in syms and "adag_{1}" in syms


def test_implicit_scalar_func_scalar_only_stays_function_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    x = sp.Symbol("x")
    g = sp.Function("g")
    expr = g(x)
    fixed = _rescue_implicit_scalar_funcs(expr, cfg)
    assert fixed == expr


def test_time_detection_unregistered_symbol_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \cos(\omega \tau)"
    ir = latex_to_ir(H_latex, cfg, t_name="t", time_symbols=())
    assert ir.has_time_dep is False
    assert len(ir.terms) == 1
    term = ir.terms[0]
    assert term.ops == []
    names = {s.name for s in term.scalar_expr.free_symbols}
    assert names == {"A", "omega", "tau"}


def test_large_expansion_hits_max_terms_ir():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[
            BosonSpec(label="a", index=1, cutoff=2),
            BosonSpec(label="a", index=2, cutoff=2),
        ],
        customs=[],
    )
    k = 10
    H_latex = rf"(a_1 + a_2)^{k}"
    with pytest.raises(DSLValidationError):
        latex_to_ir(H_latex, cfg, t_name="t")


def test_power_on_sum_boson_ir():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=5)], customs=[]
    )
    H_latex = r"(a_{1} + a_{1}^{\dagger})^{2}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 4
    monomials = []
    for term in ir.terms:
        assert term.scalar_expr.free_symbols == set()
        seq: list[tuple[str, int]] = []
        for op in term.ops:
            assert op.kind == "boson" and op.label == "a" and op.index == 1
            for _ in range(op.power):
                seq.append((op.op_name, 1))
        assert len(seq) == 2
        monomials.append(tuple(seq))
    expected = {
        (("a", 1), ("a", 1)),
        (("a", 1), ("adag", 1)),
        (("adag", 1), ("a", 1)),
        (("adag", 1), ("adag", 1)),
    }
    assert set(monomials) == expected


def test_power_on_sum_qubit_ir():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"(\sigma_{x,1} + \sigma_{y,1})^{2}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert len(ir.terms) == 4
    monomials = []
    for term in ir.terms:
        assert term.scalar_expr.free_symbols == set()
        seq: list[tuple[str, int]] = []
        for op in term.ops:
            assert op.kind == "qubit" and op.label == "q" and op.index == 1
            for _ in range(op.power):
                seq.append((op.op_name, 1))
        assert len(seq) == 2
        monomials.append(tuple(seq))
    expected = {
        (("sx", 1), ("sx", 1)),
        (("sx", 1), ("sy", 1)),
        (("sy", 1), ("sx", 1)),
        (("sy", 1), ("sy", 1)),
    }
    assert set(monomials) == expected


def test_expand_noncommutative_powers_power_one_no_change():
    A = sp.Symbol("A", commutative=False)
    expr = (A + 2) ** 1
    assert _expand_noncommutative_powers(expr) == expr


def test_rescue_implicit_scalar_funcs_no_op_args():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=2)], customs=[]
    )
    expr = sp.parse_expr("g(1 + x)", evaluate=False)
    rescued = _rescue_implicit_scalar_funcs(expr, cfg)
    # Without operator args, expression should be untouched
    assert rescued == expr


def test_rescue_implicit_scalar_funcs_with_operator_argument():
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=2)],
        customs=[],
    )
    expr = sp.parse_expr("g(a_1 + adag_1)", evaluate=False)
    rescued = _rescue_implicit_scalar_funcs(expr, cfg)
    g = sp.Symbol("g")
    a = sp.Symbol("a_1")
    adag = sp.Symbol("adag_1")
    assert rescued == g * (a + adag)


def test_expand_noncommutative_powers():
    A = sp.Symbol("A", commutative=False)
    B = sp.Symbol("B", commutative=False)
    expr = (A + B) ** 2
    expanded = _expand_noncommutative_powers(expr)
    assert expanded.expand() == A**2 + A * B + B * A + B**2


def test_expr_to_ir_skips_non_op_symbols():
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=2)], customs=[]
    )
    expr = sp.Symbol("x") + sp.Symbol("y")
    ir = expr_to_ir(expr, cfg, t_name="t")
    assert len(ir.terms) == 2
    assert all(term.ops == [] for term in ir.terms)


def test_latex_to_ir_time_symbol_detection_extra():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    # ensure time_symbols tuple extends detection
    ir = latex_to_ir(r"\cos(s) \sigma_{z,1}", cfg, t_name="t", time_symbols=("s",))
    assert ir.has_time_dep is True


def test_map_ir_terms():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"g \sigma_{z,1}", cfg)

    def double_scalar(term: Term) -> Term:
        return Term(scalar_expr=2 * term.scalar_expr, ops=term.ops)

    ir2 = map_ir_terms(ir, double_scalar)
    assert len(ir2.terms) == len(ir.terms)
    assert str(ir2.terms[0].scalar_expr) == "2*g"
