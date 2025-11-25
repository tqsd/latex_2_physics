import numpy as np
import pytest
import sympy as sp
from latex_parser.backend_qutip import (
    QutipOperatorCache,
    _build_time_dep_term_callable,
    _lookup_param_name,
    compile_collapse_ops_from_latex,
    compile_open_system_from_latex,
    compile_static_hamiltonian_from_latex,
    compile_static_hamiltonian_ir,
    compile_time_dependent_hamiltonian_ir,
    term_to_qobj,
)
from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    LocalOperatorRef,
    QubitSpec,
)
from latex_parser.ir import latex_to_ir
from qutip import (  # type: ignore
    Qobj,
    destroy,
    qeye,
    sigmam,
    sigmap,
    sigmax,
    sigmaz,
    tensor,
)
from scipy.linalg import cosm, expm


def test_deformed_boson_backend():
    """
    Validate construction of f-deformed ladder operators.
    """

    def deform(n):
        return np.sqrt(n + 1.0)

    dim = 4
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=dim, deformation=deform)],
        customs=[],
    )
    cache = QutipOperatorCache(cfg)

    diag = Qobj(np.diag(deform(np.arange(dim))), dims=[[dim], [dim]])
    a = destroy(dim)

    # Also build using Qobj.sqrtm of (n + I) to exercise QuTiP's sqrtm path
    n_op = a.dag() * a
    diag_sqrtm = (n_op + qeye(dim)).sqrtm()

    af_expected = a * diag
    adagf_expected = diag * a.dag()
    af_expected_sqrtm = a * diag_sqrtm
    adagf_expected_sqrtm = diag_sqrtm * a.dag()

    af = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="af")
    )
    adagf = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="adagf")
    )

    assert (af - af_expected).norm() < 1e-12, (af - af_expected).norm()
    assert (adagf - adagf_expected).norm() < 1e-12, (adagf - adagf_expected).norm()
    assert (af - af_expected_sqrtm).norm() < 1e-12, (af - af_expected_sqrtm).norm()
    assert (adagf - adagf_expected_sqrtm).norm() < 1e-12, (
        adagf - adagf_expected_sqrtm
    ).norm()


def test_deformed_boson_backend_other_fn():
    """
    Validate a non-sqrt deformation (e.g., exponential suppression).
    """

    def deform(n):
        return np.exp(-0.5 * n)

    dim = 5
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=dim, deformation=deform)],
        customs=[],
    )
    cache = QutipOperatorCache(cfg)

    diag = Qobj(np.diag(deform(np.arange(dim))), dims=[[dim], [dim]])
    a = destroy(dim)

    af_expected = a * diag
    adagf_expected = diag * a.dag()

    af = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="af")
    )
    adagf = cache.local_operator(
        LocalOperatorRef(kind="boson", label="a", index=1, op_name="adagf")
    )

    assert (af - af_expected).norm() < 1e-12, (af - af_expected).norm()
    assert (adagf - adagf_expected).norm() < 1e-12, (adagf - adagf_expected).norm()


def test_operator_function_cos_qubit_backend():
    """
    cos(sigma_z) should match manual matrix cosine.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )
    ir = latex_to_ir(r"\cos(\sigma_{z,1})", cfg, t_name="t")
    cache = QutipOperatorCache(cfg)
    term = ir.terms[0]
    op = term_to_qobj(term, cache)

    sz = sigmaz()
    manual = Qobj(cosm(sz.full()), dims=sz.dims)
    assert (op - manual).norm() < 1e-12, (op - manual).norm()


def test_operator_function_exp_boson_backend():
    """
    exp(n) on a boson should match manual exp of the number operator.
    """
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=3)],
        customs=[],
    )
    ir = latex_to_ir(r"\exp(n_{1})", cfg, t_name="t")
    cache = QutipOperatorCache(cfg)
    term = ir.terms[0]
    op = term_to_qobj(term, cache)

    a = destroy(3)
    n_op = a.dag() * a
    manual = Qobj(expm(n_op.full()), dims=n_op.dims)
    assert (op - manual).norm() < 1e-12, (op - manual).norm()


def test_operator_function_with_time_backend():
    """
    exp(t * sigma_z) should produce a callable operator matching manual expm.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )
    ir = latex_to_ir(r"\exp(t \sigma_{z,1})", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 1
    Hk, fk = compiled.time_terms[0]
    assert Hk is None  # callable operator term

    sz = sigmaz()
    for tval in [0.0, 0.2, -0.3]:
        op = fk(tval, compiled.args)
        manual = Qobj(expm(tval * sz.full()), dims=sz.dims)
        assert (op - manual).norm() < 1e-12, (tval, (op - manual).norm())


def test_static_hamiltonian_with_time_independent_function_backend():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(0.5 \sigma_{z,1})", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, params={})
    assert isinstance(H, Qobj)
    sz = sigmaz()
    manual = Qobj(expm(0.5 * sz.full()), dims=sz.dims)
    assert (H - manual).norm() < 1e-12


def test_static_hamiltonian_with_mixed_time_function_backend():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(0.5 t \sigma_{z,1})", cfg, t_name="t")
    with pytest.raises(Exception):
        compile_static_hamiltonian_ir(ir, cfg, params={})


def test_static_hamiltonian_with_no_time_function_backend():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(2) \sigma_{z,1}", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(ir, cfg, params={})
    assert isinstance(H, Qobj)
    sz = sigmaz()
    manual = Qobj(np.exp(2) * sz.full(), dims=sz.dims)
    assert (H - manual).norm() < 1e-12


def test_time_dependent_hamiltonian_with_mixed_time_function_backend():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(0.5 t \sigma_{z,1})", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 1
    Hk, fk = compiled.time_terms[0]
    assert Hk is None

    sz = sigmaz()
    for tval in [0.0, 0.3, -0.4]:
        op = fk(tval, compiled.args)
        manual = Qobj(expm(0.5 * tval * sz.full()), dims=sz.dims)
        assert (op - manual).norm() < 1e-12


def test_time_dependent_hamiltonian_with_no_time_function_backend():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(2) \sigma_{z,1}", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is False
    H = compiled.H
    assert isinstance(H, Qobj)
    sz = sigmaz()
    manual = Qobj(np.exp(2) * sz.full(), dims=sz.dims)
    assert (H - manual).norm() < 1e-12


def test_static_jc_qutip():
    """
    End-to-end static Jaynes–Cummings Hamiltonian test.

    The Hamiltonian is

    .. math::

        H = \\omega_c \\hat{n}_{1}
            + \\frac{\\omega_q}{2} \\sigma_{z,1}
            + g (a_{1} \\sigma_{+,1} + a_{1}^{\\dagger} \\sigma_{-,1}).

    This function builds the same Hamiltonian explicitly in QuTiP and
    checks that the compiled result matches it.
    """
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

    # User-friendly names: no braces.
    params = {
        "omega_c": 1.5,
        "omega_q": 2.0,
        "g": 0.1,
    }

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is False, "JC test should be static."

    H_compiled = compile_static_hamiltonian_ir(ir, cfg, params)

    # Build reference JC Hamiltonian with the SAME tensor ordering:
    # subsystems: [qubit (dim=2), boson (dim=N)]
    a = destroy(N)
    adag = a.dag()
    n = adag * a

    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    Iq = qeye(2)
    Ic = qeye(N)

    H_c = params["omega_c"] * tensor(Iq, n)
    H_q = (params["omega_q"] / 2.0) * tensor(sz, Ic)
    H_int = params["g"] * (tensor(sp, a) + tensor(sm, adag))

    H_ref = H_c + H_q + H_int

    diff = (H_compiled - H_ref).norm()
    assert diff < 1e-10, f"Static JC Hamiltonian mismatch, norm diff = {diff}"


def test_static_two_qubit_coupling_qutip():
    """
    End-to-end static two-qubit coupling test.

    The Hamiltonian is

    .. math::

        H = J \\sigma_{x,1} \\sigma_{x,2}.
    """
    cfg = HilbertConfig(
        qubits=[
            QubitSpec(label="q", index=1),
            QubitSpec(label="q", index=2),
        ],
        bosons=[],
        customs=[],
    )

    H_latex = r"J \sigma_{x,1} \sigma_{x,2}"

    params = {"J": 0.7}

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is False

    H_compiled = compile_static_hamiltonian_ir(ir, cfg, params)

    sx = sigmax()
    H_ref = params["J"] * tensor(sx, sx)

    diff = (H_compiled - H_ref).norm()
    assert diff < 1e-10, f"Two-qubit coupling mismatch, norm diff = {diff}"


def test_time_dep_driven_qubit_qutip():
    """
    Time-dependent driven qubit Hamiltonian test.

    The Hamiltonian is

    .. math::

        H(t) = \\frac{\\omega_0}{2} \\sigma_{z,1}
               + A \\cos(\\omega t) \\sigma_{x,1}.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )

    H_latex = r"""
        \frac{\omega_0}{2} \sigma_{z,1}
        + A \cos(\omega t) \sigma_{x,1}
    """

    params = {
        "omega_0": 2.0,
        "A": 0.3,
        "omega": 1.5,
    }

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is True

    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params, t_name="t")

    assert compiled.time_dependent is True
    assert isinstance(compiled.H, list)
    assert len(compiled.H) == 2  # H0 + one [H1, f1]
    H0 = compiled.H0
    time_terms = compiled.time_terms

    # Static part should match (omega_0 / 2) sigma_z on the single qubit
    sz = sigmaz()
    H0_ref = (params["omega_0"] / 2.0) * tensor(sz)
    diff0 = (H0 - H0_ref).norm()
    assert diff0 < 1e-10, f"Driven qubit H0 mismatch, norm diff = {diff0}"

    assert len(time_terms) == 1
    H1, f1 = time_terms[0]

    sx = sigmax()
    H1_ref = tensor(sx)
    diff1 = (H1 - H1_ref).norm()
    assert diff1 < 1e-10, f"Driven qubit H1 mismatch, norm diff = {diff1}"

    # Check f1(t, args) numerically for a couple of t values.
    for t in [0.0, 0.7]:
        val = f1(t, compiled.args)
        expected = params["A"] * sp.cos(params["omega"] * t)
        err = abs(val - complex(expected))
        assert (
            err < 1e-10
        ), f"f1(t,args) mismatch at t={t}: got {val}, expected {expected}"


def test_time_dep_driven_cavity_cos_drive_qutip():
    """
    Time-dependent driven cavity test.

    The Hamiltonian is

    .. math::

        H(t) = \\omega_c \\hat{n}_1
               + A \\cos(\\omega_d t) (a_1 + a_1^{\\dagger}).
    """
    N = 4
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[],
    )

    H_latex = r"""
        \omega_c n_{1}
        + A \cos(\omega_d t) (a_{1} + a_{1}^{\dagger})
    """

    params = {
        "omega_c": 1.0,
        "A": 0.2,
        "omega_d": 0.7,
    }

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is True

    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params, t_name="t")
    assert compiled.time_dependent is True

    H0 = compiled.H0

    # Reference static part: omega_c * n_1
    a = destroy(N)
    adag = a.dag()
    n = adag * a
    H0_ref = params["omega_c"] * n
    diff0 = (H0 - H0_ref).norm()
    assert diff0 < 1e-10, f"Driven cavity H0 mismatch, norm diff = {diff0}"

    # We expect two time-dependent terms with operators a and adag.
    assert len(compiled.time_terms) == 2, compiled.time_terms

    ops = [term[0] for term in compiled.time_terms]
    # Classify each operator by distance to a and adag
    seen = set()
    for op in ops:
        dist_a = (op - a).norm()
        dist_adag = (op - adag).norm()
        if dist_a < 1e-10:
            seen.add("a")
        elif dist_adag < 1e-10:
            seen.add("adag")
        else:
            raise AssertionError("Unexpected time-dependent cavity operator.")

    assert seen == {"a", "adag"}, seen

    # All coefficient functions should be A cos(omega_d t)
    for _, fk in compiled.time_terms:
        for t in [0.0, 0.3]:
            val = fk(t, compiled.args)
            expected = params["A"] * sp.cos(params["omega_d"] * t)
            err = abs(val - complex(expected))
            assert (
                err < 1e-10
            ), f"Driven cavity f_k mismatch at t={t}: got {val}, expected {expected}"


def test_time_dep_two_shape_qubit_qutip():
    """
    Time-dependent qubit with two different envelope shapes.

    The Hamiltonian is

    .. math::

        H(t) = \\left[A \\cos(\\omega_1 t) + B e^{-\\gamma t}\\right]
               \\sigma_{x,1}.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )

    H_latex = r"""
        (A \cos(\omega_1 t) + B \exp(-\gamma t)) \sigma_{x,1}
    """

    params = {
        "A": 0.5,
        "omega_1": 1.0,
        "B": 0.3,
        "gamma": 0.2,
    }

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is True

    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params, t_name="t")

    assert compiled.time_dependent is True
    H0 = compiled.H0
    time_terms = compiled.time_terms

    # There is no purely static term here
    diff0 = H0.norm()
    assert diff0 < 1e-10, f"Expected H0 ~ 0, got norm {diff0}"

    # Two time-dependent terms, both with operator sigma_x
    assert len(time_terms) == 2, time_terms

    sx = sigmax()
    for Hk, _ in time_terms:
        diff = (Hk - sx).norm()
        assert (
            diff < 1e-10
        ), f"Time-dependent qubit operator mismatch, norm diff = {diff}"

    # Check that the two envelope shapes match {A cos(omega_1 t), B exp(-gamma t)}
    # as a multiset, without assuming ordering.
    for t in [0.0, 0.37]:
        vals = [complex(fk(t, compiled.args)) for _, fk in time_terms]
        vals_sorted = sorted(vals, key=lambda x: abs(x))

        expected_vals = [
            params["A"] * sp.cos(params["omega_1"] * t),
            params["B"] * sp.exp(-params["gamma"] * t),
        ]
        expected_sorted = sorted(
            [complex(ev) for ev in expected_vals],
            key=lambda x: abs(x),
        )

        for v, e in zip(vals_sorted, expected_sorted):
            err = abs(v - e)
            assert err < 1e-10, f"Envelope mismatch at t={t}: got {v}, expected {e}"


def test_time_dep_gaussian_pulse_qubit_qutip():
    """
    Time-dependent Gaussian pulse on a qubit.

    The Hamiltonian is

    .. math::

        H(t) = A e^{-(t / \\tau)^2} \\sigma_{x,1}.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )

    H_latex = r"A \exp(-(t / \tau)^2) \sigma_{x,1}"

    params = {
        "A": 1.0,
        "tau": 2.0,
    }

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is True

    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params, t_name="t")

    assert compiled.time_dependent is True
    H0 = compiled.H0
    time_terms = compiled.time_terms

    diff0 = H0.norm()
    assert diff0 < 1e-10, f"Expected H0 ~ 0 for Gaussian pulse, got norm {diff0}"

    assert len(time_terms) == 1, time_terms
    H1, f1 = time_terms[0]

    sx = sigmax()
    diff1 = (H1 - sx).norm()
    assert diff1 < 1e-10, f"Gaussian pulse H1 mismatch, norm diff = {diff1}"

    for t in [0.0, 0.5, 1.0]:
        val = f1(t, compiled.args)
        expected = params["A"] * sp.exp(-((t / params["tau"]) ** 2))
        err = abs(val - complex(expected))
        assert (
            err < 1e-10
        ), f"Gaussian pulse envelope mismatch at t={t}: got {val}, expected {expected}"


def test_static_two_qubit_two_mode_jc_beamsplitter_qutip():
    """
    Static Hamiltonian test with two qubits and two modes.

    The Hamiltonian contains two Jaynes–Cummings couplings and a
    beamsplitter coupling between the two modes. It is constructed
    explicitly in QuTiP and compared against the compiled result.
    """
    # Small cutoffs to keep the matrix size reasonable
    N1 = 2
    N2 = 3

    cfg = HilbertConfig(
        qubits=[
            QubitSpec(label="q", index=1),
            QubitSpec(label="q", index=2),
        ],
        bosons=[
            BosonSpec(label="a", index=1, cutoff=N1),
            BosonSpec(label="a", index=2, cutoff=N2),
        ],
        customs=[],
    )

    H_latex = r"""
        \omega_{c1} n_{1}
        + \omega_{c2} n_{2}
        + \frac{\omega_{q1}}{2} \sigma_{z,1}
        + \frac{\omega_{q2}}{2} \sigma_{z,2}
        + g_{1} (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})
        + g_{2} (a_{2} \sigma_{+,2} + a_{2}^{\dagger} \sigma_{-,2})
        + J (a_{1}^{\dagger} a_{2} + a_{2}^{\dagger} a_{1})
    """

    # User-friendly parameter names; backend aliases handle the braces.
    params = {
        "omega_c1": 1.0,
        "omega_c2": 1.1,
        "omega_q1": 0.9,
        "omega_q2": 1.2,
        "g_1": 0.05,
        "g_2": 0.07,
        "J": 0.02,
    }

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is False, "Two-qubit/two-mode JC+BS test should be static."

    H_compiled = compile_static_hamiltonian_ir(ir, cfg, params)

    # Build reference Hamiltonian with the SAME tensor ordering:
    #   [qubit1 (dim=2), qubit2 (dim=2), mode1 (dim=N1), mode2 (dim=N2)]
    a1 = destroy(N1)
    a2 = destroy(N2)
    adag1 = a1.dag()
    adag2 = a2.dag()
    n1 = adag1 * a1
    n2 = adag2 * a2

    sz = sigmaz()
    sp = sigmap()
    sm = sigmam()

    Iq = qeye(2)
    I1 = qeye(N1)
    I2 = qeye(N2)

    # Cavity energies
    H_cav = params["omega_c1"] * tensor(Iq, Iq, n1, I2) + params["omega_c2"] * tensor(
        Iq, Iq, I1, n2
    )

    # Qubit energies
    H_q = (params["omega_q1"] / 2.0) * tensor(sz, Iq, I1, I2) + (
        params["omega_q2"] / 2.0
    ) * tensor(Iq, sz, I1, I2)

    # JC couplings
    H_int1 = params["g_1"] * (tensor(sp, Iq, a1, I2) + tensor(sm, Iq, adag1, I2))
    H_int2 = params["g_2"] * (tensor(Iq, sp, I1, a2) + tensor(Iq, sm, I1, adag2))

    # Beamsplitter between the two modes
    H_bs = params["J"] * (tensor(Iq, Iq, adag1, a2) + tensor(Iq, Iq, a1, adag2))

    H_ref = H_cav + H_q + H_int1 + H_int2 + H_bs

    diff = (H_compiled - H_ref).norm()
    assert (
        diff < 1e-10
    ), f"Two-qubit/two-mode JC+BS Hamiltonian mismatch, norm diff = {diff}"


def test_static_collapse_qubit_boson_qutip():
    """
    Static collapse operators for a qubit and a boson mode.

    The collapse operators are

    .. math::

        c_1 = \\sqrt{\\kappa} a_1, \\quad
        c_2 = \\sqrt{\\gamma} \\sigma_{-,1}.
    """
    N = 3
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[],
    )

    c_ops_latex = [
        r"\sqrt{\kappa} a_{1}",
        r"\sqrt{\gamma} \sigma_{-,1}",
    ]

    params = {
        "kappa": 0.1,
        "gamma": 0.05,
    }

    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params, t_name="t"
    )

    assert td is False, "Static collapse test should have no time dependence."
    assert len(c_ops) == 2, c_ops

    C1, C2 = c_ops

    # Reference ops
    a = destroy(N)
    sm = sigmam()
    Iq = qeye(2)
    Ic = qeye(N)

    C1_ref = (params["kappa"] ** 0.5) * tensor(Iq, a)
    C2_ref = (params["gamma"] ** 0.5) * tensor(sm, Ic)

    diff1 = (C1 - C1_ref).norm()
    diff2 = (C2 - C2_ref).norm()

    assert diff1 < 1e-10, f"C1 collapse mismatch, norm diff = {diff1}"
    assert diff2 < 1e-10, f"C2 collapse mismatch, norm diff = {diff2}"


def test_time_dep_collapse_qubit_qutip():
    """
    Time-dependent collapse operator on a qubit.

    The collapse operator is

    .. math::

        c(t) = \\sqrt{\\gamma} e^{-t/2} \\sigma_{-,1}.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )

    c_ops_latex = [r"\sqrt{\gamma} \exp(-t/2) \sigma_{-,1}"]

    params = {
        "gamma": 0.2,
    }

    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params, t_name="t"
    )

    assert td is True
    assert len(c_ops) == 1
    C0, f = c_ops[0]

    sm = sigmam()
    diffC = (C0 - sm).norm()
    assert (
        diffC < 1e-10
    ), f"Time-dependent collapse operator mismatch, norm diff = {diffC}"

    for t in [0.0, 0.4, 1.0]:
        val = f(t, args)
        expected = (params["gamma"] ** 0.5) * sp.exp(-t / 2)
        err = abs(val - complex(expected))
        assert (
            err < 1e-10
        ), f"Collapse envelope mismatch at t={t}: got {val}, expected {expected}"


def test_open_system_rabi_like_qutip():
    """
    Rabi-like open-system test with one qubit.

    The Hamiltonian is

    .. math::

        H(t) = \\frac{\\omega_0}{2} \\sigma_{z,1}
               + A \\cos(\\omega t) \\sigma_{x,1}

    and the collapse operator is

    .. math::

        c = \\sqrt{\\gamma} \\sigma_{-,1}.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )

    H_latex = r"""
        \frac{\omega_0}{2} \sigma_{z,1}
        + A \cos(\omega t) \sigma_{x,1}
    """

    c_ops_latex = [r"\sqrt{\gamma} \sigma_{-,1}"]

    params = {
        "omega_0": 2.0,
        "A": 0.3,
        "omega": 1.5,
        "gamma": 0.05,
    }

    model = compile_open_system_from_latex(
        H_latex, c_ops_latex, cfg, params, t_name="t"
    )

    # Hamiltonian should be time-dependent: a list [H0, [H1, f1]]
    assert model.time_dependent is True
    assert isinstance(model.H, list)
    assert len(model.H) == 2  # H0 + one [H1, f1]

    # Collapse ops: one static Qobj
    assert len(model.c_ops) == 1
    C = model.c_ops[0]
    assert isinstance(C, Qobj)

    sm = sigmam()
    C_ref = (params["gamma"] ** 0.5) * sm
    diff = (C - C_ref).norm()
    assert diff < 1e-10, f"Open-system Rabi collapse mismatch, norm diff = {diff}"


def test_open_system_qubit_boson_custom_qutip():
    """
    Open system with qubit, boson and custom spin-1 subsystem.

    This test checks a Hamiltonian with qubit, boson, and custom
    couplings, as well as static and time-dependent terms, together
    with three static collapse channels.
    """
    import numpy as np

    N = 3
    dim_c = 3
    sqrt2 = np.sqrt(2.0)

    # Spin-1 operators.
    Jp = Qobj(
        np.array(
            [
                [0.0, sqrt2, 0.0],
                [0.0, 0.0, sqrt2],
                [0.0, 0.0, 0.0],
            ],
            dtype=complex,
        )
    )
    Jm = Jp.dag()
    Jz = Qobj(np.diag([1.0, 0.0, -1.0]))
    Jx = 0.5 * (Jp + Jm)

    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[
            CustomSpec(
                label="c",
                index=1,
                dim=dim_c,
                operators={
                    "Jz": Jz,
                    "Jx": Jx,
                    "Jp": Jp,
                    "Jm": Jm,
                },
            )
        ],
    )

    H_latex = r"""
        \omega_c n_{1}
        + \frac{\omega_q}{2} \sigma_{z,1}
        + \omega_J J_{z,1}
        + g_{bq} \left( a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1} \right)
        + g_{bc} \left( a_{1} J_{+,1} + a_{1}^{\dagger} J_{-,1} \right)
        + A \cos(\omega_d t) J_{x,1}
    """

    c_ops_latex = [
        r"\sqrt{\kappa} a_{1}",
        r"\sqrt{\gamma_q} \sigma_{-,1}",
        r"\sqrt{\gamma_J} J_{-,1}",
    ]

    params = {
        "omega_c": 1.0,
        "omega_q": 0.7,
        "omega_J": 0.3,
        "g_bq": 0.05,
        "g_bc": -0.02,
        "A": 0.5,
        "omega_d": 0.9,
        "kappa": 0.1,
        "gamma_q": 0.05,
        "gamma_J": 0.02,
    }

    model = compile_open_system_from_latex(
        H_latex, c_ops_latex, cfg, params, t_name="t"
    )

    assert model.time_dependent is True

    # H should be in QuTiP list form: [H0, [H1, f1]].
    H_list = model.H
    assert isinstance(H_list, list), H_list
    assert len(H_list) == 2, H_list

    H0 = H_list[0]
    H1, f1 = H_list[1]

    # Reference H0.
    a = destroy(N)
    adag = a.dag()
    n = a.dag() * a
    sz = sigmaz()
    splus = sigmap()
    sminus = sigmam()

    Iq = qeye(2)
    Ic = qeye(N)
    Icust = qeye(dim_c)

    H_cav = params["omega_c"] * tensor(Iq, n, Icust)
    H_qubit = (params["omega_q"] / 2.0) * tensor(sz, Ic, Icust)
    H_custom = params["omega_J"] * tensor(Iq, Ic, Jz)
    H_bq = params["g_bq"] * (tensor(splus, a, Icust) + tensor(sminus, adag, Icust))
    H_bc = params["g_bc"] * (tensor(Iq, a, Jp) + tensor(Iq, adag, Jm))

    H0_ref = H_cav + H_qubit + H_custom + H_bq + H_bc

    diff0 = (H0 - H0_ref).norm()
    assert (
        diff0 < 1e-10
    ), f"Open-system qubit+boson+custom H0 mismatch, norm diff = {diff0}"

    # Time-dependent Jx drive.
    H1_ref = tensor(Iq, Ic, Jx)
    diff1 = (H1 - H1_ref).norm()
    assert diff1 < 1e-10, (
        "Open-system Jx drive operator mismatch, " f"norm diff = {diff1}"
    )

    for t in [0.0, 0.3]:
        val = f1(t, model.args)
        expected = params["A"] * sp.cos(params["omega_d"] * t)
        err = abs(val - complex(expected))
        assert err < 1e-10, (
            f"Open-system Jx drive envelope mismatch at t={t}: "
            f"got {val}, expected {expected}"
        )

    # Collapse operators: same as in the static collapse test, but coming
    # from the open-system compiler.
    assert len(model.c_ops) == 3, model.c_ops

    a_full = tensor(Iq, a, Icust)
    sm_full = tensor(sigmam(), Ic, Icust)
    Jm_full = tensor(Iq, Ic, Jm)

    C1, C2, C3 = model.c_ops

    C1_ref = (params["kappa"] ** 0.5) * a_full
    C2_ref = (params["gamma_q"] ** 0.5) * sm_full
    C3_ref = (params["gamma_J"] ** 0.5) * Jm_full

    for C, C_ref, label in [
        (C1, C1_ref, "a collapse"),
        (C2, C2_ref, "qubit collapse"),
        (C3, C3_ref, "custom collapse"),
    ]:
        diffC = (C - C_ref).norm()
        assert (
            diffC < 1e-10
        ), f"Open-system {label} operator mismatch, norm diff = {diffC}"


def test_static_qubit_boson_custom_diagonal_qutip():
    """
    Static diagonal Hamiltonian with qubit, boson and custom subsystem.

    The Hamiltonian is diagonal in the basis formed by a qubit, a
    boson mode and a spin-1 custom subsystem, and serves as a check of
    tensor ordering.
    """
    import numpy as np

    N = 3
    dim_c = 3

    # Spin-1 Jz on the custom subsystem (dimension 3).
    Jz = Qobj(np.diag([1.0, 0.0, -1.0]))

    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[CustomSpec(label="c", index=1, dim=dim_c, operators={"Jz": Jz})],
    )

    H_latex = r"""
        \omega_c n_{1}
        + \frac{\omega_q}{2} \sigma_{z,1}
        + \omega_J J_{z,1}
    """

    params = {
        "omega_c": 1.2,
        "omega_q": 0.7,
        "omega_J": -0.3,
    }

    # Compile via LaTeX → IR → QuTiP backend.
    H_compiled = compile_static_hamiltonian_from_latex(H_latex, cfg, params)

    # Reference Hamiltonian with explicit tensor ordering: [qubit, boson, custom].
    a = destroy(N)
    n = a.dag() * a
    sz = sigmaz()

    Iq = qeye(2)
    Ic = qeye(N)
    Icust = qeye(dim_c)

    H_cav = params["omega_c"] * tensor(Iq, n, Icust)
    H_qubit = (params["omega_q"] / 2.0) * tensor(sz, Ic, Icust)
    H_custom = params["omega_J"] * tensor(Iq, Ic, Jz)

    H_ref = H_cav + H_qubit + H_custom

    diff = (H_compiled - H_ref).norm()
    assert diff < 1e-10, (
        f"Static qubit+boson+custom diagonal Hamiltonian mismatch, "
        f"norm diff = {diff}"
    )


def test_static_qubit_boson_custom_couplings_qutip():
    """
    Static Hamiltonian with qubit–boson and boson–custom couplings.

    This test constructs a Hamiltonian containing qubit–boson and
    boson–custom couplings and compares it against the compiled
    result.
    """
    import numpy as np

    N = 3
    dim_c = 3
    sqrt2 = np.sqrt(2.0)

    # Spin-1 ladder operators on the custom subsystem.
    Jp = Qobj(
        np.array(
            [
                [0.0, sqrt2, 0.0],
                [0.0, 0.0, sqrt2],
                [0.0, 0.0, 0.0],
            ],
            dtype=complex,
        )
    )
    Jm = Jp.dag()

    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[
            CustomSpec(
                label="c",
                index=1,
                dim=dim_c,
                operators={
                    "Jp": Jp,
                    "Jm": Jm,
                },
            )
        ],
    )

    H_latex = r"""
        g_{bq} \left( a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1} \right)
        + g_{bc} \left( a_{1} J_{+,1} + a_{1}^{\dagger} J_{-,1} \right)
    """

    params = {
        "g_bq": 0.05,
        "g_bc": -0.02,
    }

    H_compiled = compile_static_hamiltonian_from_latex(H_latex, cfg, params)

    # Reference construction.
    a = destroy(N)
    adag = a.dag()
    sp = sigmap()
    sm = sigmam()

    Iq = qeye(2)
    Icust = qeye(dim_c)

    H_bq = params["g_bq"] * (tensor(sp, a, Icust) + tensor(sm, adag, Icust))
    H_bc = params["g_bc"] * (tensor(Iq, a, Jp) + tensor(Iq, adag, Jm))

    H_ref = H_bq + H_bc

    diff = (H_compiled - H_ref).norm()
    assert diff < 1e-10, (
        f"Static qubit–boson–custom coupling Hamiltonian mismatch, "
        f"norm diff = {diff}"
    )


def test_time_dep_custom_drive_qubit_boson_qutip():
    """
    Time-dependent custom-subsystem drive with qubit and boson present.

    The Hamiltonian includes static qubit, boson and custom terms and a
    time-dependent drive on the custom subsystem.
    """
    import numpy as np

    N = 3
    dim_c = 3
    sqrt2 = np.sqrt(2.0)

    # Spin-1 operators on the custom subsystem.
    Jp = Qobj(
        np.array(
            [
                [0.0, sqrt2, 0.0],
                [0.0, 0.0, sqrt2],
                [0.0, 0.0, 0.0],
            ],
            dtype=complex,
        )
    )
    Jm = Jp.dag()
    Jz = Qobj(np.diag([1.0, 0.0, -1.0]))
    Jx = 0.5 * (Jp + Jm)

    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[
            CustomSpec(
                label="c",
                index=1,
                dim=dim_c,
                operators={"Jz": Jz, "Jx": Jx},
            )
        ],
    )

    H_latex = r"""
        \omega_c n_{1}
        + \frac{\omega_q}{2} \sigma_{z,1}
        + \omega_J J_{z,1}
        + A \cos(\omega_d t) J_{x,1}
    """

    params = {
        "omega_c": 1.0,
        "omega_q": 0.7,
        "omega_J": 0.3,
        "A": 0.5,
        "omega_d": 0.9,
    }

    # Correct pipeline: LaTeX -> IR -> time-dependent compilation
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params, t_name="t")

    assert compiled.time_dependent is True

    H0 = compiled.H0
    time_terms = compiled.time_terms
    assert len(time_terms) == 1, time_terms

    # Reference static part.
    a = destroy(N)
    n = a.dag() * a
    sz = sigmaz()

    Iq = qeye(2)
    Ic = qeye(N)
    Icust = qeye(dim_c)

    H0_ref = (
        params["omega_c"] * tensor(Iq, n, Icust)
        + (params["omega_q"] / 2.0) * tensor(sz, Ic, Icust)
        + params["omega_J"] * tensor(Iq, Ic, Jz)
    )

    diff0 = (H0 - H0_ref).norm()
    assert (
        diff0 < 1e-10
    ), f"Time-dependent custom drive H0 mismatch, norm diff = {diff0}"

    # Time-dependent term: Jx on the custom subsystem with A cos(omega_d t).
    H1, f1 = time_terms[0]
    H1_ref = tensor(Iq, Ic, Jx)
    diff1 = (H1 - H1_ref).norm()
    assert (
        diff1 < 1e-10
    ), f"Time-dependent custom drive H1 mismatch, norm diff = {diff1}"

    for t in [0.0, 0.3]:
        val = f1(t, compiled.args)
        expected = params["A"] * sp.cos(params["omega_d"] * t)
        err = abs(val - complex(expected))
        assert err < 1e-10, (
            f"Time-dependent custom drive scalar envelope mismatch at t={t}: "
            f"got {val}, expected {expected}"
        )


def test_static_collapse_qubit_boson_custom_qutip():
    """
    Static collapse operators for qubit, boson and custom subsystem.

    The collapse operators are

    .. math::

        c_1 = \\sqrt{\\kappa} a_1, \\quad
        c_2 = \\sqrt{\\gamma_q} \\sigma_{-,1}, \\quad
        c_3 = \\sqrt{\\gamma_J} J_{-,1}.
    """
    import numpy as np

    N = 3
    dim_c = 3
    sqrt2 = np.sqrt(2.0)

    # Spin-1 lowering operator on the custom subsystem.
    Jp = Qobj(
        np.array(
            [
                [0.0, sqrt2, 0.0],
                [0.0, 0.0, sqrt2],
                [0.0, 0.0, 0.0],
            ],
            dtype=complex,
        )
    )
    Jm = Jp.dag()

    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[BosonSpec(label="a", index=1, cutoff=N)],
        customs=[
            CustomSpec(
                label="c",
                index=1,
                dim=dim_c,
                operators={"Jm": Jm},
            )
        ],
    )

    c_ops_latex = [
        r"\sqrt{\kappa} a_{1}",
        r"\sqrt{\gamma_q} \sigma_{-,1}",
        r"\sqrt{\gamma_J} J_{-,1}",
    ]

    params = {
        "kappa": 0.1,
        "gamma_q": 0.05,
        "gamma_J": 0.02,
    }

    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params, t_name="t"
    )

    assert td is False
    assert len(c_ops) == 3, c_ops

    a = destroy(N)

    Iq = qeye(2)
    Ic = qeye(N)
    Icust = qeye(dim_c)

    a_full = tensor(Iq, a, Icust)
    sm_full = tensor(sigmam(), Ic, Icust)
    Jm_full = tensor(Iq, Ic, Jm)

    C1, C2, C3 = c_ops

    C1_ref = (params["kappa"] ** 0.5) * a_full
    C2_ref = (params["gamma_q"] ** 0.5) * sm_full
    C3_ref = (params["gamma_J"] ** 0.5) * Jm_full

    for C, C_ref, label in [
        (C1, C1_ref, "a collapse"),
        (C2, C2_ref, "qubit collapse"),
        (C3, C3_ref, "custom collapse"),
    ]:
        diff = (C - C_ref).norm()
        assert diff < 1e-10, f"Static {label} operator mismatch, norm diff = {diff}"


def test_build_time_dep_callable_braced_param_qutip():
    """
    Test _build_time_dep_term_callable with a braced parameter.

    This test ensures that a parameter written as ``\\omega_{d}`` is
    correctly resolved via the alias ``"omega_d"`` in the arguments
    dictionary.
    """
    t = sp.Symbol("t")
    omega_d = sp.Symbol("omega_{d}")
    scalar = omega_d * t

    f_k, param_syms, alias_lists = _build_time_dep_term_callable(scalar, t_name="t")

    # One parameter, with braced name
    assert len(param_syms) == 1
    assert param_syms[0].name == "omega_{d}", param_syms

    # Alias list for omega_{d} should contain 'omega_d'
    aliases = alias_lists[0]
    assert any(key == "omega_d" for key in aliases), aliases

    args = {"omega_d": 2.0}
    val = f_k(1.5, args)
    expected = 2.0 * 1.5
    assert abs(val - expected) < 1e-12, (val, expected)


def test_build_time_dep_callable_no_params_qutip():
    """
    Test _build_time_dep_term_callable with no parameters.

    This test uses the scalar envelope :math:`\\sin(t)` and checks that
    it compiles to a function that depends only on time.
    """
    t = sp.Symbol("t")
    scalar = sp.sin(t)

    f_k, param_syms, alias_lists = _build_time_dep_term_callable(scalar, t_name="t")

    assert param_syms == [], param_syms
    assert alias_lists == [], alias_lists

    for t_val in [0.0, 0.5, 1.0]:
        val = f_k(t_val, args={})
        expected = float(sp.sin(t_val))
        assert abs(val - expected) < 1e-10, (t_val, val, expected)


def test_open_system_static_H_time_dep_collapse_qutip():
    """
    Open-system test with static Hamiltonian and time-dependent collapse.

    The Hamiltonian is static and the collapse operator is of the form

    .. math::

        c(t) = \\sqrt{\\gamma} e^{-t/2} \\sigma_{-,1}.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )

    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    c_ops_latex = [r"\sqrt{\gamma} \exp(-t/2) \sigma_{-,1}"]

    params = {
        "omega_0": 1.0,
        "gamma": 0.3,
    }

    model = compile_open_system_from_latex(
        H_latex, c_ops_latex, cfg, params, t_name="t"
    )

    # H should be static
    assert isinstance(model.H, Qobj), type(model.H)

    # One time-dependent collapse channel [C0, f]
    assert len(model.c_ops) == 1
    C0, f = model.c_ops[0]

    sm = sigmam()
    diffC = (C0 - sm).norm()
    assert diffC < 1e-10, f"C0 mismatch, norm diff = {diffC}"

    for t_val in [0.0, 0.4, 1.0]:
        val = f(t_val, model.args)
        expected = (params["gamma"] ** 0.5) * sp.exp(-t_val / 2)
        err = abs(val - complex(expected))
        assert err < 1e-10, (
            f"Time-dependent collapse envelope mismatch at t={t_val}: "
            f"got {val}, expected {expected}"
        )

    assert model.time_dependent is True


def test_static_collapse_two_qubit_sum_single_channel_qutip():
    """
    Static two-qubit collapse channel formed as a sum.

    The collapse operator is

    .. math::

        c = \\sqrt{\\gamma_1} \\sigma_{-,1}
            + \\sqrt{\\gamma_2} \\sigma_{-,2}.
    """
    cfg = HilbertConfig(
        qubits=[
            QubitSpec(label="q", index=1),
            QubitSpec(label="q", index=2),
        ],
        bosons=[],
        customs=[],
    )

    c_ops_latex = [
        r"\sqrt{\gamma_1} \sigma_{-,1} + \sqrt{\gamma_2} \sigma_{-,2}",
    ]

    params = {
        "gamma_1": 0.3,
        "gamma_2": 0.7,
    }

    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params, t_name="t"
    )

    assert td is False
    assert len(c_ops) == 1
    C = c_ops[0]

    sm = sigmam()
    identity = qeye(2)

    C_ref = (params["gamma_1"] ** 0.5) * tensor(sm, identity) + (
        params["gamma_2"] ** 0.5
    ) * tensor(identity, sm)

    diff = (C - C_ref).norm()
    assert diff < 1e-10, f"Static two-qubit collapse mismatch, norm diff = {diff}"


def test_param_alias_priority_qutip():
    """
    Test that parameter alias priority resolves the first matching key.

    This ensures that a symbol such as :math:`\\omega_{c}` resolves to
    ``"omega_c"`` instead of ``"omega_c1"`` when both keys are present
    in the parameter dictionary.
    """
    sym = sp.Symbol("omega_{c}")
    params = {
        "omega_c": 1.0,
        "omega_c1": 2.0,
    }

    key, val = _lookup_param_name(sym, params)
    assert key == "omega_c", key
    assert val == 1.0, val


def test_param_alias_missing_weird_name_qutip():
    """
    Test that missing aliases for unusual symbol names raise clearly.

    A symbol name such as ``omega_{c*1*foo}`` without a matching key in
    the parameter dictionary should trigger a clear
    :class:`DSLValidationError`.
    """
    sym = sp.Symbol("omega_{c*1*foo}")
    params = {"omega_c": 1.0}

    try:
        _lookup_param_name(sym, params)
    except DSLValidationError as exc:
        msg = str(exc)
        assert "omega_{c*1*foo}" in msg or "c*1*foo" in msg, msg
    else:
        raise AssertionError("Expected DSLValidationError for missing param alias.")


def test_time_dep_collapse_sum_rejected_qutip():
    """
    Test that time-dependent collapse sums are rejected as single channels.

    A collapse operator given as a sum of two time-dependent monomials
    in a single LaTeX string must raise :class:`DSLValidationError`.
    """
    cfg = HilbertConfig(
        qubits=[
            QubitSpec(label="q", index=1),
            QubitSpec(label="q", index=2),
        ],
        bosons=[],
        customs=[],
    )

    c_ops_latex = [
        (
            r"\sqrt{\gamma_1} \exp(-t/2) \sigma_{-,1}"
            r" + \sqrt{\gamma_2} \exp(-t/2) \sigma_{-,2}"
        )
    ]
    params = {
        "gamma_1": 0.3,
        "gamma_2": 0.7,
    }

    try:
        compile_collapse_ops_from_latex(c_ops_latex, cfg, params, t_name="t")
    except DSLValidationError as exc:
        msg = str(exc)
        assert "time-dependent collapse" in msg or "single monomial" in msg, msg
    else:
        raise AssertionError(
            "Expected DSLValidationError for time-dependent collapse sum in one "
            "channel."
        )


def test_scalar_envelope_with_supported_function_qutip():
    """
    Test scalar envelope using a supported function.

    This test uses an exponential envelope :math:`A e^{-t}` and checks
    that it compiles and evaluates correctly.
    """
    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=1)],
        bosons=[],
        customs=[],
    )

    H_latex = r"A \, \exp(-t) \, \sigma_{x,1}"
    params = {"A": 1.0}

    ir = latex_to_ir(H_latex, cfg, t_name="t")
    assert ir.has_time_dep is True

    compiled = compile_time_dependent_hamiltonian_ir(
        ir, cfg, params, t_name="t", time_symbols=None
    )
    assert compiled.time_dependent is True
    assert len(compiled.H) == 2  # H0 + one [H1, f]
    H1, f = compiled.time_terms[0]
    for t in [0.0, 0.5, 1.0]:
        val = f(t, compiled.args)
        expected = params["A"] * sp.exp(-t)
        assert abs(val - complex(expected)) < 1e-10, (t, val, expected)


def test_operator_function_static_op_time_scalar_backend():
    """
    exp(t) * exp(0.5 sigma_z) should yield static operator with time envelope.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\exp(t) \exp(0.5 \sigma_{z,1})", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 1
    Hk, fk = compiled.time_terms[0]
    assert Hk is not None

    sz = sigmaz()
    H_expected = Qobj(expm(0.5 * sz.full()), dims=sz.dims)
    assert (Hk - H_expected).norm() < 1e-12
    for tval in [0.0, 0.7]:
        val = fk(tval, compiled.args)
        expected = np.exp(tval)
        assert abs(val - expected) < 1e-12


def test_time_symbols_multiple_backend():
    """
    Multiple time-like symbols in scalar envelopes should trigger time dependence.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(
        r"A \cos(\omega t) \sigma_{x,1} + B \sin(s) \sigma_{x,1}",
        cfg,
        t_name="t",
        time_symbols=("t", "s"),
    )
    compiled = compile_time_dependent_hamiltonian_ir(
        ir,
        cfg,
        params={"A": 0.5, "omega": 1.0, "B": 0.2},
        t_name="t",
        time_symbols=("t", "s"),
    )
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 2


def test_time_symbols_operator_function_backend():
    """
    exp((t + s) n1) with summed scalar inside the operator function should be
    rejected.
    """
    cfg = HilbertConfig(
        bosons=[BosonSpec(label="a", index=1, cutoff=2)], qubits=[], customs=[]
    )
    with pytest.raises(DSLValidationError):
        latex_to_ir(
            r"\exp((t + s) n_{1})",
            cfg,
            t_name="t",
            time_symbols=("t", "s"),
        )


def test_time_dependent_collapse_zero_scalar_with_extra_time_symbol_backend():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    c_ops_latex = [r"\sqrt{\gamma} \exp(-t/2 - s) \sigma_{-,1}"]
    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params={"gamma": 0.0}, t_name="t", time_symbols=("t", "s")
    )
    if c_ops:
        C0, f = c_ops[0]
        assert abs(f(0.0, {"gamma": 0.0, "s": 0.0})) < 1e-12
    else:
        assert td is False


def test_parameter_alias_warning(caplog):
    """
    Ambiguous param names should emit a warning and pick first match.
    """
    caplog.set_level("WARNING", logger="latex_parser.backend_qutip")
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"\omega_{c} \sigma_{z,1}", cfg, t_name="t")
    H = compile_static_hamiltonian_ir(
        ir, cfg, params={"omega_{c}": 1.0, "omega_c": 2.0}
    )
    assert isinstance(H, Qobj)
    assert any("matched multiple keys" in rec.message for rec in caplog.records)


def test_time_dependent_collapse_zero_scalar_skips_backend():
    """
    Time-dependent collapse with zeroed scalar should be skipped gracefully.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    c_ops_latex = [r"\sqrt{\gamma} \exp(-t/2) \sigma_{-,1}"]
    c_ops, args, td = compile_collapse_ops_from_latex(
        c_ops_latex, cfg, params={"gamma": 0.0}, t_name="t"
    )
    if c_ops:
        C0, f = c_ops[0]
        assert abs(f(0.0, args)) < 1e-12
    else:
        assert td is False
    assert args["gamma"] == 0.0


def test_readme_quick_start_smoke_backend():
    """
    Smoke test mirroring the README quick-start snippet.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \cos(\omega t) \sigma_{x,1} + \cos(\sigma_{z,1})"
    compiled = compile_time_dependent_hamiltonian_ir(
        latex_to_ir(H_latex, cfg, t_name="t"),
        cfg,
        params={"A": 0.5, "omega": 1.0},
        t_name="t",
    )
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 1


def test_backend_parity_static_qubit_numpy_jax():
    """
    Static qubit Hamiltonian should have matching shapes/dims across backends.
    """
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    params = {"omega_0": 1.0}
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])

    # QuTiP
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    H_qutip = compile_static_hamiltonian_ir(ir, cfg, params)

    # NumPy
    import latex_parser.backend_numpy as backend_numpy

    model_np = backend_numpy.compile_open_system_from_latex(
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=[],
        t_name="t",
    )

    assert H_qutip.dims == [[2], [2]]
    assert model_np.H.shape == (2, 2)


def test_operator_function_mixed_time_rejected_static_backend():
    """
    exp((A + t) n1) should be rejected: sums inside operator functions are disallowed.
    """
    cfg = HilbertConfig(
        bosons=[BosonSpec(label="a", index=1, cutoff=2)], qubits=[], customs=[]
    )
    with pytest.raises(DSLValidationError):
        latex_to_ir(r"\exp((A + t) n_{1})", cfg, t_name="t")


def test_operator_function_mixed_time_split_backend():
    """
    exp(t) * exp(n1) should compile with static op and time envelope.
    """
    cfg = HilbertConfig(
        bosons=[BosonSpec(label="a", index=1, cutoff=2)], qubits=[], customs=[]
    )
    ir = latex_to_ir(r"\exp(t) \exp(n_{1})", cfg, t_name="t")
    compiled = compile_time_dependent_hamiltonian_ir(ir, cfg, params={}, t_name="t")
    assert compiled.time_dependent is True
    assert len(compiled.time_terms) == 1
    Hk, fk = compiled.time_terms[0]
    a = destroy(2)
    n_op = a.dag() * a
    manual = Qobj(expm(n_op.full()), dims=n_op.dims)
    assert (Hk - manual).norm() < 1e-12
    for tval in [0.0, 0.4]:
        assert abs(fk(tval, compiled.args) - np.exp(tval)) < 1e-12


def test_backend_parity_static_collapse_numpy():
    """
    Static collapse operators should convert cleanly to NumPy.
    """
    H_latex = r"\omega_c n_{1}"
    c_ops_latex = [r"\sqrt{\kappa} a_{1}"]
    params = {"omega_c": 1.0, "kappa": 0.2}
    cfg = HilbertConfig(
        qubits=[], bosons=[BosonSpec(label="a", index=1, cutoff=3)], customs=[]
    )

    model_q = compile_open_system_from_latex(
        H_latex, c_ops_latex, cfg, params, t_name="t"
    )
    assert len(model_q.c_ops) == 1
    Cq = model_q.c_ops[0]
    assert Cq.dims == [[3], [3]]

    import latex_parser.backend_numpy as backend_numpy

    model_np = backend_numpy.compile_open_system_from_latex(
        H_latex=H_latex, params=params, config=cfg, c_ops_latex=c_ops_latex, t_name="t"
    )
    assert len(model_np.c_ops) == 1
    Cnp = model_np.c_ops[0]
    assert Cnp.shape == (3, 3)
    assert Cnp.dtype.kind in {"f", "c"}
