import os

import jax
import jax.numpy as jnp
import pytest
import sympy as sp

from latex_parser.backend_jax import (
    JaxOperatorCache,
    _param_aliases,
    _term_to_jax_static,
    compile_static_hamiltonian_ir,
)
from latex_parser.dsl import HilbertConfig, QubitSpec
from latex_parser.ir import Term, latex_to_ir

# Keep JAX on CPU for determinism
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_platforms", "cpu")


def _build_parametric_hamiltonian(ir, cfg, *, dtype=jnp.complex128):
    """
    Return a JAX-friendly function H(params) assembling the Hamiltonian
    as a linear combination of fixed operator matrices and JAX-evaluable
    scalar coefficients.
    """
    cache = JaxOperatorCache(cfg, dtype=dtype)
    pieces = []

    for term in ir.terms:
        # Operator part (independent of scalars)
        op_only = Term(scalar_expr=sp.Integer(1), ops=term.ops)
        op_mat = _term_to_jax_static(op_only, cache, params={}, dtype=dtype)

        free_syms = sorted(term.scalar_expr.free_symbols, key=lambda s: s.name)
        sym_names = [s.name for s in free_syms]

        if free_syms:
            coeff_fn = sp.lambdify(free_syms, term.scalar_expr, modules="jax")
        else:
            const_val = jnp.asarray(term.scalar_expr, dtype=dtype)

            def coeff_fn():  # type: ignore
                return const_val

        def scalar_func(params, sym_names=sym_names, coeff_fn=coeff_fn):
            if sym_names:
                args = []
                for name in sym_names:
                    aliases = _param_aliases(name)
                    matched = None
                    for key in aliases:
                        if key in params:
                            matched = params[key]
                            break
                    if matched is None:
                        raise KeyError(
                            f"Missing value for parameter; tried aliases {aliases}. "
                            f"Available keys: {sorted(params.keys())}"
                        )
                    args.append(matched)
                val = coeff_fn(*args)
            else:
                val = coeff_fn()
            return jnp.asarray(val, dtype=dtype)

        pieces.append((scalar_func, op_mat))

    def assemble(params):
        H = None
        for coeff_fn, op in pieces:
            coeff = coeff_fn(params)
            contrib = coeff * op
            H = contrib if H is None else H + contrib
        if H is None:
            H = jnp.zeros_like(cache.global_identity)
        return H

    return assemble


def test_jax_ising_param_optimization():
    """
    Optimize coupling strengths and a global angle in a 3-qubit Ising ZZ chain
    using JAX autodiff to reduce the ground-state energy.
    """

    cfg = HilbertConfig(
        qubits=[QubitSpec(label="q", index=i) for i in (1, 2, 3)], bosons=[], customs=[]
    )
    # User intent: exp(-i phi_k ZZ) for two edges. The DSL does not allow
    # operator functions on two-body products, so we use the identity
    # exp(-i phi ZZ) = cos(phi) I - i sin(phi) ZZ.
    # Build base ZZ operators from a simple linear IR.
    ir_linear = latex_to_ir(
        r"\sigma_{z,1} \sigma_{z,2} + \sigma_{z,2} \sigma_{z,3}", cfg, t_name="t"
    )

    # Sanity: compile works with fixed params on linear form (non-differentiable path)
    _ = compile_static_hamiltonian_ir(
        ir_linear, cfg, params={"J_12": 0.5, "J_23": -0.3, "phi_1": 0.2, "phi_2": -0.1}
    )

    cache = JaxOperatorCache(cfg, dtype=jnp.complex128)
    term_ops = []
    for term in ir_linear.terms:
        op_only = Term(scalar_expr=sp.Integer(1), ops=term.ops)
        term_ops.append(
            _term_to_jax_static(op_only, cache, params={}, dtype=jnp.complex128)
        )

    def H_fn(params):
        Js = [params["J_12"], params["J_23"]]
        phis = [params["phi_1"], params["phi_2"]]
        H = None
        for J_val, phi_val, op_mat in zip(Js, phis, term_ops):
            cos_phi = jnp.cos(phi_val)
            sin_phi = jnp.sin(phi_val)
            identity = cache.global_identity
            exp_op = cos_phi * identity - 1j * sin_phi * op_mat
            contrib = J_val * exp_op
            H = contrib if H is None else H + contrib
        if H is None:
            H = jnp.zeros_like(cache.global_identity)
        return H

    @jax.jit
    def ground_energy(params):
        H = H_fn(params)
        evals = jnp.linalg.eigvalsh(H)
        return evals[0].real

    # Native reference: couplings = 1, angles = 0 (exp(0)=1)
    ref_params = {
        "J_12": jnp.array(1.0),
        "J_23": jnp.array(1.0),
        "phi_1": jnp.array(0.0),
        "phi_2": jnp.array(0.0),
    }
    energy_ref = ground_energy(ref_params)

    def loss_fn(p):
        e = ground_energy(p)
        return (e - energy_ref) ** 2

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    params = {
        "J_12": jnp.array(0.5),
        "J_23": jnp.array(-0.3),
        "phi_1": jnp.array(0.2),
        "phi_2": jnp.array(-0.1),
    }
    print("parameter optimization trace:", flush=True)
    print(params)
    energy0 = ground_energy(params)
    print(f"Initial energy: {energy0:.6f}, reference energy: {energy_ref:.6f}")

    lr = 0.1
    loss0 = loss_fn(params)
    loss_final = loss0
    energy_final = energy0
    for step in range(80):
        loss, grad = value_and_grad(params)
        params = {k: v - lr * grad[k] for k, v in params.items()}
        energy_final = ground_energy(params)
        loss_final = loss_fn(params)
        if step % 10 == 0 or step == 79:
            print(f"Step {step:02d}: energy={energy_final:.6f}, loss={loss_final:.6e}")

    print("Optimized parameters:", flush=True)
    print(params)
    assert loss_final < loss0 * 0.5


def test_invalid_param_missing_raises():
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    ir = latex_to_ir(r"J \sigma_{z,1}", cfg, t_name="t")
    with pytest.raises(KeyError):
        H_fn = _build_parametric_hamiltonian(ir, cfg)
        H_fn({"theta": jnp.array(0.1)})


if __name__ == "__main__":
    test_jax_ising_param_optimization()
