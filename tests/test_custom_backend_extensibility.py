import numpy as np
import sympy as sp
from qutip import Qobj  # type: ignore

from latex_parser.backend_base import BackendBase, BackendOptions
from latex_parser.dsl import (
    CustomSpec,
    HilbertConfig,
    register_operator_macro,
)
from latex_parser.dsl_constants import register_operator_function
from latex_parser.ir import latex_to_ir, OperatorFunctionRef
from latex_parser.backend_qutip import compile_static_hamiltonian_from_latex
from latex_parser.backend_jax import (
    compile_static_hamiltonian_from_latex as compile_static_jax,
)


class DummyBackend(BackendBase):
    """Minimal backend to test extensibility; returns NumPy arrays."""

    def _make_cache(self, config, options):
        return {"config": config}

    def _compile_static(self, ir, cache, params, options=None):
        # Return a matrix with scalar coefficients on the diagonal for test visibility.
        diag = []
        for term in ir.terms:
            coeff = complex(
                term.scalar_expr.subs({sp.Symbol(k): v for k, v in params.items()})
            )
            diag.append(coeff)
        if not diag:
            diag = [0.0]
        return np.diag(diag)

    def _compile_time_dependent(
        self, ir, cache, params, *, t_name, time_symbols, options=None
    ):
        return self._compile_static(ir, cache, params, options=options)


def test_custom_backend_with_extension_hooks():
    # Register a new LaTeX pattern and operator functions
    register_operator_macro("foo", "Jx")
    register_operator_function("sinh")

    # Build CustomSpec providing Jx operator
    Jx = Qobj(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex))
    custom_template = CustomSpec(label="c", index=1, dim=2, operators={"Jx": Jx})
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[custom_template])

    # LaTeX uses new macro and operator function
    ir = latex_to_ir(r"\sinh(\foo_{1})", cfg)
    assert len(ir.terms) >= 1
    assert any(
        isinstance(op, OperatorFunctionRef) for term in ir.terms for op in term.ops
    )

    backend = DummyBackend()
    out = backend.compile_static_from_ir(ir, cfg, params={}, options=BackendOptions())
    assert out.shape == (1, 1)
    # Dummy backend places scalar (1) on the diagonal
    assert np.allclose(out, np.array([[1.0]]))

    # Qutip backend path
    H_q = compile_static_hamiltonian_from_latex(r"\sinh(\foo_{1})", cfg, params={})
    H_q_arr = np.array(H_q.full(), dtype=complex)
    assert H_q_arr.shape == (2, 2)

    # JAX backend path (if JAX available)
    try:
        H_jax = compile_static_jax(r"\sinh(\foo_{1})", cfg, params={})
        assert H_jax.shape == (2, 2)
    except Exception:
        # Allow environments without JAX
        pass
