# flake8: noqa
"""
JAX backend workflow with autodiff and platform configuration.

Goals:
- Show how to guard imports when JAX may be absent.
- Configure dtype/platform overrides via JaxBackendOptions.
- Compile a small Hamiltonian and take gradients with jax.grad.
- Illustrate parameter tracking (compiled.parameters set).

This file is safe to import even if JAX is not installed; demos check at runtime.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force CPU by default to avoid GPU/driver issues in example contexts.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from latex_parser.backend_jax import JaxBackend, JaxBackendOptions
from latex_parser.dsl import HilbertConfig, QubitSpec
from latex_parser.ir import latex_to_ir


def _static_matrix(compiled: Any) -> Any:
    """
    Return the static Hamiltonian component from a compiled JAX model.

    CompiledOpenSystemJax exposes ``H`` which is either a single array
    (static) or a time-dependent list whose first element is the static H0.
    """
    H = compiled.H
    return H[0] if isinstance(H, list) else H


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:  # pragma: no cover - optional dependency
        print("JAX not installed; install with `pip install jax jaxlib`.")
        raise RuntimeError("JAX missing") from exc
    return jax, jnp


def compile_with_custom_options() -> Dict[str, Any]:
    """
    Compile a driven qubit with explicit dtype/platform overrides.
    """
    jax, jnp = _require_jax()
    opts = JaxBackendOptions(dtype=jnp.complex64, platform="cpu")
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \cos(\omega t) \sigma_{x,1}"
    params = {"A": 0.5, "omega": 1.0}
    backend = JaxBackend()
    compiled = backend.compile_open_system_from_latex(
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=None,
        t_name="t",
        options=opts,
    )
    print("Compiled type:", type(compiled.H))
    print("Parameters tracked:", compiled.parameters)
    return {"compiled": compiled, "opts": opts, "jax": jax, "jnp": jnp}


def gradient_of_expectation() -> None:
    """
    Compute a simple gradient of an expectation value with respect to A.
    """
    jax, jnp = _require_jax()
    context = compile_with_custom_options()
    compiled = context["compiled"]
    H = _static_matrix(compiled)

    def energy(A_val: float) -> Any:
        H_scaled = (A_val / 0.5) * H
        # Ground state energy (simplified for demo): min eigenvalue.
        eigvals = jnp.linalg.eigvalsh(H_scaled)
        return jnp.min(eigvals)

    grad_fn = jax.grad(energy)
    print("dE/dA at A=0.5:", grad_fn(0.5))


def time_dependent_jax_ir() -> None:
    """
    Show how IR time dependence is reflected in JAX compilation.
    """
    _require_jax()
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \cos(\omega t) \sigma_{x,1}"
    params = {"A": 0.5, "omega": 1.0}
    backend = JaxBackend()
    compiled = backend.compile_open_system_from_latex(
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=None,
        t_name="t",
    )
    print("Time-dependent flag:", compiled.time_dependent)
    print("H list structure:", compiled.H)


def inspect_ir_for_jax() -> None:
    """
    Inspect IR before passing to JAX backend (developer aid).
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \cos(\omega t) \sigma_{x,1} + B \sigma_{z,1}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    print("IR has_time_dep:", ir.has_time_dep)
    for idx, term in enumerate(ir.terms):
        print(f"Term {idx}: scalar={term.scalar_expr}, ops={term.ops}")


def explore_platform_settings() -> None:
    """
    Show how to set JAX platform via options/env.
    """
    try:
        import os
        from latex_parser.backend_jax import _apply_platform
    except Exception:  # pragma: no cover - optional
        print("JAX not installed; skipping platform demo.")
        return
    for platform in ("cpu", "gpu"):
        os.environ["JAX_PLATFORM_NAME"] = platform
        _apply_platform(platform)
        print("Requested platform:", platform)


def dtype_demonstration() -> None:
    """
    Compare float32 vs float64 compilation for the same Hamiltonian.
    """
    try:
        _, jnp = _require_jax()
    except RuntimeError:
        return
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\delta \sigma_{x,1}"
    params = {"delta": 0.4}
    backend = JaxBackend()
    for dtype in (jnp.complex64, jnp.complex128):
        opts = JaxBackendOptions(dtype=dtype, platform="cpu")
        compiled = backend.compile_open_system_from_latex(
            H_latex=H_latex,
            params=params,
            config=cfg,
            c_ops_latex=None,
            t_name="t",
            options=opts,
        )
        H_static = _static_matrix(compiled)
        print("dtype:", dtype, "H dtype:", H_static.dtype)


def simple_batch_eval() -> None:
    """
    Evaluate a compiled JAX Hamiltonian over a batch of parameter values.
    """
    try:
        jax, jnp = _require_jax()
    except RuntimeError:
        return
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"A \sigma_{x,1}"
    params = {"A": 1.0}
    backend = JaxBackend()
    compiled = backend.compile_open_system_from_latex(
        H_latex=H_latex,
        params=params,
        config=cfg,
        c_ops_latex=None,
        t_name="t",
    )
    H0 = _static_matrix(compiled)

    def evaluate_batch(A_vals):
        return jax.vmap(lambda a: a * H0)(A_vals)

    batch = jnp.linspace(0.1, 1.0, 4)
    result = evaluate_batch(batch)
    print("Batch scales:", batch)
    print("Batch H shape:", result.shape)


def guard_missing_jax_install() -> None:
    """
    Provide a single place to verify JAX installation status.
    """
    try:
        _require_jax()
        print("JAX present.")
    except RuntimeError:
        print("JAX missing; install to run autodiff demos.")


def main() -> None:
    try:
        compile_with_custom_options()
        gradient_of_expectation()
        time_dependent_jax_ir()
        explore_platform_settings()
        dtype_demonstration()
        simple_batch_eval()
    except RuntimeError:
        print("Skipping JAX-specific demos.")
    inspect_ir_for_jax()
    guard_missing_jax_install()


if __name__ == "__main__":
    main()
