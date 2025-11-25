# flake8: noqa
"""
Advanced operator-function usage and validation.

This example covers:
- Allowed operator-valued functions (exp, cos, sin, etc.).
- Scalar factors on operator functions.
- Mixing operator functions with time-dependent envelopes.
- How to handle invalid operator functions and surface clear errors.

Run individual functions from a REPL; nothing executes on import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Tuple

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.dsl import HilbertConfig, QubitSpec
from latex_parser.dsl_constants import ALLOWED_OPERATOR_FUNCTIONS
from latex_parser.ir import HamiltonianIR, latex_to_ir


def _describe_ir(ir: HamiltonianIR) -> None:
    print("has_time_dep:", ir.has_time_dep)
    for idx, term in enumerate(ir.terms):
        print(f"Term {idx}: scalar={term.scalar_expr}")
        print("  ops:", term.ops)


def list_allowed_functions() -> None:
    """
    Print the currently allowed operator-valued functions.
    """
    print("Allowed operator functions:", sorted(ALLOWED_OPERATOR_FUNCTIONS))


def simple_operator_function() -> None:
    """
    Compile exp(sigma_z) acting on a qubit.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\exp(\sigma_{z,1})"
    ir = latex_to_ir(H, cfg, t_name="t")
    _describe_ir(ir)


def operator_function_with_scalar() -> None:
    """
    Compile a scalar times an operator function, e.g. g * exp(sigma_z).
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"g \exp(\sigma_{z,1})"
    ir = latex_to_ir(H, cfg, t_name="t")
    _describe_ir(ir)


def operator_function_with_power() -> None:
    """
    Operator function applied to a powered operator (sigma_x^2).
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\cos(\sigma_{x,1}^{2})"
    ir = latex_to_ir(H, cfg, t_name="t")
    _describe_ir(ir)


def operator_function_plus_time_envelope() -> None:
    """
    Mix a time-dependent scalar with an operator function argument.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"A \cos(\omega t) \exp(\sigma_{z,1})"
    ir = latex_to_ir(H, cfg, t_name="t")
    _describe_ir(ir)


def invalid_operator_function_sum() -> None:
    """
    Demonstrate rejection of operator-function arguments that are sums.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    bad = [
        r"\exp(\sigma_{x,1} + \sigma_{z,1})",
        r"\tanh(\sigma_{z,1})",
    ]
    for expr in bad:
        try:
            latex_to_ir(expr, cfg, t_name="t")
        except Exception as exc:  # noqa: BLE001 - user-facing demo
            print(expr, "->", exc)


def batch_operator_functions(exprs: Iterable[str]) -> None:
    """
    Compile a list of operator-function expressions and report properties.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    for expr in exprs:
        try:
            ir = latex_to_ir(expr, cfg, t_name="t")
        except Exception as exc:  # noqa: BLE001 - user-facing demo
            print(expr, "-> error:", exc)
            continue
        print(expr, "-> has_time_dep:", ir.has_time_dep)
        for term in ir.terms:
            print("  scalar:", term.scalar_expr)
            print("  ops:", term.ops)


def operator_function_simplify() -> None:
    """
    Apply SymPy simplify to scalar factors surrounding operator functions.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"2*g \exp(\sigma_{z,1}) + g*g \exp(\sigma_{x,1})"
    ir = latex_to_ir(H, cfg, t_name="t")
    simplified: list[Tuple[sp.Expr, list]] = []
    for term in ir.terms:
        simplified.append((sp.simplify(term.scalar_expr), term.ops))
    print("Simplified scalars:")
    for s, ops in simplified:
        print("  ", s, "| ops:", ops)


def operator_function_with_aliases() -> None:
    """
    Confirm parameter aliases still work when operator functions present.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\omega_{0} \exp(\sigma_{z,1})"
    ir = latex_to_ir(H, cfg, t_name="t")
    required = {sym.name for sym in ir.terms[0].scalar_expr.free_symbols}
    print("Required scalar symbols:", required)


def combine_operator_functions_with_products() -> None:
    """
    Compose operator functions with products of standard operators.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\exp(\sigma_{z,1}) \sigma_{x,1} + \exp(\sigma_{x,1}) \sigma_{z,1}"
    ir = latex_to_ir(H, cfg, t_name="t")
    _describe_ir(ir)


def register_and_use_custom_operator_function() -> None:
    """
    Demonstrate how to register a new operator function at runtime.
    """
    from latex_parser.dsl_constants import register_operator_function

    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    register_operator_function("sinh")
    H = r"\sinh(\sigma_{x,1})"
    ir = latex_to_ir(H, cfg, t_name="t")
    _describe_ir(ir)


def stress_test_operator_functions() -> None:
    """
    Build a small suite of varied operator-function expressions to ensure
    they remain within the allowed grammar.
    """
    exprs = [
        r"\cos(\sigma_{x,1}) + \cos(\sigma_{z,1})",
        r"\exp(\sigma_{x,1}) \exp(\sigma_{z,1})",
        r"A \exp(\sigma_{x,1}) + B \cos(\sigma_{y,1})",
        r"\exp(\sigma_{x,1}^{3})",
        r"\cos(\sigma_{z,1}) \cos(\omega t)",
    ]
    batch_operator_functions(exprs)


def operator_function_error_messages() -> None:
    """
    Surface friendly error messages for common mistakes.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    mistakes = [
        r"\exp(\sigma_{x,1} + \sigma_{z,1})",  # sum in argument
        r"\exp(\sigma_{x,1}^{-1})",  # negative power
        r"\log(\sigma_{x,1})",  # unsupported function
    ]
    for expr in mistakes:
        try:
            latex_to_ir(expr, cfg, t_name="t")
        except Exception as exc:  # noqa: BLE001 - user-facing demo
            print("[expected] ", expr, "->", exc)


def operator_functions_with_params() -> None:
    """
    Track scalar parameters that accompany operator functions.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"g \exp(\sigma_{z,1}) + \kappa \cos(\sigma_{x,1})"
    ir = latex_to_ir(H, cfg, t_name="t")
    required = set()
    for term in ir.terms:
        required |= {s.name for s in term.scalar_expr.free_symbols}
    print("Scalar symbols alongside operator functions:", sorted(required))


def main() -> None:
    list_allowed_functions()
    simple_operator_function()
    operator_function_with_scalar()
    operator_function_with_power()
    operator_function_plus_time_envelope()
    invalid_operator_function_sum()
    batch_operator_functions(
        [
            r"\cos(\sigma_{x,1})",
            r"\exp(\sigma_{z,1})",
            r"\sin(\sigma_{y,1}) \cos(\omega t)",
        ]
    )
    operator_function_simplify()
    operator_function_with_aliases()
    combine_operator_functions_with_products()
    register_and_use_custom_operator_function()
    stress_test_operator_functions()
    operator_function_error_messages()
    operator_functions_with_params()


if __name__ == "__main__":
    main()
