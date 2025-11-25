"""
IR debugging and validation cookbook.

This example is meant for developers who need to:
- Inspect the intermediate representation (IR) produced from LaTeX.
- Understand how time dependence is detected.
- Diagnose missing-parameter errors before hitting a backend.
- See how operator ordering is preserved in the IR.

Run functions individually from a REPL; nothing executes on import.
"""

# flake8: noqa
from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint
from typing import Iterable

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.backend_utils import collect_parameter_names, validate_required_params
from latex_parser.dsl import BosonSpec, HilbertConfig, QubitSpec
from latex_parser.ir import HamiltonianIR, latex_to_ir, parse_latex_expr


def _print_terms(ir: HamiltonianIR) -> None:
    """Helper: pretty-print IR terms."""
    print("has_time_dep:", ir.has_time_dep)
    for idx, term in enumerate(ir.terms):
        print(f"Term {idx}: scalar={term.scalar_expr}")
        print("  ops:", term.ops)


def inspect_basic_ir() -> None:
    """
    Parse a simple driven qubit and print IR contents.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    _print_terms(ir)


def inspect_operator_ordering() -> None:
    """
    Show that operator ordering is preserved in the IR.

    Note: SymPy may reorder internally; the IR preserves the product order
    after non-commutative handling in `latex_to_ir`.
    """
    cfg = HilbertConfig(
        qubits=[],
        bosons=[BosonSpec(label="a", index=1, cutoff=3)],
        customs=[],
    )
    H_latex = r"a_{1}^{\dagger} a_{1}"
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    _print_terms(ir)


def detect_time_dependence() -> None:
    """
    Show how time dependence is flagged via scalar free symbols.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H_td = r"A \cos(\omega t) \sigma_{x,1}"
    H_static = r"\frac{\omega_0}{2} \sigma_{z,1}"
    ir_td = latex_to_ir(H_td, cfg, t_name="t")
    ir_static = latex_to_ir(H_static, cfg, t_name="t")
    print("TD term:")
    _print_terms(ir_td)
    print("Static term:")
    _print_terms(ir_static)


def validate_params_against_ir() -> None:
    """
    Demonstrate param collection + validation without running a backend.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\omega \sigma_{z,1} + g \cos(\nu t) \sigma_{x,1}"
    c_ops = [r"\sqrt{\gamma} \sigma_{-,1}"]
    ir_H = latex_to_ir(H, cfg, t_name="t")
    time_names = {"t"}
    required = collect_parameter_names(ir_H, cfg, time_names)
    for c in c_ops:
        ir_c = latex_to_ir(c, cfg, t_name="t")
        required |= collect_parameter_names(ir_c, cfg, time_names)
    print("Required symbols:", sorted(required))
    params_ok = {"omega": 1.0, "g": 0.5, "nu": 2.0, "gamma": 0.1}
    params_missing = {"omega": 1.0}
    validate_required_params(required, params_ok, time_names)
    try:
        validate_required_params(required, params_missing, time_names)
    except Exception as exc:  # noqa: BLE001 - user-facing demo
        print("Expected failure:", exc)


def parse_latex_directly(exprs: Iterable[str]) -> None:
    """
    Show the raw SymPy expressions produced by `parse_latex_expr`.
    """
    for latex in exprs:
        expr = parse_latex_expr(latex)
        print("LaTeX:", latex)
        print("SymPy:", expr)
        print("Free symbols:", [s.name for s in expr.free_symbols])
        print("-")


def explore_symbol_aliases() -> None:
    """
    Display how different spellings map to the same SymPy symbol names.
    """
    cases = [
        r"\omega_c",
        r"\omega_{c}",
        r"\omega_{ c }",
        r"\omega_{c} t",
    ]
    parse_latex_directly(cases)


def inspect_operator_functions() -> None:
    """
    Show operator-valued function handling in the IR.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    exprs = [
        r"\exp(\sigma_{z,1})",
        r"\cos(\phi) \sigma_{x,1}",  # scalar cos(phi) times operator
        r"\exp(\sigma_{z,1}) \sigma_{x,1}",  # invalid operator-valued scalar
    ]
    valid = []
    invalid = []
    for e in exprs:
        try:
            ir = latex_to_ir(e, cfg, t_name="t")
            valid.append((e, ir))
        except Exception as exc:  # noqa: BLE001 - user-facing demo
            invalid.append((e, exc))
    print("Valid expressions:")
    for e, ir in valid:
        print(" ", e)
        _print_terms(ir)
    print("Invalid expressions:")
    for e, exc in invalid:
        print(" ", e, "->", exc)


def show_ir_math_ops() -> None:
    """
    Apply simple SymPy math to IR scalar parts (e.g., simplify).
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"g (1+1) \sigma_{x,1}"
    ir = latex_to_ir(H, cfg, t_name="t")
    _print_terms(ir)
    simplified = []
    for term in ir.terms:
        scalar_simplified = sp.simplify(term.scalar_expr)
        simplified.append((scalar_simplified, term.ops))
    print("After SymPy simplify:")
    pprint(simplified)


def explore_time_symbol_overrides() -> None:
    """
    Demonstrate adding extra time-like symbols beyond the default t_name.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"A \cos(\omega s) \sigma_{x,1}"
    ir = latex_to_ir(H, cfg, t_name="t", time_symbols=("s",))
    print("Extra time symbol 's' yields has_time_dep =", ir.has_time_dep)
    _print_terms(ir)


def show_expansion_guard() -> None:
    """
    Illustrate the expansion guard for large operator sums.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"( \sigma_{x,1} + \sigma_{y,1} )^3"
    ir = latex_to_ir(H, cfg, t_name="t")
    print("Expanded term count:", len(ir.terms))
    _print_terms(ir)


def rescue_merged_time_scalars() -> None:
    """
    Show how merged time scalars (omega_{dt}) are rescued into omega_{d} * t.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\omega_{dt} \sigma_{x,1}"
    ir = latex_to_ir(H, cfg, t_name="t")
    _print_terms(ir)


if __name__ == "__main__":
    inspect_basic_ir()
    inspect_operator_ordering()
    detect_time_dependence()
    validate_params_against_ir()
    explore_symbol_aliases()
    inspect_operator_functions()
    show_ir_math_ops()
    explore_time_symbol_overrides()
    show_expansion_guard()
    rescue_merged_time_scalars()
