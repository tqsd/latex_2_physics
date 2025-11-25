# flake8: noqa
"""
Parameter validation, alias rules, and backend dispatch.

This example walks through:
- Collecting required parameters from IR (Hamiltonian + collapse ops).
- Resolving aliases like omega_{c} vs omega_c.
- Seeing early validation errors before touching a backend.
- Dispatching to different backends after validation.

Call functions individually; nothing runs on import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.backend_utils import (
    collect_parameter_names,
    param_aliases,
    resolve_param,
    validate_required_params,
)
from latex_parser.compile_core import compile_model_core
from latex_parser.dsl import HilbertConfig, QubitSpec
from latex_parser.ir import latex_to_ir


def list_aliases(symbol: str) -> None:
    """
    Show the alias spellings we try for a given parameter name.
    """
    print("Aliases for", symbol, ":", param_aliases(symbol))


def collect_required_params(
    H_latex: str, c_ops_latex: Iterable[str] | None, cfg: HilbertConfig
) -> Tuple[set[str], set[str]]:
    """
    Return (required_symbols, time_names) for a Hamiltonian/c_ops pair.
    """
    ir = latex_to_ir(H_latex, cfg, t_name="t")
    time_names = {"t"}
    required = collect_parameter_names(ir, cfg, time_names)
    for c in c_ops_latex or []:
        ir_c = latex_to_ir(c, cfg, t_name="t")
        required |= collect_parameter_names(ir_c, cfg, time_names)
    return required, time_names


def demonstrate_validation() -> None:
    """
    Run validation for a simple driven qubit with collapse.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}"
    c_ops = [r"\sqrt{\gamma} \sigma_{-,1}"]
    required, time_names = collect_required_params(H, c_ops, cfg)
    print("Required symbols:", sorted(required))
    params_good = {"omega_0": 2.0, "A": 0.5, "omega": 1.0, "gamma": 0.1}
    validate_required_params(required, params_good, time_names)
    params_missing = {"omega_0": 2.0}
    try:
        validate_required_params(required, params_missing, time_names)
    except Exception as exc:  # noqa: BLE001 - user-facing demo
        print("Expected failure:", exc)


def resolve_examples() -> None:
    """
    Demonstrate resolve_param on multiple spellings.
    """
    params = {"omega_c": 1.0, "kappa": 0.1, "A": 0.5}
    for name in ["omega_{c}", "omega c", "kappa", "A"]:
        key, val = resolve_param(name, params)
        print(f"{name!r} resolved to key={key!r}, val={val}")


def dispatch_after_validation() -> None:
    """
    Show full validation + dispatch using compile_model_core.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\delta \sigma_{x,1}"
    params = {"delta": 0.4}
    model = compile_model_core(
        backend="qutip",
        H_latex=H,
        params=params,
        config=cfg,
        c_ops_latex=None,
        t_name="t",
        time_symbols=None,
    )
    print("Dispatched model type:", type(model))


def advanced_alias_patterns() -> None:
    """
    Show aliases with braces, spaces, and asterisks.
    """
    cases = ["omega_{c}", "omega_{ c }", "omega*c", "omega c", "{omega_c}"]
    for case in cases:
        print(case, "->", param_aliases(case))


def guard_against_operator_symbols() -> None:
    """
    Confirm operator symbols are not treated as params during collection.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\omega \sigma_{z,1} + \sigma_{x,1}"
    ir = latex_to_ir(H, cfg, t_name="t")
    req = collect_parameter_names(ir, cfg, {"t"})
    print("Parameters collected (should exclude operators):", req)


def validate_with_multiple_time_symbols() -> None:
    """
    Collect params when multiple time-like symbols are in use.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"A \cos(\omega t) \sigma_{x,1} + B \sin(\nu s) \sigma_{x,1}"
    ir = latex_to_ir(H, cfg, t_name="t", time_symbols=("s",))
    req = collect_parameter_names(ir, cfg, {"t", "s"})
    print("Required with extra time symbols:", sorted(req))


def validate_collapse_time_dependence() -> None:
    """
    Show param collection for time-dependent collapse operators.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    c_ops = [r"\sqrt{\gamma(t)} \sigma_{-,1}", r"\sqrt{\kappa} \sigma_{x,1}"]
    req = set()
    for c in c_ops:
        ir = latex_to_ir(c, cfg, t_name="t")
        req |= collect_parameter_names(ir, cfg, {"t"})
    print("Collapse-required params:", sorted(req))


def compare_backends_after_validation() -> None:
    """
    Dispatch the same LaTeX to multiple backends after validation.
    """
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    H = r"\delta \sigma_{x,1}"
    params = {"delta": 0.4}
    for backend in ("qutip", "numpy"):
        model = compile_model_core(
            backend=backend,
            H_latex=H,
            params=params,
            config=cfg,
            c_ops_latex=None,
            t_name="t",
            time_symbols=None,
        )
        print(f"Backend {backend} produced type:", type(model))


def resolve_bulk(names: Iterable[str], params: Dict[str, complex]) -> None:
    """
    Resolve a list of names against a parameter dict.
    """
    for name in names:
        try:
            key, val = resolve_param(name, params)
            print(f"{name!r} -> {key!r}={val}")
        except Exception as exc:  # noqa: BLE001 - user-facing demo
            print(f"{name!r} -> error {exc}")


def resolve_demonstration_suite() -> None:
    """
    Suite of resolve_param calls covering edge cases.
    """
    params = {"omega_c": 1.0, "g": 0.5, "alpha": 0.3}
    names = ["omega_{c}", "omega*c", "omega c", "omega{c}", "g", "beta"]
    resolve_bulk(names, params)


def main() -> None:
    list_aliases("omega_{c}")
    demonstrate_validation()
    resolve_examples()
    dispatch_after_validation()
    advanced_alias_patterns()
    guard_against_operator_symbols()
    validate_with_multiple_time_symbols()
    validate_collapse_time_dependence()
    compare_backends_after_validation()
    resolve_demonstration_suite()
    print("End of parameter validation tour.")


# Notes for developers:
# - The registry-based dispatch in compile_core already performs the
#   required-parameter collection shown above; this file exposes the same
#   helpers for learning and debugging.
# - To plug in your own backend, register it via `register_backend` and the
#   same validation will run automatically.


if __name__ == "__main__":
    main()
