# flake8: noqa
"""
Open-system example: static Hamiltonian with collapse operators.

What this shows:
- Adding collapse operators via LaTeX strings (`c_ops_latex`).
- Reusing the same parameter dict for Hamiltonian and collapse channels.
- Inspecting compiled `H`, `c_ops`, and `args` from the QuTiP backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.latex_api import compile_model as compile_latex_model


def main() -> None:
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    c_ops = [r"\sqrt{\gamma} \sigma_{-,1}"]
    params = {"omega_0": 1.0, "gamma": 0.2}
    model = compile_latex_model(
        H_latex=H_latex,
        params=params,
        c_ops_latex=c_ops,
        qubits=1,
        t_name="t",
    )
    print("H (static):", model.H)
    print("c_ops:", model.c_ops)
    print("args:", model.args)


if __name__ == "__main__":
    main()
