# flake8: noqa
"""
Static qubit Hamiltonian compiled from LaTeX.

What this shows:
- Minimal inputs to get a working Hamiltonian (`qubits=1`, no collapse ops).
- Default backend dispatch (QuTiP) via `compile_latex_model`.
- Inspecting the compiled `H` object that you can pass to solvers.

Tip: change `t_name` or add `backend="numpy"` to explore other backends.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.latex_api import compile_latex_model


def main() -> None:
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1}"
    params = {"omega_0": 2.0}
    model = compile_latex_model(H_latex=H_latex, params=params, qubits=1, t_name="t")
    print("Static qubit Hamiltonian:\n", model.H)


if __name__ == "__main__":
    main()
