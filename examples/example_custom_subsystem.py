# flake8: noqa
"""
Custom finite-dimensional subsystem example (spin-1 with Jx operator).

What this shows:
- How to wrap custom operator matrices (here: spin-1 ladder/Jx) into `CustomSpec`.
- Configuring a `HilbertConfig` with only a custom subsystem (no qubits/bosons).
- Compiling LaTeX that references custom operators (`Jx_{1}`) and printing `H`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from qutip import Qobj  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.latex_api import compile_model as compile_latex_model
from latex_parser.dsl import CustomSpec, HilbertConfig


def main() -> None:
    sqrt2 = np.sqrt(2.0)
    Jp = Qobj(np.array([[0, sqrt2, 0], [0, 0, sqrt2], [0, 0, 0]], dtype=complex))
    Jm = Jp.dag()
    Jx = 0.5 * (Jp + Jm)

    spec = CustomSpec(
        label="c",
        index=1,
        dim=3,
        operators={"Jx": Jx, "Jp": Jp, "Jm": Jm},
    )
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[spec])
    H_latex = r"\omega_J Jx_{1}"
    params = {"omega_J": 0.3}
    model = compile_latex_model(H_latex=H_latex, params=params, config=cfg)
    print("Custom subsystem Hamiltonian:\n", model.H)


if __name__ == "__main__":
    main()
