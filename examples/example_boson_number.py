# flake8: noqa
"""
Bosonic number operator example with a single mode.

What this shows:
- Using bosonic subsystems (cutoff provided via `bosons=[cutoff]`).
- Purely static Hamiltonian with a single scalar parameter.
- Printed `H` is a QuTiP `Qobj` matching the chosen cutoff dimension.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.latex_api import compile_latex_model


def main() -> None:
    H_latex = r"\omega n_{1}"
    params = {"omega": 1.25}
    model = compile_latex_model(H_latex=H_latex, params=params, bosons=[4], t_name="t")
    print("Boson number Hamiltonian:\n", model.H)


if __name__ == "__main__":
    main()
