# flake8: noqa
"""
Single-qubit drive with a time-dependent cosine envelope.

What this shows:
- Time-dependent scalar envelopes detected automatically from `t_name`.
- The QuTiP backend returns an H-list suitable for `mesolve`.
- How `time_dependent` is flagged on the compiled model.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.latex_api import compile_model as compile_latex_model


def main() -> None:
    H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}"
    params = {"omega_0": 2.0, "A": 0.3, "omega": 1.5}
    model = compile_latex_model(H_latex=H_latex, params=params, qubits=1, t_name="t")
    print("Time-dependent flag:", model.time_dependent)
    print("H representation (QuTiP list for td):\n", model.H)


if __name__ == "__main__":
    main()
