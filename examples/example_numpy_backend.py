# flake8: noqa
"""
Using the NumPy backend to build a dense matrix instead of QuTiP objects.

What this shows:
- Selecting `backend="numpy"` to bypass QuTiP.
- Providing a `config` explicitly via `make_config`.
- Result is a plain NumPy array; useful for custom workflows or JIT-wrapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.latex_api import compile_latex_model, make_config


def main() -> None:
    cfg = make_config(qubits=[1], bosons=[])
    H_latex = r"\delta \sigma_{x,1}"
    params = {"delta": 0.4}
    model = compile_latex_model(
        H_latex=H_latex,
        params=params,
        backend="numpy",
        config=cfg,
    )
    print("NumPy backend matrix:\n", model)


if __name__ == "__main__":
    main()
