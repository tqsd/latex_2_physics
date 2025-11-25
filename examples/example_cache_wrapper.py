# flake8: noqa
"""
Minimal BaseOperatorCache wrapper showing identities and ordering.

What this shows:
- Subclassing `BaseOperatorCache` to get subsystem ordering and identity caches.
- Inspecting `cache.subsystems` and `cache.global_identity`.
- A minimal pattern backend authors can reuse for NumPy-like backends.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.backend_cache import BaseOperatorCache
from latex_parser.dsl import HilbertConfig, QubitSpec


class NumpyCache(BaseOperatorCache[np.ndarray]):
    def _local_identity(self, dim: int) -> np.ndarray:
        return np.eye(dim)

    def _kron(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.kron(a, b)


def main() -> None:
    cfg = HilbertConfig(qubits=[QubitSpec(label="q", index=1)], bosons=[], customs=[])
    cache = NumpyCache(cfg)
    print("Subsystem order:", cache.subsystems)
    print("Global identity:\n", cache.global_identity)


if __name__ == "__main__":
    main()
