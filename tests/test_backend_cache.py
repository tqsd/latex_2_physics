import numpy as np
import pytest

from latex_parser.backend_cache import BaseOperatorCache
from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    QubitSpec,
)


class _DummyCache(BaseOperatorCache[np.ndarray]):
    """Simple concrete cache for testing the shared base implementation."""

    def _local_identity(self, dim: int) -> np.ndarray:
        return np.eye(dim)

    def _kron(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.kron(a, b)


def test_subsystem_order_and_identities():
    cache = _DummyCache(
        HilbertConfig(
            qubits=[QubitSpec(label="q", index=1)],
            bosons=[BosonSpec(label="a", index=2, cutoff=3)],
            customs=[CustomSpec(label="c", index=1, dim=4, operators={"X": np.eye(4)})],
        )
    )

    kinds = [(ss.kind, ss.label, ss.index, ss.dim) for ss in cache.subsystems]
    assert kinds == [("qubit", "q", 1, 2), ("boson", "a", 2, 3), ("custom", "c", 1, 4)]
    assert [id_op.shape for id_op in cache.identities] == [(2, 2), (3, 3), (4, 4)]
    assert cache.global_identity.shape == (24, 24)


def test_empty_config_builds_scalar_identity():
    cache = _DummyCache(HilbertConfig(qubits=[], bosons=[], customs=[]))
    assert cache.identities == []
    assert cache.global_identity.shape == (1, 1)
    assert cache.global_identity[0, 0] == 1.0


def test_duplicate_subsystem_rejected():
    with pytest.raises(DSLValidationError):
        _DummyCache(
            HilbertConfig(
                qubits=[QubitSpec(label="q", index=1), QubitSpec(label="q", index=1)],
                bosons=[],
                customs=[],
            )
        )


def test_abstract_methods_enforced():
    class MissingKron(BaseOperatorCache[np.ndarray]):  # type: ignore[misc]
        def _local_identity(self, dim: int) -> np.ndarray:
            return np.eye(dim)

    with pytest.raises(TypeError):
        MissingKron(HilbertConfig(qubits=[], bosons=[], customs=[]))
