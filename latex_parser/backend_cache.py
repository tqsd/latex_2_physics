from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import reduce
from typing import Generic, List, Tuple, TypeVar

from latex_parser.dsl import DSLValidationError, HilbertConfig

TIdentity = TypeVar("TIdentity")


@dataclass(frozen=True)
class SubsystemInfo:
    """Metadata describing one tensor-product factor."""

    kind: str
    label: str
    index: int
    dim: int
    spec: object


class BaseOperatorCache(ABC, Generic[TIdentity]):
    """
    Shared subsystem bookkeeping for backend operator caches.

    Subclasses provide backend-specific identity creation and Kronecker products.
    """

    def __init__(self, config: HilbertConfig) -> None:
        self.config = config
        self.subsystems: List[SubsystemInfo] = []
        self.subsystem_index: dict[Tuple[str, str, int], int] = {}
        self.identities: List[TIdentity] = []
        self._identity_factors: Tuple[TIdentity, ...] | None = None
        self.global_identity: TIdentity | None = None

        self._build_subsystems()
        self._build_identities()
        self._build_global_identity()

    @abstractmethod
    def _local_identity(self, dim: int) -> TIdentity:
        """Return a backend-specific identity of shape (dim, dim)."""

    @abstractmethod
    def _kron(self, a: TIdentity, b: TIdentity) -> TIdentity:
        """Kronecker product between backend-specific matrices."""

    def _build_subsystems(self) -> None:
        subs: List[SubsystemInfo] = []

        for q in self.config.qubits:
            subs.append(
                SubsystemInfo(kind="qubit", label=q.label, index=q.index, dim=2, spec=q)
            )

        for b in self.config.bosons:
            subs.append(
                SubsystemInfo(
                    kind="boson",
                    label=b.label,
                    index=b.index,
                    dim=b.cutoff,
                    spec=b,
                )
            )

        for c in self.config.customs:
            subs.append(
                SubsystemInfo(
                    kind="custom", label=c.label, index=c.index, dim=c.dim, spec=c
                )
            )

        self.subsystems = subs
        for pos, ss in enumerate(self.subsystems):
            key = (ss.kind, ss.label, ss.index)
            if key in self.subsystem_index:
                raise DSLValidationError(
                    "Duplicate subsystem "
                    f"(kind={ss.kind}, label={ss.label}, index={ss.index}) "
                    "in HilbertConfig."
                )
            self.subsystem_index[key] = pos

    def _build_identities(self) -> None:
        self.identities = [self._local_identity(ss.dim) for ss in self.subsystems]
        self._identity_factors = tuple(self.identities)
        self.global_identity = None

    def _build_global_identity(self) -> None:
        if not self.subsystems:
            self.global_identity = self._local_identity(1)
            self._identity_factors = tuple()
            return

        assert self.identities, "Identities must be built before global identity."
        self.global_identity = reduce(self._kron, self.identities)
        self._identity_factors = tuple(self.identities)
