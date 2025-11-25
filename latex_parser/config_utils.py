from __future__ import annotations

from typing import Callable, Mapping, Sequence, Union

from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    HilbertConfig,
    QubitSpec,
    deformation_callable_from_latex,
)
from latex_parser.auto_config import infer_config_from_latex

BosonEntry = Union[
    int,
    tuple[int, str],
    tuple[int, Callable[[object], object]],
    tuple[int, str, Callable[[object], object]],
    tuple[int, str, str],  # (cutoff, label, deformation_latex)
    tuple[int, str, Callable[[object], object]],
]


def make_config(
    qubits: int | Sequence[int] = 0,
    bosons: Sequence[BosonEntry] | None = None,
    customs: Sequence[CustomSpec] | None = None,
) -> HilbertConfig:
    """
    Build a :class:`HilbertConfig` from simple specifications.
    """
    if isinstance(qubits, int):
        qubit_specs = [QubitSpec(label="q", index=i) for i in range(1, qubits + 1)]
    else:
        qubit_specs = [QubitSpec(label="q", index=i) for i in qubits]

    boson_specs = []
    if bosons:
        for idx, entry in enumerate(bosons, start=1):
            label = "a"
            deformation = None
            deformation_latex = None
            if isinstance(entry, tuple):
                if len(entry) == 2:
                    cutoff, second = entry
                    if isinstance(second, str):
                        label = second
                    else:
                        deformation = second
                elif len(entry) == 3:
                    cutoff, label, deformation = entry
                else:
                    raise TypeError(
                        "Boson entries must be int or tuples of length 2 or 3 "
                        "like (cutoff, label) or (cutoff, label, deformation)."
                    )
            else:
                cutoff = entry

            if isinstance(deformation, str):
                deformation_latex = deformation
                _, deformation = deformation_callable_from_latex(
                    deformation_latex, int(cutoff)
                )
            if deformation is None and deformation_latex is not None:
                _, deformation = deformation_callable_from_latex(
                    deformation_latex, int(cutoff)
                )

            boson_specs.append(
                BosonSpec(
                    label=label,
                    index=idx,
                    cutoff=int(cutoff),
                    deformation=deformation,
                    deformation_latex=deformation_latex,
                )
            )

    custom_specs = list(customs) if customs else []

    return HilbertConfig(qubits=qubit_specs, bosons=boson_specs, customs=custom_specs)


def resolve_config(
    *,
    config: HilbertConfig | None,
    auto_config: bool,
    H_latex: str,
    c_ops_latex: Sequence[str] | None,
    qubits: int | Sequence[int],
    bosons: Sequence[int] | Sequence[tuple[int, str]] | None,
    customs: Sequence[CustomSpec] | None,
    default_boson_cutoff: int,
    custom_templates: Mapping[str, CustomSpec] | None,
) -> HilbertConfig:
    """
    Centralized config resolution for all compile entry points.
    """
    if config is not None:
        return config
    if auto_config:
        return infer_config_from_latex(
            H_latex,
            c_ops_latex,
            default_boson_cutoff=default_boson_cutoff,
            custom_templates=custom_templates,
        )
    return make_config(qubits=qubits, bosons=bosons, customs=customs)
