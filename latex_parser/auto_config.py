from __future__ import annotations

import re
from typing import Mapping, Sequence

from latex_parser.dsl import (
    BosonSpec,
    CustomSpec,
    DSLValidationError,
    HilbertConfig,
    QubitSpec,
    canonicalize_physics_latex,
)

__all__ = ["infer_config_from_latex"]


def _scan_operator_indices(
    latex_items: Sequence[str],
) -> tuple[set[int], set[int], dict[int, set[str]]]:
    r"""
    Scan canonicalized LaTeX snippets and collect operator indices by kind.
    """
    boson_bases = {"a", "adag", "af", "adagf", "n"}
    qubit_bases = {"sx", "sy", "sz", "sp", "sm"}

    qubits: set[int] = set()
    bosons: set[int] = set()
    customs: dict[int, set[str]] = {}

    pattern = re.compile(r"([A-Za-z][A-Za-z0-9]*)_\{?(\d+)\}?")

    for item in latex_items:
        canonical = canonicalize_physics_latex(item)
        for m in pattern.finditer(canonical):
            base_raw, idx_str = m.groups()
            base = base_raw.lstrip("\\")
            idx = int(idx_str)
            if base in boson_bases:
                bosons.add(idx)
            elif base in qubit_bases:
                qubits.add(idx)
            else:
                customs.setdefault(idx, set()).add(base)

    return qubits, bosons, customs


def infer_config_from_latex(
    H_latex: str,
    c_ops_latex: Sequence[str] | None = None,
    *,
    qubit_label: str = "q",
    boson_label: str = "a",
    custom_label: str = "c",
    default_boson_cutoff: int = 2,
    custom_templates: Mapping[str, CustomSpec] | None = None,
) -> HilbertConfig:
    r"""
    Build a HilbertConfig by inspecting operator indices in LaTeX.
    """
    items = [H_latex] + list(c_ops_latex or [])
    qubit_idxs, boson_idxs, custom_ops = _scan_operator_indices(items)

    max_qubit_idx = max(qubit_idxs) if qubit_idxs else 0
    qubit_specs = [
        QubitSpec(label=qubit_label, index=i) for i in range(1, max_qubit_idx + 1)
    ]

    max_boson_idx = max(boson_idxs) if boson_idxs else 0
    boson_specs = [
        BosonSpec(label=boson_label, index=i, cutoff=default_boson_cutoff)
        for i in range(1, max_boson_idx + 1)
    ]

    custom_specs: list[CustomSpec] = []
    if custom_ops:
        template = custom_templates.get(custom_label) if custom_templates else None
        if template is None:
            raise DSLValidationError(
                "Custom operators detected in LaTeX but no custom template provided. "
                "Pass custom_templates={label: CustomSpec(...)} or supply an "
                "explicit HilbertConfig."
            )
        for idx in sorted(custom_ops):
            required_ops = custom_ops[idx]
            missing = required_ops - set(template.operators.keys())
            if missing:
                raise DSLValidationError(
                    f"Custom template for label '{custom_label}' missing operators "
                    f"{sorted(missing)} required by LaTeX."
                )
            custom_specs.append(
                CustomSpec(
                    label=custom_label,
                    index=idx,
                    dim=template.dim,
                    operators=template.operators,
                    role=template.role,
                )
            )

    return HilbertConfig(qubits=qubit_specs, bosons=boson_specs, customs=custom_specs)
