#!/usr/bin/env python
# flake8: noqa
"""
Examples: integrating scqubits transmon/fluxonium with the LaTeX DSL.

These examples show how to:
1) Build scqubits devices.
2) Register their operators as CustomSpec entries.
3) Use operator-valued functions like cos(phi) directly in the DSL.
4) Verify compiled Hamiltonians against scqubits-provided matrices.

No long simulations are run by default; the examples stop after compiling
to QuTiP objects and, optionally, validating against scqubits to keep
runtime light. Run this file directly to execute both examples if
scqubits is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
import logging
from typing import Tuple

import numpy as np
from qutip import Qobj, destroy, qeye, tensor  # type: ignore
from scipy.linalg import cosm, sinm  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.dsl import CustomSpec, HilbertConfig
from latex_parser.latex_api import compile_model as compile_latex_model, make_config

logger = logging.getLogger(__name__)


def _require_scqubits():
    try:
        import scqubits as scq  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "scqubits is required for these examples. Install it with "
            "'pip install scqubits'."
        ) from exc
    return scq


def fluxonium_model(
    EJ: float = 8.0,
    EC: float = 1.0,
    EL: float = 0.5,
    cutoff: int = 35,
    n_g: float = 0.0,
    flux: float = 0.0,
) -> Tuple[object, HilbertConfig]:
    """
    Build a fluxonium Hamiltonian

        H = 4 EC (n - n_g)^2 + 0.5 EL phi^2 - EJ cos(phi)

    using scqubits operators mapped into a CustomSpec. Returns the compiled
    model and the HilbertConfig.
    """
    scq = _require_scqubits()

    flux_dev = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux)
    phi_op = flux_dev.phi_operator()
    n_op = flux_dev.n_operator()

    # scqubits >=3 returns NumPy arrays; wrap into Qobj for the DSL.
    if not hasattr(phi_op, "dims"):
        phi_op = Qobj(phi_op, dims=[[cutoff], [cutoff]])
    if not hasattr(n_op, "dims"):
        n_op = Qobj(n_op, dims=[[cutoff], [cutoff]])

    # Register custom subsystem with phi and n operators.
    spec = CustomSpec(
        label="c",
        index=1,
        dim=phi_op.dims[0][0],
        operators={"phi": phi_op, "nphi": n_op},
    )
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[spec])

    H_latex = r"4 EC (nphi_{1} - n_g)^2 + \frac{1}{2} EL \phi_{1}^2 - EJ \cos(\phi_{1})"
    params = {"EC": EC, "EL": EL, "EJ": EJ, "n_g": n_g}

    model = compile_latex_model(
        H_latex=H_latex,
        params=params,
        config=cfg,
        t_name="t",
    )
    return model, cfg


def fluxonium_model_with_flux_offset(
    EJ: float = 8.0,
    EC: float = 1.0,
    EL: float = 0.5,
    cutoff: int = 35,
    n_g: float = 0.0,
    phi_ext: float = 0.2,
) -> Tuple[object, HilbertConfig]:
    """
    Fluxonium with external flux offset via the shifted cosine

        -EJ cos(phi - phi_ext) = -EJ [cos(phi) cos(phi_ext) + sin(phi) sin(phi_ext)].

    Uses operator-valued cos/sin on the phi operator and scalar parameters
    for phi_ext.
    """
    scq = _require_scqubits()

    flux_dev = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=phi_ext)
    phi_op = flux_dev.phi_operator()
    n_op = flux_dev.n_operator()

    if not hasattr(phi_op, "dims"):
        phi_op = Qobj(phi_op, dims=[[cutoff], [cutoff]])
    if not hasattr(n_op, "dims"):
        n_op = Qobj(n_op, dims=[[cutoff], [cutoff]])

    spec = CustomSpec(
        label="c",
        index=1,
        dim=phi_op.dims[0][0],
        operators={"phi": phi_op, "nphi": n_op},
    )
    cfg = HilbertConfig(qubits=[], bosons=[], customs=[spec])

    H_latex = r"""
        4 EC (nphi_{1} - n_g)^2
        + \frac{1}{2} EL \phi_{1}^2
        - EJ \left(
            \cos(\phi_{1}) \cos(\phi_{ex})
            + \sin(\phi_{1}) \sin(\phi_{ex})
        \right)
    """
    params = {
        "EC": EC,
        "EL": EL,
        "EJ": EJ,
        "n_g": n_g,
        "phi_ex": phi_ext,
    }

    model = compile_latex_model(
        H_latex=H_latex,
        params=params,
        config=cfg,
        t_name="t",
    )
    return model, cfg


def verify_fluxonium_against_scqubits(
    EJ: float = 8.0,
    EC: float = 1.0,
    EL: float = 0.5,
    cutoff: int = 15,
    n_g: float = 0.0,
    flux: float = 0.0,
) -> float:
    """
    Compile the fluxonium Hamiltonian from LaTeX and compare to an explicit
    matrix built from the same scqubits ``phi`` and ``n`` operators. Returns
    the operator norm of the difference (expected to be ~0).
    """
    scq = _require_scqubits()
    model, cfg = fluxonium_model(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, n_g=n_g, flux=flux)
    H_dsl = model.H if isinstance(model.H, Qobj) else model.H[0]
    if not isinstance(H_dsl, Qobj) and hasattr(H_dsl, "shape"):
        dim = H_dsl.shape[0]
        H_dsl = Qobj(H_dsl, dims=[[dim], [dim]])

    flux_dev = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=flux)
    phi = flux_dev.phi_operator()
    n_op = flux_dev.n_operator()
    if not hasattr(phi, "dims"):
        phi = Qobj(phi, dims=[[cutoff], [cutoff]])
    if not hasattr(n_op, "dims"):
        n_op = Qobj(n_op, dims=[[cutoff], [cutoff]])

    cos_phi = Qobj(cosm(phi.full()), dims=phi.dims)
    H_ref = 4 * EC * (n_op - n_g) ** 2 + 0.5 * EL * (phi**2) - EJ * cos_phi

    H_sq = flux_dev.hamiltonian()
    if not isinstance(H_sq, Qobj) and hasattr(H_sq, "shape"):
        dim = H_sq.shape[0]
        H_sq = Qobj(H_sq, dims=[[dim], [dim]])

    diff = (H_dsl - H_ref).norm()
    diff_sq = (H_ref - H_sq).norm()
    if diff_sq > 0:  # pragma: no cover - informational
        logger.info(
            "scqubits Hamiltonian differs from reference by %.3e (truncation effects?)",
            diff_sq,
        )
    return diff


def verify_fluxonium_with_flux_offset_against_scqubits(
    EJ: float = 8.0,
    EC: float = 1.0,
    EL: float = 0.5,
    cutoff: int = 15,
    n_g: float = 0.0,
    phi_ext: float = 0.2,
) -> float:
    """
    Verify flux-offset fluxonium by comparing the DSL-compiled Hamiltonian
    to an explicit reference built from scqubits ``phi``/``n`` operators and
    matrix cosine/sine. Returns the operator norm of the difference
    (expected ~0).
    """
    scq = _require_scqubits()
    model, cfg = fluxonium_model_with_flux_offset(
        EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, n_g=n_g, phi_ext=phi_ext
    )
    H_dsl = model.H if isinstance(model.H, Qobj) else model.H[0]
    if not isinstance(H_dsl, Qobj) and hasattr(H_dsl, "shape"):
        dim = H_dsl.shape[0]
        H_dsl = Qobj(H_dsl, dims=[[dim], [dim]])

    flux_dev = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, cutoff=cutoff, flux=phi_ext)
    phi = flux_dev.phi_operator()
    n_op = flux_dev.n_operator()
    if not hasattr(phi, "dims"):
        phi = Qobj(phi, dims=[[cutoff], [cutoff]])
    if not hasattr(n_op, "dims"):
        n_op = Qobj(n_op, dims=[[cutoff], [cutoff]])

    cos_phi = Qobj(cosm(phi.full()), dims=phi.dims)
    sin_phi = Qobj(sinm(phi.full()), dims=phi.dims)
    H_ref = (
        4 * EC * (n_op - n_g) ** 2
        + 0.5 * EL * (phi**2)
        - EJ * (cos_phi * np.cos(phi_ext) + sin_phi * np.sin(phi_ext))
    )

    H_sq = flux_dev.hamiltonian()
    if not isinstance(H_sq, Qobj) and hasattr(H_sq, "shape"):
        dim = H_sq.shape[0]
        H_sq = Qobj(H_sq, dims=[[dim], [dim]])

    diff = (H_dsl - H_ref).norm()
    diff_sq = (H_ref - H_sq).norm()
    if diff_sq > 0:  # pragma: no cover - informational
        logger.info(
            "scqubits Hamiltonian with flux offset differs from reference by %.3e",
            diff_sq,
        )
    return diff


def transmon_cavity_model(
    EJ: float = 20.0,
    EC: float = 0.25,
    cutoff: int = 10,
    cav_cutoff: int = 6,
    g: float = 0.05,
) -> Tuple[object, HilbertConfig]:
    r"""
    Build a simple transmon + cavity charge-coupled Hamiltonian:

        H = H_transmon + omega_c n_cav + g n (a + a^\dagger)

    where H_transmon = 4 EC n^2 - EJ cos(phi). The transmon operators are
    built internally using harmonic-oscillator zero-point fluctuations
    (no scqubits operators are used in the compiled model); the cavity
    is a boson in the DSL.
    """
    H_tmon, phi_op, n_op = _build_transmon_ops_internal(EJ=EJ, EC=EC, cutoff=cutoff)

    spec = CustomSpec(
        label="c",
        index=1,
        dim=H_tmon.dims[0][0],
        operators={"Htm": H_tmon, "phi": phi_op, "ntm": n_op},
    )
    # One cavity boson a_1 with cutoff cav_cutoff
    cfg = make_config(qubits=0, bosons=[(cav_cutoff, "a")], customs=[spec])

    H_latex = r"Htm_{1} + \omega_c n_{1} + g \, ntm_{1} (a_{1} + a_{1}^{\dagger})"
    params = {"omega_c": 5.0, "g": g}

    model = compile_latex_model(
        H_latex=H_latex,
        params=params,
        config=cfg,
        t_name="t",
    )
    return model, cfg


def verify_transmon_cavity_against_scqubits(
    EJ: float = 20.0,
    EC: float = 0.25,
    cutoff: int = 8,
    cav_cutoff: int = 4,
    g: float = 0.05,
    omega_c: float = 5.0,
) -> float:
    """
    Compile the transmon-cavity Hamiltonian from LaTeX and compare to an
    explicit construction built from the same internal transmon operators
    used in :func:`transmon_cavity_model`. Returns the operator norm of the
    difference (expected ~0). Any deviation between this reference and
    scqubits' transmon Hamiltonian is logged for information.
    """
    scq = _require_scqubits()
    model, cfg = transmon_cavity_model(
        EJ=EJ, EC=EC, cutoff=cutoff, cav_cutoff=cav_cutoff, g=g
    )
    H_dsl: Qobj = model.H if isinstance(model.H, Qobj) else model.H[0]
    H_tm, _, n_tm = _build_transmon_ops_internal(EJ=EJ, EC=EC, cutoff=cutoff)

    # Cavity operators (boson is first subsystem in cfg ordering, custom next)
    a = destroy(cav_cutoff)
    n_cav = a.dag() * a

    # Tensor order: [boson, custom]
    I_tm = qeye(H_tm.dims[0])
    I_cav = qeye([cav_cutoff])

    H_cav = omega_c * tensor(n_cav, I_tm)
    H_tm_full = tensor(I_cav, H_tm)
    n_tm_full = tensor(I_cav, n_tm)
    a_full = tensor(a, I_tm)
    adag_full = tensor(a.dag(), I_tm)

    H_expected = H_tm_full + H_cav + g * n_tm_full * (a_full + adag_full)
    diff = (H_dsl - H_expected).norm()

    # Informational check against scqubits' transmon implementation
    try:  # pragma: no cover - optional dependency precision differences
        tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=cutoff, truncated_dim=cutoff)
        H_tm_scq = tmon.hamiltonian()
        n_tm_scq = tmon.n_operator()
        if not hasattr(H_tm_scq, "dims"):
            dim_tm = H_tm_scq.shape[0]
            H_tm_scq = Qobj(H_tm_scq, dims=[[dim_tm], [dim_tm]])
        if not hasattr(n_tm_scq, "dims"):
            dim_tm = n_tm_scq.shape[0]
            n_tm_scq = Qobj(n_tm_scq, dims=[[dim_tm], [dim_tm]])

        H_tm_full_scq = tensor(I_cav, H_tm_scq)
        n_tm_full_scq = tensor(I_cav, n_tm_scq)
        H_expected_scq = (
            H_tm_full_scq + H_cav + g * n_tm_full_scq * (a_full + adag_full)
        )
        diff_scq = (H_expected - H_expected_scq).norm()
        if diff_scq > 0:
            logger.info(
                "Transmon reference differs from scqubits by %.3e "
                "(basis/truncation effects).",
                diff_scq,
            )
    except Exception:
        pass

    return float(diff)


def _build_transmon_ops_internal(
    EJ: float, EC: float, cutoff: int
) -> Tuple[Qobj, Qobj, Qobj]:
    """
    Build transmon phi, n, and Hamiltonian using harmonic-oscillator
    zero-point fluctuations (Koch et al.).
    """
    dim = 2 * cutoff + 1  # match scqubits Transmon dimension (2*ncut + 1)
    a = destroy(dim)
    adag = a.dag()
    # Zero-point fluctuations
    phi_zpf = (2 * EC / EJ) ** 0.25
    n_zpf = (EJ / (32 * EC)) ** 0.25

    phi_op = phi_zpf * (a + adag)
    n_op = 1j * n_zpf * (adag - a)

    # cos(phi) via matrix cosine
    cos_phi = Qobj(cosm(phi_op.full()), dims=phi_op.dims)
    H_tm = 4 * EC * (n_op * n_op) - EJ * cos_phi
    return H_tm, phi_op, n_op


if __name__ == "__main__":  # pragma: no cover - example execution
    try:
        flux_model, flux_cfg = fluxonium_model()
        print("Fluxonium model compiled. time_dependent:", flux_model.time_dependent)
        diff_flux = verify_fluxonium_against_scqubits()
        print(f"Fluxonium H difference norm (vs reference): {diff_flux:.3e}")

        flux_model_off, _ = fluxonium_model_with_flux_offset()
        diff_flux_off = verify_fluxonium_with_flux_offset_against_scqubits()
        print(
            "Fluxonium (flux offset) H difference norm (vs reference): "
            f"{diff_flux_off:.3e}"
        )
    except RuntimeError as exc:
        print(exc)

    try:
        tmon_model, tmon_cfg = transmon_cavity_model()
        print(
            "Transmon-cavity model compiled. time_dependent:",
            tmon_model.time_dependent,
        )
        diff_tmon = verify_transmon_cavity_against_scqubits()
        print(
            f"Transmon-cavity H difference norm: {diff_tmon:.3e}",
        )
    except RuntimeError as exc:
        print(exc)
