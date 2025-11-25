#!/usr/bin/env python
# coding: utf-8
# flake8: noqa

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qutip import (
    about,
    basis,
    destroy,
    mesolve,
    qeye,
    tensor,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from latex_parser.latex_api import compile_model as compile_latex_model


def run_vacuum_rabi():
    """
    Reproduce the QuTiP vacuum Rabi oscillations tutorial, but build
    H and c_ops from your LaTeX DSL instead of manual Qobj expressions.
    """
    # === Problem parameters (same physics as the QuTiP example) ===
    N = 15  # number of cavity Fock states
    omega_c = 1.0 * 2 * np.pi  # \omega_c: cavity frequency
    omega_a = 1.0 * 2 * np.pi  # \omega_a: atom frequency
    g = 0.05 * 2 * np.pi  # g: coupling strength
    kappa = 0.005  # \kappa: cavity dissipation rate
    gamma = 0.05  # \gamma: atom dissipation rate
    n_th_a = 0.0  # thermal photon number (environment)
    use_rwa = True

    tlist = np.linspace(0, 40, 100)

    # === Build model at T = 0 using the simple API ===
    if use_rwa:
        H_latex = r"""
            \omega_c \hat{n}_{1}
            + \frac{\omega_a}{2} \sigma_{z,1}
            + g \left( a_{1}^{\dagger} \sigma_{-,1}
                     + a_{1} \sigma_{+,1} \right)
        """
    else:
        H_latex = r"""
            \omega_c \hat{n}_{1}
            + \frac{\omega_a}{2} \sigma_{z,1}
            + g \left( a_{1}^{\dagger} + a_{1} \right)
                \left( \sigma_{-,1} + \sigma_{+,1} \right)
        """

    c_ops_latex = [
        r"\sqrt{\kappa (1 + n_{th})} \, a_{1}",
        r"\sqrt{\kappa n_{th}} \, a_{1}^{\dagger}",
        r"\sqrt{\gamma} \, \sigma_{-,1}",
    ]

    params = {
        "omega_c": omega_c,
        "omega_a": omega_a,
        "g": g,
        "kappa": kappa,
        "gamma": gamma,
        "n_th": n_th_a,
    }

    compiled_0 = compile_latex_model(
        H_latex,
        params,
        c_ops_latex=c_ops_latex,
        qubits=1,
        bosons=[N],
        t_name="t",
    )

    H_0 = compiled_0.H  # Qobj (static)
    c_ops_0 = compiled_0.c_ops
    args_0 = compiled_0.args  # currently just the params dict

    # === Initial state ===
    # Hilbert ordering: [qubit, boson] -> 2 ⊗ N
    # "Atom excited, cavity ground" -> |e> ⊗ |0>.
    # QuTiP tutorial uses tensor(basis(N, 0), basis(2, 0)) with different ordering;
    # we swap to match [qubit, boson].
    psi0 = tensor(basis(2, 0), basis(N, 0))

    # === Operators for expectation values (cavity & atom excitation) ===
    # Cavity annihilation in our ordering: I_qubit ⊗ a
    a = tensor(qeye(2), destroy(N))
    n_cav = a.dag() * a

    # Atomic excitation projector (σ_+ σ_- in our ordering)
    sm_raising = tensor(destroy(2).dag(), qeye(N))  # σ_+
    atom_excited_proj = sm_raising.dag() * sm_raising  # σ_- σ_+

    e_ops = [n_cav, atom_excited_proj]

    # === Evolve at T = 0 ===
    output_0 = mesolve(H_0, psi0, tlist, c_ops_0, e_ops, args=args_0)

    # === Now finite temperature: n_th_a > 0 ===
    n_th_a = 2.0

    params_T = dict(params)
    params_T["n_th"] = n_th_a
    compiled_T = compile_latex_model(
        H_latex,
        params_T,
        c_ops_latex=c_ops_latex,
        qubits=1,
        bosons=[N],
        t_name="t",
    )

    H_T = compiled_T.H
    c_ops_T = compiled_T.c_ops
    args_T = compiled_T.args

    output_T = mesolve(H_T, psi0, tlist, c_ops_T, e_ops, args=args_T)

    # === Plot T = 0 ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tlist, output_0.expect[0], label="Cavity")
    ax.plot(tlist, output_0.expect[1], label="Atom excited state")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Occupation probability")
    ax.set_title("Vacuum Rabi oscillations at T = 0")
    plt.tight_layout()
    plt.show()

    # === Plot T > 0 ===
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tlist, output_T.expect[0], label="Cavity")
    ax.plot(tlist, output_T.expect[1], label="Atom excited state")
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Occupation probability")
    ax.set_title(f"Vacuum Rabi oscillations at T = {n_th_a}")
    plt.tight_layout()
    plt.show()

    # === Optional: analytic test in RWA, no collapse operators ===
    if use_rwa:
        output_no_cops = mesolve(H_0, psi0, tlist, [], e_ops, args=args_0)
        freq = 0.25 * np.sqrt(g**2 * (N + 1))
        atom_pop = np.array(output_no_cops.expect[1])
        analytic = (np.cos(tlist * freq)) ** 2
        assert np.allclose(atom_pop, analytic, atol=1e-3)

    print()
    print("QuTiP / environment info:")
    about()


if __name__ == "__main__":
    run_vacuum_rabi()
