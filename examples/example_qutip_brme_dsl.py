# flake8: noqa
r"""
Reimplementation of the QuTiP ``brmesolve`` tutorial using the LaTeX DSL.

Original notebook steps (``qutip_example.ipynb``):
1) Constant Hamiltonian H = a^\dagger a for a truncated oscillator.
2) Add a time-dependent drive H(t) = H + sin(t)(a + a^\dagger).
3) Add simple dissipation with a decaying envelope.
4) Add a composite, explicitly time-dependent collapse operator.

This example shows how each Hamiltonian/collapse operator can be expressed
in LaTeX, compiled with the parser, and then passed directly to QuTiP
``brmesolve`` while keeping the rest of the workflow unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from latex_parser.latex_api import compile_model, make_config
from qutip import about, basis, mesolve, plot_expectation_values


def _number_op(cutoff: int):
    """Use the DSL to build the bosonic number operator."""
    cfg = make_config(bosons=[cutoff])
    model = compile_model(
        H_latex=r"n_{1}", params={}, config=cfg, c_ops_latex=None, backend="qutip"
    )
    return model.H


def compile_constant_oscillator(cutoff: int):
    cfg = make_config(bosons=[cutoff])
    H_latex = r"a_{1}^{\dagger} a_{1}"
    model = compile_model(
        H_latex=H_latex, params={}, config=cfg, c_ops_latex=None, backend="qutip"
    )
    return model.H


def compile_driven_oscillator(cutoff: int):
    cfg = make_config(bosons=[cutoff])
    H_latex = r"a_{1}^{\dagger} a_{1} + \sin(t) \left(a_{1} + a_{1}^{\dagger}\right)"
    model = compile_model(
        H_latex=H_latex, params={}, config=cfg, c_ops_latex=None, backend="qutip"
    )
    return model.H


def compile_decay_collapse_ops(cutoff: int, kappa: float):
    cfg = make_config(bosons=[cutoff])
    model = compile_model(
        H_latex=r"a_{1}^{\dagger} a_{1}",
        params={"kappa": kappa},
        config=cfg,
        c_ops_latex=[
            r"\sqrt{\kappa} \exp(-t/2) \, a_{1}",
            r"\sqrt{\kappa} \exp(-t/2) \, a_{1}^{\dagger}",
        ],
        backend="qutip",
    )
    return model.H, model.c_ops, model.args


def compile_composite_collapse_ops(cutoff: int, kappa: float):
    cfg = make_config(bosons=[cutoff])
    model = compile_model(
        H_latex=r"a_{1}^{\dagger} a_{1}",
        params={"kappa": kappa},
        config=cfg,
        c_ops_latex=[
            r"\sqrt{\kappa} \, a_{1} \exp(I t)",
            r"\sqrt{\kappa} \, a_{1}^{\dagger} \exp(-I t)",
        ],
        backend="qutip",
    )
    return model.H, model.c_ops, model.args


def main() -> None:
    N = 2
    psi0 = basis(N, N - 1)
    times = np.linspace(0, 10, 100)
    n_op = _number_op(N)

    # 1) Constant Hamiltonian
    H_const = compile_constant_oscillator(N)
    result_const = mesolve(H_const, psi0, times, e_ops=[n_op])
    plot_expectation_values(result_const, ylabels=["<n>"])

    # 2) Driven Hamiltonian with sin(t)
    H_drive = compile_driven_oscillator(N)
    result_drive = mesolve(H_drive, psi0, times, e_ops=[n_op])
    plot_expectation_values(result_drive, ylabels=["<n>"])

    # 3) Dissipation with decaying envelope (Bloch-Redfield a_ops form)
    kappa = 0.2
    H_decay, c_ops_decay, args_decay = compile_decay_collapse_ops(N, kappa)
    result_decay = mesolve(
        H_decay, psi0, times, c_ops=c_ops_decay, e_ops=[n_op], args=args_decay
    )
    plot_expectation_values(result_decay, ylabels=["<n>"])

    # 4) Composite time-dependent collapse operator
    H_comp, c_ops_comp, args_comp = compile_composite_collapse_ops(N, kappa)
    result_comp = mesolve(
        H_comp, psi0, times, c_ops=c_ops_comp, e_ops=[n_op], args=args_comp
    )
    plot_expectation_values(result_comp, ylabels=["<n>"])

    # Parity checks mirroring the notebook assertions
    assert np.allclose(result_const.expect[0], 1.0)
    assert np.all(np.diff(result_comp.expect[0]) <= 0.0)
    about()
    plt.show()


if __name__ == "__main__":
    main()
