Time-Dependent Drive Example
============================

**Physics:** Compile a Rabi-oscillation Hamiltonian with a time-dependent drive.

**The Model**

The classic Rabi problem combines a static qubit splitting with a resonant oscillating drive:

.. math::

    H(t) = \frac{\omega_0}{2} \sigma_z + \Omega \cos(\omega_d t) \sigma_x

where:

- :math:`\omega_0` is the qubit transition frequency
- :math:`\Omega` is the drive Rabi frequency (drive strength)
- :math:`\omega_d` is the drive frequency
- :math:`\sigma_x` is the Pauli X operator (in-plane rotation)
- The :math:`\cos(\omega_d t)` envelope creates an oscillating drive

This is the foundational model for qubit gates in superconducting circuits, trapped ions, and other platforms.

**What you'll learn**

- How to write time-dependent envelopes in LaTeX (``\cos(\omega t)``, ``\sin(\omega t)``, ``\exp(-t)``, etc.)
- How the parser **automatically detects** time dependence
- How to extract the time-dependent Hamiltonian and ``args`` dictionary for use with QuTiP's ``mesolve``

**Code**

.. literalinclude:: ../../examples/example_time_dependent_drive.py
   :language: python
   :linenos:

**Run it**

.. code-block:: bash

    python examples/example_time_dependent_drive.py

**What happens**

1. The LaTeX includes the symbol ``t`` (the time variable), so the parser **marks this as time-dependent**
2. The IR distinguishes between:
   - **Static terms:** :math:`\frac{\omega_0}{2} \sigma_z` (depends on :math:`\omega_0` only)
   - **Time-dependent terms:** :math:`\Omega \cos(\omega_d t) \sigma_x` (depends on :math:`t`)
3. QuTiP receives the Hamiltonian in list form: ``[H0, [H1, envelope_fn]]``
4. Each envelope is a callable ``fn(t, args)`` that the solver evaluates at each time step

**Example Output**

.. code-block:: text

    Static part (H0):
    [[0.  0. ]
     [0.  1. ]]
    
    Time-dependent part (H1):
    [[0.  1.]
     [1.  0.]]
    
    Envelope function: <function ...>
    
    # When mesolve evaluates at time t=1.5:
    envelope_fn(1.5, {}) = 0.07073...  (= cos(2*1.5))

**Try this**

- Change the envelope to :math:`\sin(\omega_d t)` in the LaTeX and re-run
- Add exponential decay: :math:`\Omega \exp(-t/\tau) \cos(\omega_d t) \sigma_x` (turn-off envelope)
- Modify :math:`\omega_d` to be off-resonance: change the parameter from ``2.0`` to ``3.0``

**Numerical Simulation Example**

To actually solve the Rabi equations:

.. code-block:: python

    from qutip import mesolve, basis
    import numpy as np
    
    # Compile the model (from the example)
    # ... model = compile_model(...)
    
    # Initial state: ground state
    psi0 = basis(2, 0)
    
    # Time points (0 to 2π / Ω ≈ 6.3 seconds for Rabi period)
    times = np.linspace(0, 2*np.pi, 100)
    
    # Solve the Schrödinger equation
    result = mesolve(model.H, psi0, times, args=model.args)
    
    # Analyze: Rabi oscillations between ground and excited states
    excited_pop = [e[1] for e in result.expect[0]]

**Next steps**

- See :doc:`example_collapse_ops` to add dissipation (decay channels)
- Check :doc:`example_jax_autodiff_workflow` for automatic differentiation of pulse sequences

