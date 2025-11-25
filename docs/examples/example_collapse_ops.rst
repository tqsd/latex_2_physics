Collapse Operators (Open Systems)
=================================

**Physics:** Compile a Hamiltonian with Lindblad collapse operators for open-system dissipation.

**The Model**

In realistic quantum systems, dissipation and decoherence occur due to coupling with the environment. The Lindblad master equation describes the evolution:

.. math::

    \dot{\rho}(t) = -i[H(t), \rho(t)] + \sum_i \left( c_i(t) \rho c_i(t)^{\dagger} - \frac{1}{2} \{c_i(t)^{\dagger} c_i(t), \rho\} \right)

where:

- :math:`H(t)` is the system Hamiltonian
- :math:`c_i(t)` are the **collapse operators** (Lindblad operators) describing dissipation channels
- Common examples: :math:`c = \sqrt{\kappa} a` (photon loss), :math:`c = \sqrt{\gamma} \sigma_-` (qubit decay)

**Example: Damped Rabi oscillations**

.. math::

    H(t) = \frac{\omega_0}{2} \sigma_z + \Omega \cos(\omega_d t) \sigma_x
    
    c_1 = \sqrt{\gamma} \sigma_-  \quad \text{(qubit decay)}
    
    c_2 = \sqrt{\gamma_\phi} \sigma_z  \quad \text{(dephasing)}

where :math:`\gamma` is the decay rate and :math:`\gamma_\phi` is the dephasing rate.

**What you'll learn**

- How to write collapse operators in LaTeX using :math:`\sqrt{\text{rate}} \times \text{operator}` notation
- How the parser compiles them into the backend's format
- How to use them with QuTiP's ``mesolve`` for open-system simulations

**Code**

.. literalinclude:: ../../examples/example_collapse_ops.py
   :language: python
   :linenos:

**Run it**

.. code-block:: bash

    python examples/example_collapse_ops.py

**What happens**

1. Each collapse operator string is parsed independently
2. Static operators become ``Qobj`` instances
3. Time-dependent operators (e.g., :math:`\sqrt{\gamma(t)} \sigma_-`) become tuples ``[operator, envelope_function]``
4. The model returns ``model.c_ops`` as a list ready for ``mesolve``

**Example Output**

.. code-block:: text

    Hamiltonian (static + time-dependent):
    [H0, [H1, f_envelope]]
    
    Collapse operators:
    [Qobj(decay), Qobj(dephasing)]
    
    # Use with mesolve:
    result = mesolve(model.H, psi0, times, c_ops=model.c_ops, args=model.args)

**Full Simulation Example**

.. code-block:: python

    from qutip import mesolve, basis, expect, sigmaz
    import numpy as np
    
    # Compile the model
    # ... model = compile_model(...)
    
    # Initial state
    psi0 = basis(2, 0)  # Ground state
    
    # Time evolution
    times = np.linspace(0, 10, 100)
    
    # Open-system evolution with dissipation
    result = mesolve(
        model.H,
        psi0,
        times,
        c_ops=model.c_ops,
        e_ops=[sigmaz()],
        args=model.args
    )
    
    # Plot the decay of excited state population
    import matplotlib.pyplot as plt
    plt.plot(times, result.expect[0])
    plt.xlabel('Time')
    plt.ylabel('⟨σ_z⟩')
    plt.title('Rabi oscillations with decay')
    plt.show()

**Try this**

- Remove the dephasing operator and re-run to see faster oscillations
- Increase the decay rate :math:`\gamma` to see damping accelerate
- Add time-dependent decay: :math:`c = \sqrt{\gamma(t)} \sigma_-` with :math:`\gamma(t) = \gamma_0 \exp(-t/\tau)` (turn-on dissipation)

**Important Notes**

- **Static collapse operators** work with all backends (QuTiP, NumPy, JAX)
- **Time-dependent collapse operators** are **QuTiP-only** (due to ``mesolve`` constraints)
- Each static collapse operator reduces by one order on the total state (density matrix formalism)
- For efficient simulation, keep the number of collapse operators small

**Next steps**

- See :doc:`example_boson_number` for bosonic systems with photon loss
- Check :doc:`example_jax_autodiff_workflow` for optimizing dissipation rates via autodiff

