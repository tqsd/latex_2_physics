Bosonic Number Example
======================

**Physics:** Compile a Hamiltonian with bosonic (cavity) modes using the number operator.

**The Model**

Cavity quantum electrodynamics (cavity QED) and circuit QED systems use bosonic modes to represent photons or resonator excitations. The bosonic Hilbert space is truncated to a finite cutoff:

.. math::

    H = \omega_c a^{\dagger} a + \omega_q \frac{\sigma_z}{2}

where:

- :math:`a` is the bosonic **lowering operator** (annihilation)
- :math:`a^{\dagger}` is the bosonic **raising operator** (creation)
- :math:`\hat{n} = a^{\dagger} a` is the **number operator** (photon count)
- :math:`\omega_c` is the cavity frequency
- :math:`\omega_q` is the qubit frequency

The bosonic Hilbert space is **truncated** at a cutoff (e.g., cutoff=10 means states :math:`|0\rangle, |1\rangle, \ldots, |9\rangle`).

**Example: Simple cavity QED**

.. math::

    H = \omega_c \hat{n} + \omega_q \frac{\sigma_z}{2} + g (a \sigma_+ + a^{\dagger} \sigma_-)

where :math:`g` is the coupling strength between the qubit and cavity.

**What you'll learn**

- How to declare bosonic modes with cutoffs
- How to use the number operator :math:`\hat{n}` and ladder operators
- How the total Hilbert space dimension scales with cutoff and number of subsystems

**Code**

.. literalinclude:: ../../examples/example_boson_number.py
   :language: python
   :linenos:

**Run it**

.. code-block:: bash

    python examples/example_boson_number.py

**What happens**

1. The ``bosons`` parameter specifies bosonic modes with a cutoff: ``bosons=[(10, "a")]`` means cutoff=10, label="a"
2. The parser creates operators :math:`a`, :math:`a^{\dagger}`, and :math:`\hat{n}` for use in the Hamiltonian
3. Each basis state is now a **tensor product** of qubit states and Fock states (truncated)
4. Total Hilbert space dimension: ``2^(qubits) × cutoff^(bosons)``

**Example Output**

.. code-block:: text

    Hilbert space:
    - 1 qubit (dim=2)
    - 1 boson with cutoff=10 (dim=10)
    - Total dimension: 2 × 10 = 20
    
    Hamiltonian (8 terms):
    Compiled to 20×20 matrix in QuTiP format

**Operator Reference**

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - LaTeX
     - Meaning
     - Notes
   * - ``a_{j}``
     - Lowering operator
     - Maps :math:`|n\rangle \to \sqrt{n}|n-1\rangle`
   * - ``a_{j}^{\dagger}``
     - Raising operator
     - Maps :math:`|n\rangle \to \sqrt{n+1}|n+1\rangle`
   * - ``\hat{n}_{j}`` or ``n_{j}``
     - Number operator
     - Equal to :math:`a^{\dagger} a`; eigenvalues 0, 1, 2, ...

**Try this**

- Increase the cutoff to 20: compare the computation time and Hilbert space dimension
- Add a second cavity: ``bosons=[(10, "a"), (10, "b")]`` and include cross-cavity coupling
- Use the number operator for nonlinear effects: ``\lambda n_1 (n_1 - 1)`` (Kerr nonlinearity)

**Photon Loss Example**

Cavities always lose photons. Add dissipation:

.. code-block:: python

    H = r"\omega_c a_{1}^{\dagger} a_{1}"
    c_ops = [r"\sqrt{\kappa} a_{1}"]  # Photon loss
    
    model = compile_model(
        H_latex=H,
        c_ops_latex=c_ops,
        params={"omega_c": 5.0, "kappa": 0.1},
        bosons=[(10, "a")]
    )

**Important Notes**

- **Cutoff selection is critical:** Too low → aliasing (unphysical), Too high → slow computation
- **Dimension grows exponentially:** With many modes, consider sparse representations (advanced topic)
- The number operator is useful for **nonlinear** and **measurement** effects

**Next steps**

- See :doc:`example_custom_subsystem` for more complex 3+ level systems
- Check :doc:`example_collapse_ops` for photon loss and other dissipation channels

