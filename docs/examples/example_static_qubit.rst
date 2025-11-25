Static Qubit Example
=====================

**Physics:** Compile a simple static two-level system (qubit) Hamiltonian.

**The Model**

In a typical quantum optics or AMO setup, you start with a static qubit:

.. math::

    H = \frac{\omega_0}{2} \sigma_z

where:

- :math:`\omega_0` is the qubit transition frequency (in units where :math:`\hbar = 1`)
- :math:`\sigma_z = |0\rangle\langle 0| - |1\rangle\langle 1|` is the Pauli Z operator
- The factor :math:`1/2` places the ground state at :math:`-\omega_0/2` and excited state at :math:`+\omega_0/2`

This is the **simplest possible Hamiltonian** — static (no time dependence), single qubit, no dissipation.

**What you'll learn**

- How to express a static Hamiltonian in LaTeX.
- How to call ``compile_model`` and inspect the returned object.
- The minimal inputs needed: a LaTeX string, parameters, and Hilbert space configuration.

**Code**

.. literalinclude:: ../../examples/example_static_qubit.py
   :language: python
   :linenos:

**Run it**

From the project root:

.. code-block:: bash

    python examples/example_static_qubit.py

**What happens**

1. The LaTeX string ``r"\frac{\omega_0}{2} \sigma_{z,1}"`` is canonicalized
2. It's parsed into an IR with one term: :math:`\frac{\omega_0}{2}` (scalar) × :math:`\sigma_z` (operator)
3. The parameter ``omega_0 = 2.0`` is substituted
4. QuTiP compiles it to a ``Qobj`` (quantum object):

   .. code-block:: text

       Quantum object: dims = [[2], [2]], shape = (2, 2), type = oper, isherm = True
       Qobj data =
       [[ 1.  0.]
        [ 0. -1.]]

**Try this**

- Change ``omega_0`` to other values (e.g., ``1.0``, ``10.0``) and re-run to see eigenvalues scale.
- Switch backend to ``"numpy"`` to get a dense ndarray instead of a ``Qobj``.
- Add a second qubit parameter (e.g., ``qubits=2``) and modify the LaTeX to ``r"\omega_1 \sigma_{z,1} + \omega_2 \sigma_{z,2}"`` to compile a two-qubit system.

**Next steps**

- See :doc:`example_time_dependent_drive` to add time-dependent drives.
- Check :doc:`example_collapse_ops` for dissipation (open-system dynamics).

