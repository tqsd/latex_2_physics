Custom Subsystem Example
========================

This example shows how to create and use a custom finite-dimensional subsystem
(for example, a spin greater than 1/2) by supplying a ``CustomSpec`` with named
operators.

What you'll learn
------------------

- How to provide a ``CustomSpec`` to ``make_config`` or ``compile_model``.
- How custom operator names are referenced from LaTeX (e.g., ``Jx_{1}``).

Source
------

.. literalinclude:: ../../examples/example_custom_subsystem.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_custom_subsystem.py

Notes
-----

- The DSL only checks that operator names exist in the provided template;
  values are opaque backend objects.
- Use this pattern when modeling collective spins or other non-qubit subsystems.
