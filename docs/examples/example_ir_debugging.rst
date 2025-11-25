IR Debugging and Inspection
===========================

This example shows how to inspect the Intermediate Representation (IR) produced
by the parser. It is useful when debugging parsing issues, parameter
resolution, or operator recognition.

What you'll learn
------------------

- How to call ``latex_to_ir`` and inspect ``ir.terms`` and ``ir.has_time_dep``.
- How to use IR inspection to diagnose missing parameters and operator parsing.

Source
------

.. literalinclude:: ../../examples/example_ir_debugging.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_ir_debugging.py

Notes
-----

- The IR is the authoritative representation of what the parser understood.
- Use it to confirm that symbols you expected to be operators are not
  accidentally treated as scalars, and vice versa.
