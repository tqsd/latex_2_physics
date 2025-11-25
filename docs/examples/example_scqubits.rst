SCQubits Example
=================

This example demonstrates interoperability with the `scqubits` package (if
available) and shows how to compose models that include non-standard qubit
implementations.

What you'll learn
------------------

- How to adapt latex_parser outputs to other quantum libraries.
- How to inspect and adapt operator shapes when integrating with third-party
  packages.

Source
------

.. literalinclude:: ../../examples/example_scqubits.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_scqubits.py

Notes
-----

- This example may require `scqubits` to be installed; it demonstrates
  integration patterns rather than a turnkey workflow.
