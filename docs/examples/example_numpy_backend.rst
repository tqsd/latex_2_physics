NumPy Backend Example
======================

This example demonstrates compiling LaTeX-defined models to dense NumPy
matrices by reusing the QuTiP backend and converting Qobj objects to arrays.
It is useful for small systems and quick prototyping.

What you'll learn
------------------

- How to request the NumPy backend and inspect the resulting ``ndarray``.
- How time-dependent terms are represented for NumPy backends.

Source
------

.. literalinclude:: ../../examples/example_numpy_backend.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_numpy_backend.py

Notes
-----

- The example converts QuTiP objects to NumPy arrays; it is intended for
  clarity and education rather than performance.
