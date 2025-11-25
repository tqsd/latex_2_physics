Custom Backend (Minimal) Example
================================

This example provides a minimal custom backend implementation that reuses the
shared caching utilities. It's a good template for implementing new numerical
backends or connecting to external libraries.

What you'll learn
------------------

- Which hooks a backend must implement (cache creation and compile methods).
- How to register a backend with ``register_backend``.

Source
------

.. literalinclude:: ../../examples/custom_backend.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/custom_backend.py

Notes
-----

- This example is intentionally minimal. Use it as the basis for a production
  backend by replacing the identity and kron implementations with optimized
  primitives.
