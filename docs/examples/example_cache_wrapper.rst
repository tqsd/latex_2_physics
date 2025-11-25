Operator Cache & Wrapper Example
================================

This example explores the operator cache and helper wrappers that reduce
recomputation when compiling multiple related operators. It's useful for
understanding how to build custom backends that reuse precomputed local
operators.

What you'll learn
------------------

- How ``BaseOperatorCache`` organizes subsystem order and identities.
- How to fetch local and global (tensor-embedded) operators efficiently.

Source
------

.. literalinclude:: ../../examples/example_cache_wrapper.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_cache_wrapper.py

Notes
-----

- Useful when building custom backends to avoid repeated Kronecker products.
