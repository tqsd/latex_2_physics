Advanced Operator Functions
===========================

This example explores operator-valued functions such as ``exp(A)``, ``cos(A)``,
and ``sqrtm(A)`` when applied to local operators. It explains the grammar
restrictions (single operator argument, optional integer power) and shows how
backends treat these constructs.

What you'll learn
------------------

- Allowed operator functions and their constraints.
- How operator functions are represented in the IR (``OperatorFunctionRef``).

Source
------

.. literalinclude:: ../../examples/example_operator_functions_advanced.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_operator_functions_advanced.py

Notes
-----

- Not all backends support the same operator functions; check backend docs.
- Use IR inspection to verify arguments and scalar factors for operator functions.
