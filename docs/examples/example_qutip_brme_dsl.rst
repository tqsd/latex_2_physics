QuTiP BRME DSL Example
======================

This example (`example_qutip_brme_dsl.py`) demonstrates a more specialized
workflow (BRME-style notation) and shows how the DSL maps physics-style LaTeX
into operator references consumed by the QuTiP backend.

What you'll learn
------------------

- How to express BRME-style interaction terms in LaTeX.
- How the DSL canonicalizes and maps those operators to QuTiP objects.

Source
------

.. literalinclude:: ../../examples/example_qutip_brme_dsl.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_qutip_brme_dsl.py

Notes
-----

- This example is targeted at users integrating domain-specific LaTeX
  conventions (BRME) with the parser. Inspect IR output to verify mappings.
