Parameter Validation and Aliases
================================

This example demonstrates how parameter names are resolved via aliasing rules
(braces, underscores, and spaces) and how the library validates that all
required numeric parameters are present before backend compilation.

What you'll learn
------------------

- The aliasing behavior for parameter names and how to supply values in ``params``.
- How missing parameters produce informative ``DSLValidationError`` messages.

Source
------

.. literalinclude:: ../../examples/example_param_validation_and_aliases.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_param_validation_and_aliases.py

Notes
-----

- Use this example when you encounter parameter lookup failures; it shows the
  candidate names the library will try.
