Backend Extensibility Example
=============================

This example walks through the backend registry, demonstrates how to discover
available backends and register a new one, and shows how capability metadata is
used to choose appropriate backends for specific tasks.

What you'll learn
------------------

- How to query the backend registry and read capability flags.
- How to register a new backend and make it available to ``compile_model``.

Source
------

.. literalinclude:: ../../examples/example_backend_extensibility.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_backend_extensibility.py

Notes
-----

- Use this example to scaffold integrations with specialized numerical libraries.
