JAX Autodiff Workflow
======================

This example demonstrates using the JAX backend to obtain parametrized
functions suitable for `jax.grad`, `jax.vmap`, and `jax.jit`. It's aimed at
users who want to perform parameter optimization or batched evaluations.

What you'll learn
------------------

- How to compile a model for the JAX backend.
- How to use the returned functions in JAX workflows (grad, vmap, jit).

Source
------

.. literalinclude:: ../../examples/example_jax_autodiff_workflow.py
   :language: python
   :linenos:

Run
---

.. code-block:: bash

    python examples/example_jax_autodiff_workflow.py

Notes
-----

- JAX must be installed and available in your environment for this example to
  run. If JAX is not present the example will either skip or raise.
