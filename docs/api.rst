API Reference
==============

.. contents::
   :local:
   :depth: 2

|

Overview
========

The **latex_parser** API is organized into five core modules:

1. **DSL** — LaTeX parsing, validation, and subsystem definitions
2. **IR** — Intermediate Representation (parsed model structure)
3. **Backend Base** — Abstract backend interfaces and contracts
4. **Backend Cache** — Shared operator caching and subsystem management
5. **Compile Core** — Main compilation pipeline and backend registry

Each module is **independent and inspectable**, making it easy to debug, extend, or integrate
individual components into your own workflows.

---

|

Core Modules
============

Domain-Specific Language (DSL)
------------------------------

The DSL module handles LaTeX parsing, canonicalization, validation, and subsystem definitions.

.. automodule:: latex_parser.dsl
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

* ``HilbertConfig`` — Define your quantum system structure (qubits, bosons, custom subsystems)
* ``QubitSpec``, ``BosonSpec``, ``CustomSpec`` — Subsystem definitions
* ``DSLValidationError`` — Raised when LaTeX is invalid or parameters are missing

**Key Functions:**

* ``canonicalize_physics_latex()`` — Rewrite physics notation to internal macros
* ``register_latex_pattern()`` — Add custom LaTeX rewrite rules
* ``register_operator_function()`` — Add custom operator-valued functions

|

Intermediate Representation (IR)
--------------------------------

The IR module parses LaTeX into a transparent data structure suitable for backend compilation.

.. automodule:: latex_parser.ir
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

* ``HamiltonianIR`` — The parsed model structure
* ``ScalarTerm`` — A ``(scalar_expr, operator_term)`` pair
* ``OperatorTerm`` — Ordered list of operators in the term

**Key Functions:**

* ``latex_to_ir()`` — Parse LaTeX to IR (main entry point for debugging)
* Time-dependence detection is automatic during parsing

**Why inspect the IR?**

The IR is **fully transparent**. You can examine it to:

* Verify the parser understood your LaTeX correctly
* Collect required parameters without compilation
* Check if time-dependence was detected
* Debug issues before sending to a backend

Example:

.. code-block:: python

    from latex_parser.ir import latex_to_ir
    from latex_parser.dsl import HilbertConfig, QubitSpec
    
    config = HilbertConfig(qubits=[QubitSpec(label="q", index=0)])
    ir = latex_to_ir(r"\omega \sigma_z", config)
    
    print(ir.terms)           # List of ScalarTerm objects
    print(ir.free_symbols)    # {'omega'} — required parameters
    print(ir.has_time_dep)    # False — static system

|

Backend Base Classes
--------------------

The backend_base module defines the abstract interfaces that all backends must follow.

.. automodule:: latex_parser.backend_base
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

* ``CoreBackend`` — Abstract base (not used directly; see backends)
* ``CompiledModel`` — Wrapper for backend results with metadata

**Why these exist:**

These define the **backend contract**. Custom backends should inherit from
appropriate base classes (see examples and :doc:`usage`).

|

Operator Cache (Shared Subsystem Management)
---------------------------------------------

The backend_cache module provides a base class for managing subsystem bookkeeping,
Kronecker products, and identity operators. All backends use this to ensure consistent
operator ordering and tensor-product construction.

.. automodule:: latex_parser.backend_cache
   :members:
   :undoc-members:
   :show-inheritance:

**Key Classes:**

* ``BaseOperatorCache`` — Abstract base for operator caching
* ``SubsystemInfo`` — Metadata for a single subsystem (qubit, boson, custom)

**Why use this?**

``BaseOperatorCache`` handles:

* Automatic qubit/boson/custom subsystem indexing
* Global identity construction (tensor product of all local identities)
* Kronecker-product ordering guarantees (critical for correctness!)

Custom backends should inherit from ``BaseOperatorCache`` to get this for free.
See ``examples/custom_backend.py`` for a minimal example.

|

Compilation Pipeline & Backend Registry
----------------------------------------

The compile_core module orchestrates the full LaTeX → IR → Backend pipeline
and manages the backend registry.

.. automodule:: latex_parser.compile_core
   :members:
   :undoc-members:
   :show-inheritance:

**Key Functions:**

* ``compile_model_core()`` — Main compilation function (used by latex_api)
* ``register_backend()`` — Register a new backend
* ``get_registered_backends()`` — List available backends

**Backend Registry:**

The registry is a central hub for:

* **Discovery** — ``get_registered_backends()`` to list available backends
* **Registration** — ``register_backend(name, fn)`` to add your own backend
* **Metadata** — Attach backend capabilities (optional)

Example:

.. code-block:: python

    from latex_parser.compile_core import register_backend, get_registered_backends
    
    # List available backends
    available = get_registered_backends()
    print(available)  # ["qutip", "numpy", "jax", ...]
    
    # Register your custom backend
    def my_backend(H_latex, params, config, **kwargs):
        # Your compilation logic here
        return my_result
    
    register_backend("mybackend", my_backend)
    
    # Now use it
    from latex_parser.latex_api import compile_model
    model = compile_model(H_latex=H, params=p, backend="mybackend")

---

|

User-Facing API (latex_api)
===========================

The main entry point for users is in ``latex_parser.latex_api``:

.. code-block:: python

    from latex_parser.latex_api import compile_model
    
    model = compile_model(
        H_latex=r"\omega \sigma_z",
        params={"omega": 1.0},
        qubits=1,
        backend="qutip",
    )

This function wraps all internal modules and provides a **single, simple interface** for 99% of use cases.

---

|

Common Workflows
================

**Parsing to IR (without backend compilation):**

.. code-block:: python

    from latex_parser.ir import latex_to_ir
    from latex_parser.dsl import HilbertConfig, QubitSpec
    
    config = HilbertConfig(qubits=[QubitSpec(label="q", index=0)])
    ir = latex_to_ir(r"\omega_0 \sigma_z + A \sigma_x", config)
    
    # Inspect the IR
    for term in ir.terms:
        print(f"Scalar: {term.scalar_expr}, Operators: {term.ops}")

**Validating LaTeX without compiling:**

.. code-block:: python

    from latex_parser.dsl import canonicalize_physics_latex
    
    H_latex = r"\sigma_z"
    canonical = canonicalize_physics_latex(H_latex)
    # Returns rewritten LaTeX (or raises DSLValidationError)

**Registering a custom backend:**

.. code-block:: python

    from latex_parser.compile_core import register_backend
    from latex_parser.backend_cache import BaseOperatorCache
    
    class MyBackendCache(BaseOperatorCache):
        def _local_identity(self, dim):
            return create_identity(dim)  # Your framework
        
        def _kron(self, a, b):
            return your_kron_product(a, b)
    
    def my_backend(H_latex, params, config, **kwargs):
        cache = MyBackendCache(config)
        # ... use cache to build operators ...
        return result
    
    register_backend("mybackend", my_backend)

**Creating a custom subsystem:**

.. code-block:: python

    from latex_parser.dsl import CustomSpec, HilbertConfig
    from latex_parser.latex_api import compile_model
    
    # Define a spin-3/2 system
    spin32 = CustomSpec(label="spin32", index=0, dim=4)
    
    model = compile_model(
        H_latex=r"\sigma_z",
        params={},
        customs=[spin32],
    )

---

|

API Stability & Versioning
===========================

**Stable APIs** (unlikely to change):

* ``compile_model()`` in ``latex_api`` — main user entry point
* ``HilbertConfig``, ``QubitSpec``, ``BosonSpec``, ``CustomSpec`` in ``dsl`` — subsystem definitions
* ``latex_to_ir()`` in ``ir`` — IR construction
* ``BaseOperatorCache`` in ``backend_cache`` — base for custom backends
* ``register_backend()`` in ``compile_core`` — backend registration

**Semi-stable APIs** (may evolve, but with deprecation warnings):

* IR structure (``HamiltonianIR``, ``ScalarTerm``) — parsing internals
* Backend base classes — abstract interfaces may be refined
* Operator function registry — new functions may be added

**Experimental APIs** (subject to change):

* Internal DSL parsing internals — implementation details
* Logging structures — may change for better diagnostics

---

|

Type Hints & IDE Support
========================

The codebase uses **Python type hints throughout** for excellent IDE support:

.. code-block:: python

    from latex_parser.latex_api import compile_model
    from typing import Dict
    
    H_latex: str = r"\omega \sigma_z"
    params: Dict[str, float] = {"omega": 1.0}
    
    model = compile_model(H_latex=H_latex, params=params, qubits=1)
    # IDE will auto-complete available methods and attributes!

---

|

Error Handling
==============

**Common exceptions:**

.. code-block:: python

    from latex_parser.dsl import DSLValidationError
    
    try:
        model = compile_model(H_latex=H, params=p, qubits=1)
    except DSLValidationError as e:
        # Missing parameters, invalid operators, syntax errors, etc.
        print(f"Invalid LaTeX: {e}")
    except ImportError as e:
        # Backend library not installed (e.g., jax)
        print(f"Missing dependency: {e}")
    except Exception as e:
        # Other compilation errors
        print(f"Compilation failed: {e}")

All errors include **detailed messages** with recovery suggestions.

---

|

**Next:** Return to :doc:`usage` for practical workflows, or :doc:`examples` for complete working examples.
