=====================================
Welcome to latex_parser
=====================================

.. rst-class:: subtitle

    **Transform Physics into Code** — Write quantum Hamiltonians in LaTeX and run them on any backend.

What is latex_parser?
=====================

``latex_parser`` is a Python library that lets you express quantum models in **familiar physics-style LaTeX**
and automatically compile them to numerical backends. Instead of translating equations by hand into code,
you write what you'd put in a paper and the library handles the rest.

.. code-block:: python

    from latex_parser.latex_api import compile_model

    # Write physics in LaTeX
    H = r"\frac{\omega_0}{2} \sigma_z + A \cos(\omega t) \sigma_x"
    
    # Compile to a backend
    model = compile_model(
        H_latex=H,
        params={"omega_0": 1.0, "A": 0.5, "omega": 2.0},
        qubits=1,
        backend="qutip"
    )
    
    # Use with your solver
    from qutip import mesolve
    result = mesolve(model.H, psi0, times, c_ops=model.c_ops)

Key Features
============

✅ **Physics-style input**
    Write Hamiltonians exactly as you would in a paper: ``\sigma_{x,1}``, ``a_{j}^{\dagger}``, ``\cos(\omega t)``, etc.

✅ **Multiple backends**
    Same LaTeX → swap between QuTiP, NumPy, JAX, or your own custom backend without changing equations.

✅ **Structured, inspectable IR**
    The intermediate representation is transparent—debug parsing, validate parameters, and understand what the code is doing.

✅ **Open-system support**
    Hamiltonians and time-dependent collapse operators with full control over structure.

✅ **Advanced quantum features**
    Bosons with deformations, custom subsystems, operator functions, and more.

Who is this for?
================

- **Quantum physicists** who want to keep equations in LaTeX and avoid manual code translation.
- **Researchers** exploring multiple numerical backends without rewriting simulations.
- **Educators** teaching quantum mechanics with reproducible, equation-driven examples.
- **Developers** needing a flexible, extensible framework for quantum model compilation.

Getting Started in 3 Steps
===========================

**Step 1: Install**

See :doc:`install` for detailed instructions, or run:

.. code-block:: bash

    pip install latex-parser

**Step 2: Write your first model**

Create a simple qubit Hamiltonian:

.. code-block:: python

    from latex_parser.latex_api import compile_model
    
    # Simple Rabi oscillation
    H = r"A \cos(\omega t) \sigma_x"
    model = compile_model(
        H_latex=H,
        params={"A": 1.0, "omega": 2.0},
        qubits=1,
        backend="qutip"
    )
    print(f"Static term (H0): {model.H0}")
    print(f"Time-dependent term: {model.time_terms}")

**Step 3: Run and explore**

Execute simulations or inspect the compiled model:

.. code-block:: python

    from qutip import mesolve, basis
    
    # Initial state
    psi0 = basis(2, 0)  # Ground state
    
    # Time points
    times = np.linspace(0, 10, 100)
    
    # Simulate
    result = mesolve(model.H, psi0, times, c_ops=model.c_ops, args=model.args)
    
    # Analyze results
    print(result.expect[0])  # Expectation values

What Happens Behind the Scenes?
================================

The library uses a **three-stage compilation pipeline**:

.. image:: _static/pipeline.png
   :alt: LaTeX → IR → Backend pipeline
   :align: center

1. **LaTeX Canonicalization** — Physics notation is rewritten to an internal macro system
2. **Intermediate Representation (IR)** — Parsed into scalar coefficients and operator terms
3. **Backend Compilation** — IR is converted to backend-specific objects (QuTiP Qobj, NumPy arrays, JAX functions)

For details, see :doc:`usage` and the :ref:`compilation pipeline section <pipeline>` below.

Common Use Cases
================

**Time-dependent Hamiltonian (Rabi drive)**

.. code-block:: latex

    H(t) = \frac{\omega_0}{2} \sigma_z + A \cos(\omega t) \sigma_x

.. code-block:: python

    H = r"\frac{\omega_0}{2} \sigma_z + A \cos(\omega t) \sigma_x"
    model = compile_model(H_latex=H, params={"omega_0": 1.0, "A": 0.5, "omega": 2.0}, qubits=1)

**Open-system dynamics (decay)**

.. code-block:: latex

    H(t) = \omega_0 a^{\dagger} a
    
    c_1(t) = \sqrt{\kappa} a

.. code-block:: python

    H = r"\omega_0 a_{1}^{\dagger} a_{1}"
    c_ops = [r"\sqrt{\kappa} a_{1}"]
    model = compile_model(
        H_latex=H,
        c_ops_latex=c_ops,
        params={"omega_0": 1.0, "kappa": 0.1},
        bosons=[(10, "a")]
    )

**Multi-qubit entanglement (Jaynes-Cummings)**

.. code-block:: latex

    H = \omega_0 \sigma_{z,1} + \omega_c a^{\dagger} a + g (a \sigma_+ + a^{\dagger} \sigma_-)

.. code-block:: python

    H = r"\omega_0 \sigma_{z,1} + \omega_c a_{1}^{\dagger} a_{1} + g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})"
    model = compile_model(
        H_latex=H,
        params={"omega_0": 1.0, "omega_c": 2.0, "g": 0.1},
        qubits=1,
        bosons=[(5, "a")]
    )

Documentation Structure
=======================

.. toctree::
   :maxdepth: 1
   :caption: Start Here

   install
   usage

.. toctree::
   :maxdepth: 1
   :caption: Learn by Example

   examples

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

Backend Maturity & Features
============================

Not all backends support the same features. Here's a quick overview:

.. list-table::
   :widths: 30 15 15 15 25
   :header-rows: 1
   :align: center

   * - Feature
     - QuTiP
     - JAX
     - NumPy
     - Notes
   * - Static Hamiltonians
     - ✅ Mature
     - ✅ Mature
     - ✅ Mature
     - All backends support this
   * - Time-dependent Hamiltonians
     - ✅ Mature
     - ✅ Mature
     - ✅ Mature
     - Works with all backends
   * - Collapse operators (static)
     - ✅ Mature
     - ⚠️ Limited
     - ⚠️ Limited
     - QuTiP is optimized; others are basic
   * - Collapse operators (time-dependent)
     - ✅ Mature
     - ❌ Not supported
     - ❌ Not supported
     - QuTiP only
   * - Open-system solvers
     - ✅ Mature
     - ❌ Not included
     - ❌ Not included
     - Use QuTiP for Lindblad evolution
   * - Automatic differentiation
     - ⚠️ Limited
     - ✅ Mature
     - ❌ Not supported
     - JAX supports grad/vmap
   * - Custom backends
     - ✅ Easy
     - ✅ Easy
     - ✅ Easy
     - Implement ``BackendBase``
   * - Debugging & Inspection
     - ✅ Excellent
     - ✅ Good
     - ✅ Good
     - All support IR inspection

**Recommendation:**
  - For **open-system simulation** → Use **QuTiP**
  - For **automatic differentiation** → Use **JAX**
  - For **dense-matrix operations** → Use **NumPy** or **JAX**
  - For **custom workflows** → Implement a custom backend

Known Limitations
=================

Before you start, be aware of these constraints:

**DSL Restrictions**

- Operator functions (like ``\cos(\sigma_z)``) cannot contain operator sums inside them.
  ✅ Allowed: ``\cos(\sigma_z + n_1)`` (sum of scalars and single operators)
  ❌ Rejected: ``\cos(\sigma_x + \sigma_y)`` (sum of operators inside function)

- Time-dependent collapse operators must be **single monomials** (no sums).
  ✅ Allowed: ``\sqrt{\gamma} \exp(-t) \sigma_-``
  ❌ Rejected: ``\sqrt{\gamma_1} \exp(-t) \sigma_{-,1} + \sqrt{\gamma_2} \exp(-t) \sigma_{-,2}`` (use two separate strings)

- Large symbolic expansions (like ``(a + b)^{100}``) are capped at 512 terms to prevent memory blow-up.

**Backend Limitations**

- **QuTiP**: Best for open-system simulation; dense-matrix only; no GPU acceleration.
- **JAX**: Supports automatic differentiation but no built-in open-system solvers; requires manual implementation for dissipation.
- **NumPy**: Dense-matrix only; no optimization; good for prototyping and educational use.
- **Time-dependent collapse operators**: Only supported in QuTiP.

**Parameter Matching**

- Parameter names are matched with aliases (braces removed, spaces/asterisks stripped).
  For example, ``\omega_{c}`` matches ``omega_c``, ``omega_c1`` (first match wins).
  Ambiguous parameters can lead to unexpected bindings.

**Quantum Numbers & Truncation**

- Boson Hilbert spaces are truncated to a cutoff; choose carefully to avoid aliasing.
- No automatic validation that your cutoff is sufficient; always check convergence.

**Compilation Speed**

- Large systems (many qubits/bosons) with complex time-dependence can be slow to compile.
- Suggested: use static systems for development, then add time-dependence once validated.

Next Steps
==========

1. **Install the library**: :doc:`install`
2. **Learn the basics**: :doc:`usage`
3. **Explore examples**: :doc:`examples` (17 worked examples with code and explanations)
4. **Deep dive**: :doc:`api` (full API reference with dataclass descriptions)

Questions? Check the troubleshooting section in :doc:`usage` or open an issue on the repository.

---

**Ready?** Go to :doc:`install` now.
