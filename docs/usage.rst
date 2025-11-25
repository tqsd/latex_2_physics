Usage Guide
===========

.. contents::
   :local:
   :depth: 2

|

Quick Start (5 Minutes)
=======================

Let's compile your first quantum model from LaTeX in 5 minutes:

.. code-block:: python

    from latex_parser.latex_api import compile_model

    # 1. Define your Hamiltonian in LaTeX (as you would in a paper)
    H_latex = r"\frac{\omega_0}{2} \sigma_z + A \cos(\omega t) \sigma_x"
    
    # 2. List your parameters
    params = {
        "omega_0": 2.0,      # Transition frequency
        "A": 0.3,            # Drive amplitude
        "omega": 1.5,        # Drive frequency
    }
    
    # 3. Compile to your chosen backend (QuTiP by default)
    model = compile_model(
        H_latex=H_latex,
        params=params,
        qubits=1,            # One qubit
        backend="qutip",     # or "numpy", "jax"
    )
    
    # 4. Use the result
    print(f"Hamiltonian: {model.H}")
    print(f"Type: {type(model.H)}")  # QuTiP Qobj
    # For time-dependent models, model.args contains parameter dict

**What just happened?**

1. Your LaTeX was **canonicalized** — ``\sigma_z`` and other physics notation was converted to internal macros.
2. The DSL was **validated** — operators, functions, and syntax were checked.
3. An **Intermediate Representation (IR)** was built — the model was decomposed into ``(scalar_expr, operator_term)`` pairs.
4. **Parameters were validated** — all required symbols (``omega_0``, ``A``, ``omega``) were found and substituted.
5. The backend **compiled** the IR to a ``Qobj`` (QuTiP), ndarray (NumPy), or JAX function.

All in one line of Python! 🎉

|

Understanding the Compilation Pipeline
========================================

The **LaTeX → IR → Backend** pipeline has five distinct stages. Understanding each helps with debugging and extension:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────┐
    │ Stage 1: Canonicalize LaTeX                                 │
    │ Rewrite: a_{j}^†  →  \mop_adjoint{a}{j}                    │
    │         σ_{x,j}  →  \mop_sigma_x{j}                        │
    │         Ω_c      →  \Omega_{c}                              │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Stage 2: DSL Validation & Parsing                           │
    │ Check: Allowed operators, functions, subsystem definitions  │
    │ Parse: Split into terms, extract operators & scalars        │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Stage 3: Build Intermediate Representation (IR)             │
    │ Output: List of (SymPy_scalar_expr, OperatorTerm) pairs     │
    │         Time-dependence detected from free symbols in       │
    │         scalar expressions (look for t_name or time_symbols)│
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Stage 4: Validate & Collect Parameters                      │
    │ Find: All free symbols in scalar expressions                │
    │ Resolve: Aliases (ω_c ↔ omega_c ↔ omega c)                │
    │ Validate: All required params are present                   │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Stage 5: Backend Dispatch & Compilation                     │
    │ Route: To QuTiP, NumPy, JAX, or custom backend              │
    │ Compile: Each backend builds operators from IR using shared │
    │          operator cache (BaseOperatorCache)                 │
    └─────────────────────────────────────────────────────────────┘
                            ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ Output: Backend-specific object                             │
    │         QuTiP: Qobj (or list of Qobj)                       │
    │         NumPy: ndarray                                      │
    │         JAX: Parametrized function                          │
    │         Custom: Your type                                   │
    └─────────────────────────────────────────────────────────────┘

**Key insight:** The **IR is transparent and inspectable**. If something goes wrong,
you can examine each stage independently. See :doc:`examples` for IR debugging examples.

|

Parameter Handling & Aliases
=============================

LaTeX has many ways to write the same symbol. ``latex_parser`` normalizes them automatically:

**Alias equivalence rules:**

* ``\omega_c``, ``\omega_{c}``, ``omega_c``, ``omega_{c}``, ``\omega c``, ``omega c`` → all refer to the **same parameter**
* Underscores, braces, and spaces are ignored
* Case-sensitive: ``omega`` ≠ ``Omega``

**Example:**

.. code-block:: python

    H_latex = r"\omega_{c} \sigma_z + \Omega_R \sigma_x"
    
    # These all work:
    params_a = {"omega_c": 1.0, "Omega_R": 0.5}
    params_b = {"omega c": 1.0, "Omega R": 0.5}
    params_c = {r"\omega_{c}": 1.0, r"\Omega_R": 0.5}
    
    # All produce the same result!
    model_a = compile_model(H_latex=H_latex, params=params_a, qubits=1)
    model_b = compile_model(H_latex=H_latex, params=params_b, qubits=1)
    model_c = compile_model(H_latex=H_latex, params=params_c, qubits=1)

**What counts as a parameter?**

* **Scalars** — Free symbols in scalar expressions (``\omega``, ``A``, etc.)
* **NOT operators** — ``\sigma_z``, ``a^\dagger`` are not parameters
* **NOT constants** — ``\pi``, ``e``, integers (``2``, ``\frac{1}{2}``) are pre-defined

**Validation happens early:**

If you forget a parameter, you get a clear error:

.. code-block:: python

    H_latex = r"\omega_0 \sigma_z + \Omega \sigma_x"
    params = {"omega_0": 1.0}  # Forgot Omega!
    
    try:
        model = compile_model(H_latex=H_latex, params=params, qubits=1)
    except DSLValidationError as e:
        print(e)
        # Output: "Missing required parameter: Omega (in \Omega \sigma_x)"

|

Time-Dependent Hamiltonians
=============================

``latex_parser`` automatically detects time dependence in scalar expressions:

**Static Hamiltonian:**

.. code-block:: python

    H_latex = r"\omega_0 \sigma_z + A \sigma_x"
    model = compile_model(H_latex=H_latex, params={"omega_0": 1.0, "A": 0.5}, qubits=1)
    
    print(type(model.H))  # <class 'qutip.Qobj'> — a single static matrix

**Time-Dependent Hamiltonian (cosine envelope):**

.. code-block:: python

    H_latex = r"\omega_0 \sigma_z + A \cos(\omega t) \sigma_x"
    model = compile_model(
        H_latex=H_latex,
        params={"omega_0": 1.0, "A": 0.5, "omega": 2.0},
        t_name="t",  # Tell the parser what your time variable is called
        qubits=1
    )
    
    print(type(model.H))  # <class 'list'> — list of [H0, [H1, args_dict]]
    # Ready for mesolve with time-dependent Hamiltonian!

**How does time-dependence detection work?**

1. After building the IR, the parser looks at **free symbols in each scalar expression**
2. If the time variable (``t_name``, default is ``t``) appears, that term is **time-dependent**
3. Static and time-dependent terms are **separated** and returned in QuTiP format

**Supported time dependencies:**

* ``\cos(\omega t)``, ``\sin(\omega t)`` — oscillating envelopes
* ``\exp(-\gamma t)`` — exponential decay
* ``\sqrt{t}`` — smooth turn-on functions
* Any SymPy expression in ``t`` is supported

|

Collapse Operators
===================

For **open quantum systems**, specify collapse operators (Lindblad operators):

.. code-block:: python

    H_latex = r"\omega_0 \sigma_z + A \sigma_x"
    c_ops_latex = [
        r"\sqrt{\gamma_1} \sigma_{-}",           # Decay from excited to ground
        r"\sqrt{\gamma_2} (\sigma_x + i\sigma_y)/\sqrt{2}",  # Dephasing
    ]
    
    model = compile_model(
        H_latex=H_latex,
        c_ops_latex=c_ops_latex,
        params={
            "omega_0": 1.0, "A": 0.5,
            "gamma_1": 0.01,  # Decay rate
            "gamma_2": 0.005, # Dephasing rate
        },
        qubits=1,
        backend="qutip",  # Required for collapse operators
    )
    
    print(model.c_ops)  # List of Qobj collapse operators

**Important:** Collapse operators must contain at least one operator term. Scalar-only collapse strings are rejected.

|

Backend Selection
=================

Choose the right backend for your task:

**QuTiP (Default)**
    Best for: Master equations, time evolution, expectation values, measurements.
    
    .. code-block:: python
    
        model = compile_model(H_latex=H, params=p, qubits=1, backend="qutip")
        # Returns: QuTiP Qobj or list [H0, [H1, args_dict]]
        # Use with: qutip.mesolve, mcsolve, eigenenergies, etc.

**NumPy**
    Best for: Dense matrices, testing, educational use, checking results.
    
    .. code-block:: python
    
        model = compile_model(H_latex=H, params=p, qubits=1, backend="numpy")
        # Returns: numpy.ndarray (dense, potentially large)
        # Use with: numpy.linalg, scipy.sparse, etc.

**JAX**
    Best for: Automatic differentiation, GPU compilation, parameter optimization, batching.
    
    .. code-block:: python
    
        model = compile_model(H_latex=H, params=p, qubits=1, backend="jax")
        # Returns: JAX-compilable function (scalable to larger systems)
        # Use with: jax.grad, jax.vmap, jax.jit, etc.
        
        import jax
        def loss(params):
            H = model.evaluate(params)
            eigenvalues = jax.numpy.linalg.eigvalsh(H)
            return eigenvalues[0]  # Ground state energy
        
        grad_fn = jax.grad(loss)
        grads = grad_fn({"omega_0": 1.0, ...})

**Custom Backend**
    Best for: Integration with other frameworks, specialized computations.
    
    See the ``examples/custom_backend.py`` for a minimal template using ``BaseOperatorCache``.

|

Subsystems: Qubits, Bosons, Custom
===================================

Define your system's structure:

**Qubits only:**

.. code-block:: python

    model = compile_model(
        H_latex=r"\sigma_z",
        params={},
        qubits=2,  # Two qubits: indices 0, 1
    )

**Bosons (with Hilbert cutoff):**

.. code-block:: python

    model = compile_model(
        H_latex=r"\omega a^\dagger a",
        params={"omega": 1.0},
        bosons=[(0, 2)],  # Boson at mode 0, cutoff dimension 2
    )

**Mixed qubit-boson system:**

.. code-block:: python

    model = compile_model(
        H_latex=r"\omega \sigma_z + g (\sigma_+ a + \sigma_- a^\dagger)",
        params={"omega": 1.0, "g": 0.1},
        qubits=1,          # One qubit
        bosons=[(0, 4)],   # One boson, cutoff 4
    )

**Custom subsystems:**

.. code-block:: python

    from latex_parser.dsl import CustomSpec
    
    # Define a spin-3/2 system (dimension 4)
    spin32_spec = CustomSpec(
        label="spin32",
        index=0,
        dim=4,
        # Custom algebra can be added via register_operator_function
    )
    
    model = compile_model(
        H_latex=r"\sigma_z",  # Uses your custom algebra
        params={},
        customs=[spin32_spec],
    )

See ``examples/example_custom_subsystem.py`` for a complete walkthrough.

|

Backend Registry
================

The **backend registry** lets you register your own backends and discover available ones:

**Using the registry:**

.. code-block:: python

    from latex_parser.compile_core import register_backend, get_registered_backends
    
    # List available backends
    backends = get_registered_backends()
    print(backends)  # ["qutip", "numpy", "jax", ...]
    
    # Check if a backend is available
    if "mybackend" in backends:
        model = compile_model(H_latex=H, params=p, backend="mybackend")

**Registering a custom backend:**

.. code-block:: python

    from latex_parser.compile_core import register_backend
    
    def my_backend_fn(H_latex, params, config, c_ops_latex=None, t_name="t", time_symbols=None, **kwargs):
        """Your backend compilation logic."""
        # 1. Parse LaTeX → IR
        from latex_parser.ir import latex_to_ir
        ir = latex_to_ir(H_latex, config, t_name, time_symbols)
        
        # 2. Compile IR to your backend
        my_result = compile_my_backend(ir, params, config)
        
        # 3. Return backend-specific object
        return my_result
    
    # Register with the system
    register_backend("mybackend", my_backend_fn)
    
    # Now use it
    model = compile_model(H_latex=H, params=p, backend="mybackend")

See ``examples/custom_backend.py`` and ``examples/example_backend_extensibility.py`` for complete examples.

|

Extending the DSL
=================

Customize LaTeX patterns, operator functions, and more:

**Register custom LaTeX patterns:**

.. code-block:: python

    from latex_parser.dsl import register_latex_pattern
    
    # Example: Map \mathcal{L} to a custom operator
    register_latex_pattern(r"\mathcal{L}", "custom_L_operator")
    
    H_latex = r"\mathcal{L}"
    # Now your pattern is recognized

**Register custom operator functions:**

.. code-block:: python

    from latex_parser.dsl_constants import register_operator_function
    
    # Add support for \sinh (hyperbolic sine of an operator)
    # Allowed: exp, cos, sin, cosh, sinh, sqrtm (square root of matrix)
    register_operator_function("sinh", "sinh_of_operator")

**Add custom subsystems:**

See the subsystems section above and ``examples/example_custom_subsystem.py``.

|

Common Pitfalls & Troubleshooting
==================================

**Pitfall 1: Collapse operator is scalar-only**

.. code-block:: python

    # ❌ Wrong: c_ops_latex = [r"\gamma"]  # Just a number!
    # ✅ Right: c_ops_latex = [r"\sqrt{\gamma} \sigma_{-}"]  # Number × operator

**Pitfall 2: Forgetting a parameter**

.. code-block:: python

    H_latex = r"\omega_0 \sigma_z"
    params = {}  # Forgot omega_0!
    
    # → DSLValidationError: Missing required parameter: omega_0
    # Fix: params = {"omega_0": 1.0}

**Pitfall 3: Inconsistent alias naming**

.. code-block:: python

    H_latex = r"\omega_{c} \sigma_z"
    params = {"omega_C": 1.0}  # Case mismatch (c vs C)
    
    # → DSLValidationError: Missing required parameter: omega_c
    # Fix: Use consistent case or provide all variants

**Pitfall 4: Huge expansion from high powers**

.. code-block:: python

    H_latex = r"(a_1 + a_2 + a_3 + a_4)^{10}"  # Expands to ~1000 terms
    
    # → May hit MAX_EXPANDED_TERMS guard (default: 10000)
    # Fix: Simplify or use symbolic expansions where possible

**Pitfall 5: JAX backend requires JAX installed**

.. code-block:: python

    # pip install jax jaxlib first!
    model = compile_model(H_latex=H, params=p, backend="jax")
    # → ImportError if jax not found

**Pitfall 6: Time-dependence with wrong variable name**

.. code-block:: python

    H_latex = r"\Omega \cos(t)"
    # By default, t_name="t", so this is detected as time-dependent ✅
    
    # But:
    H_latex = r"\Omega \cos(\tau)"
    model = compile_model(H_latex=H_latex, params={}, qubits=1)
    # ❌ Treats \tau as a parameter, not a variable → Error: missing parameter
    
    # Fix: Specify t_name="tau"
    model = compile_model(H_latex=H_latex, params={}, qubits=1, t_name="tau")

|

Debugging & Inspection
======================

**Inspect the IR (Intermediate Representation):**

.. code-block:: python

    from latex_parser.ir import latex_to_ir
    from latex_parser.dsl import HilbertConfig, QubitSpec
    
    H_latex = r"\omega_0 \sigma_z + A \cos(\omega t) \sigma_x"
    config = HilbertConfig(qubits=[QubitSpec(label="q", index=0)])
    
    ir = latex_to_ir(H_latex, config, t_name="t")
    
    # Inspect the IR
    print(f"Terms: {len(ir.terms)}")
    for i, term in enumerate(ir.terms):
        print(f"Term {i}: scalar={term.scalar_expr}, ops={term.ops}")
    print(f"Time-dependent: {ir.has_time_dep}")
    print(f"Free symbols: {ir.free_symbols}")

**Collect parameters without compiling:**

.. code-block:: python

    from latex_parser.ir import latex_to_ir
    
    ir = latex_to_ir(H_latex, config, t_name="t")
    required_params = ir.free_symbols
    print(f"Required parameters: {required_params}")

**Check what your backend received:**

.. code-block:: python

    # Enable logging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    model = compile_model(H_latex=H, params=p, qubits=1)
    # Now you'll see detailed traces of each compilation stage

See ``examples/example_ir_debugging.py`` for a complete debugging walkthrough.

|

Backend Features & Maturity
============================

Different backends support different features. Choose based on your use case:

**QuTiP Backend (Recommended for open systems)**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Feature
     - Support
     - Notes
   * - Static Hamiltonians
     - ✅ **Mature**
     - Fully optimized, production-ready
   * - Time-dependent Hamiltonians
     - ✅ **Mature**
     - ``mesolve`` handles time-dependent lists natively
   * - Collapse operators (static)
     - ✅ **Mature**
     - Optimized for Lindblad simulation
   * - Collapse operators (time-dependent)
     - ✅ **Mature**
     - Full support; QuTiP-only feature
   * - Open-system solvers
     - ✅ **Mature**
     - ``mesolve``, ``mcsolve`` included
   * - Automatic differentiation
     - ⚠️ **Limited**
     - Manual diff only; use JAX for autodiff
   * - Custom backends
     - ✅ **Easy**
     - Subclass ``BackendBase``
   * - Debugging
     - ✅ **Excellent**
     - Full IR inspection, easy troubleshooting

**Use QuTiP when:**
  - Simulating open systems (dissipation, decay)
  - You need collapse operators (especially time-dependent)
  - Speed and stability are priorities
  - You want built-in solvers

**JAX Backend (For automatic differentiation)**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Feature
     - Support
     - Notes
   * - Static Hamiltonians
     - ✅ **Mature**
     - JAX arrays, fast and differentiable
   * - Time-dependent Hamiltonians
     - ✅ **Mature**
     - Parametrized functions, easy grad/vmap
   * - Collapse operators (static)
     - ⚠️ **Limited**
     - Returns array, not integrated solver
   * - Collapse operators (time-dependent)
     - ❌ **Not supported**
     - Use custom solver or QuTiP
   * - Open-system solvers
     - ❌ **Not included**
     - JAX includes only Hamiltonian compilation
   * - Automatic differentiation
     - ✅ **Mature**
     - Full support: ``grad``, ``vmap``, ``jit``
   * - Custom backends
     - ✅ **Easy**
     - Subclass ``BackendBase``
   * - Debugging
     - ✅ **Good**
     - IR inspection works; JAX tracing may limit prints

**Use JAX when:**
  - Optimizing Hamiltonian parameters (pulse sequences, gate fidelities)
  - You need automatic differentiation (gradient-based optimization)
  - GPU acceleration is required
  - Closed-system simulations only

**NumPy Backend (For prototyping)**

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Feature
     - Support
     - Notes
   * - Static Hamiltonians
     - ✅ **Mature**
     - Standard NumPy arrays
   * - Time-dependent Hamiltonians
     - ✅ **Mature**
     - Returns parametrized function
   * - Collapse operators (static)
     - ⚠️ **Limited**
     - Returns arrays, no integration
   * - Collapse operators (time-dependent)
     - ❌ **Not supported**
     - Would require custom solver
   * - Open-system solvers
     - ❌ **Not included**
     - Dense-matrix only
   * - Automatic differentiation
     - ❌ **Not supported**
     - Use JAX instead
   * - Custom backends
     - ✅ **Easy**
     - Reference implementation
   * - Debugging
     - ✅ **Good**
     - Simple, easy to inspect arrays

**Use NumPy when:**
  - Prototyping new models quickly
  - You only need dense Hamiltonian matrices
  - Learning the library (simplest backend)
  - Educational use

|

Known Limitations
=================

Before you start, be aware of these constraints and gotchas:

**DSL Constraints**

1. **Operator functions cannot contain operator sums inside**

   ✅ **Allowed:**
   
   .. code-block:: latex
   
       \cos(\sigma_z + \alpha)      % Scalar sum + single operator OK
       \sin(\sigma_z)               % Single operator OK
   
   ❌ **Rejected:**
   
   .. code-block:: latex
   
       \cos(\sigma_x + \sigma_y)    % Operator sum inside function
       \exp(\sigma_z \sigma_x)      % Multiple operators
   
   **Workaround:** Expand manually or use the IR directly.

2. **Time-dependent collapse operators must be single monomials**

   ✅ **Allowed:**
   
   .. code-block:: latex
   
       \sqrt{\gamma} \exp(-t/2) \sigma_-    % Single operator term
   
   ❌ **Rejected:**
   
   .. code-block:: latex
   
       \sqrt{\gamma_1} \sigma_{-,1} + \sqrt{\gamma_2} \sigma_{-,2}    % Sum
   
   **Workaround:** Use two separate collapse operator strings.

3. **Symbolic expansions are capped at 512 terms**

   Large powers like :math:`(a + b)^{100}` will raise an error to prevent memory blow-up.
   
   **Workaround:** Expand by hand or rewrite the Hamiltonian.

**Backend Limitations**

1. **QuTiP is dense-matrix only** — Large systems (>15 qubits) become slow
2. **JAX: No built-in dissipation solvers** — Open systems require custom integration
3. **NumPy: No optimization** — Slowest backend for production use
4. **Time-dependent collapse operators: QuTiP-only** — JAX and NumPy don't support

**Parameter Handling**

1. **Ambiguous parameter aliases** — If you have both ``omega_c`` and ``omega_c1``, the first match wins
   
   .. code-block:: python
   
       params = {"omega_c": 1.0, "omega_c1": 2.0}
       # Which does \omega_c match? Answer: the first in the params dict (ambiguous!)
   
   **Solution:** Use unambiguous names or inspect ``ir.free_symbols`` to debug

2. **Parameter validation is strict** — Missing or misnamed parameters raise early
   
   .. code-block:: python
   
       H = r"\omega \sigma_z"
       params = {"omega": 1.0}
       # But your LaTeX uses \omega_0, not \omega?
       # → DSLValidationError: "Missing numeric value for omega_0"
   
   **Solution:** Check parameter names carefully and use aliases

**Quantum Numbers**

1. **Boson cutoffs are **not** automatically validated**
   
   If your system has fast oscillations, a cutoff=5 may be too small.
   
   **Solution:** Check convergence by increasing cutoff and comparing results

2. **No automatic dimension checking**
   
   You can accidentally create an over-truncated system. The library won't warn you.

**Compilation Speed**

1. **Large systems are slow** — Many qubits + complex time-dependence can take seconds
   
   Example: 10 qubits + 5 time-dependent terms ≈ 2–5 seconds to compile
   
   **Solution:** Use static systems while developing, add time-dependence once validated

2. **Symbolic simplification is deferred** — Complex expressions aren't simplified before backend dispatch

**Backend Differences**

The same LaTeX may produce different numerical results with different backends due to:
  - Floating-point precision differences
  - Different truncation strategies (for sparse representations, if used)
  - JAX JIT compilation precision

**Solution:** Test with QuTiP first, then switch backends

**Type System**

The library uses Python type hints. IDEs should help, but:
  - Custom backends must implement the ``BackendBase`` interface
  - Errors from incorrect types are caught at runtime, not compile-time

---

**Next:** Browse :doc:`examples` for more workflows, or jump to :doc:`api` for detailed reference.
