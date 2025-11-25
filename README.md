# latex_parser: Physics LaTeX to Quantum Simulation

Write open quantum system models in **physics-style LaTeX** and compile them to numerical backends (QuTiP, JAX, NumPy).

```python
from latex_parser.latex_api import compile_model

# Write physics, not code (index subsystems explicitly)
H = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}"

# Get a compiled model
model = compile_model(
    H_latex=H,
    params={"omega_0": 2.0, "A": 0.5, "omega": 1.0},
    qubits=1,
    backend="qutip"
)

# Use with any solver
from qutip import mesolve, basis
result = mesolve(model.H, basis(2, 0), times, args=model.args)
```

---

## Quick Links

📖 **[Documentation](https://latex-parser.readthedocs.io/)** — Full guides, API reference, 17 examples
💾 **[Installation](https://latex-parser.readthedocs.io/en/latest/install.html)** — `pip install latex-parser`
🎯 **[Quick Start](https://latex-parser.readthedocs.io/en/latest/welcome.html)** — Welcome page with overview
📝 **[Usage Guide](https://latex-parser.readthedocs.io/en/latest/usage.html)** — Detailed tutorials

---

## The Pipeline: LaTeX → IR → Backend

```
Physics LaTeX          Intermediate Rep.         Backend-ready
─────────────────────────────────────────────────────────────
H = ω σ_z +       H_IR:                         QuTiP Qobj
    A cos(ωt)σ_x  - terms: [...]          →    NumPy array
    + c_ops       - free_symbols: {ω,A}        JAX function
                   - time_dependent: true       Custom backend
```

1. **LaTeX Canonicalization** — Rewrite physics notation to internal macros
2. **Intermediate Representation (IR)** — Parse into scalar coefficients × operator terms
3. **Backend Dispatch** — Compile IR to backend-specific objects

---

## What Can You Do?

✅ **Write physics in LaTeX:**
  - Qubits: `\sigma_{x,1}`, `\sigma_{z,2}`, `\sigma_{\pm,j}`
  - Bosons: `a_{j}`, `a_{j}^{\dagger}`, `\hat{n}_{j}`
  - Custom systems: `J_{x,1}`, `\phi_{1}`
  - Time-dependence: `\cos(\omega t)`, `\exp(-t/2)`

✅ **Swap backends** without changing equations:
  - QuTiP (open systems, native solvers)
  - JAX (automatic differentiation, GPU)
  - NumPy (prototyping, dense matrices)
  - Custom (implement your own)

✅ **Simulate dissipation:**
  - Time-independent and time-dependent collapse operators
  - Lindblad master equations via QuTiP's `mesolve`

✅ **Debug transparently:**
  - Inspect the IR directly
  - Validate parameters before compilation
  - Clear error messages with recovery hints

---

## Feature Matrix: Backend Maturity

| Feature | QuTiP | JAX | NumPy | Notes |
|---------|-------|-----|-------|-------|
| Static Hamiltonians | ✅ Mature | ✅ Mature | ✅ Mature | All backends support |
| Time-dependent H | ✅ Mature | ✅ Mature | ✅ Mature | Envelopes via `\cos(\omega t)` etc. |
| Collapse ops (static) | ✅ Mature | ✅ Compile* | ⚠️ Limited | QuTiP optimized |
| Collapse ops (time-dep) | ✅ Mature | ✅ Compile* | ❌ No | QuTiP only |
| Open-system solvers | ✅ Mature | ❌ No | ❌ No | Use QuTiP for dissipation |
| Autodiff | ⚠️ Limited | ✅ Mature | ❌ No | JAX supports `grad`, `vmap` |
| Custom backends | ✅ Easy | ✅ Easy | ✅ Easy | Subclass `BackendBase` |
| Debugging | ✅ Excellent | ✅ Good | ✅ Good | All support IR inspection |

**Recommendation:**
- **Open-system simulation** → QuTiP
- **Optimization/autodiff** → JAX
- **Prototyping** → NumPy
- **Custom workflows** → Implement `BackendBase`

`*` JAX compiles static/time-dependent collapse operators; users supply a JAX-compatible solver for evolution.

---

## Installation

### Basic Setup

```bash
pip install latex-parser
```

This installs the core library with QuTiP backend support.

### Optional Backends

**JAX (for autodiff & GPU):**
```bash
pip install jax jaxlib
```

**NumPy (included by default):**
Already available.

**[See the Installation Guide for details](https://latex-2-physics.readthedocs.io/en/latest/install.html)**

---

## 5-Minute Quick Start

### 1. Static Qubit Hamiltonian

```python
from latex_parser.latex_api import compile_model

H = r"\frac{\omega_0}{2} \sigma_{z,1}"
model = compile_model(H_latex=H, params={"omega_0": 2.0}, qubits=1)

print(model.H)  # QuTiP Qobj
```

**Physics:** $H = \frac{\omega_0}{2} \sigma_z$ (qubit at rest frequency $\omega_0$)

### 2. Time-Dependent Drive (Rabi oscillation)

```python
H = r"\frac{\omega_0}{2} \sigma_{z,1} + \Omega \cos(\omega_d t) \sigma_{x,1}"
model = compile_model(
    H_latex=H,
    params={"omega_0": 1.0, "Omega": 0.5, "omega_d": 2.0},
    qubits=1,
    t_name="t"
)

from qutip import mesolve, basis
result = mesolve(model.H, basis(2, 0), times, args=model.args)
```

**Physics:** Rabi oscillations under resonant drive

### 3. Open System with Dissipation

```python
H = r"\omega_0 \sigma_{z,1}"
c_ops = [r"\sqrt{\gamma} \sigma_{-,1}"]  # Spontaneous decay

model = compile_model(
    H_latex=H,
    c_ops_latex=c_ops,
    params={"omega_0": 1.0, "gamma": 0.1},
    qubits=1
)

result = mesolve(model.H, psi0, times, c_ops=model.c_ops, args=model.args)
```

**Physics:** Exponential decay to ground state with rate $\gamma$

### 4. Cavity QED (Jaynes-Cummings)

```python
H = r"\omega_c a_1^\dagger a_1 + \frac{\omega_q}{2} \sigma_{z,1} + g (a_1 \sigma_{+,1} + a_1^\dagger \sigma_{-,1})"

model = compile_model(
    H_latex=H,
    params={"omega_c": 5.0, "omega_q": 2.0, "g": 0.1},
    qubits=1,
    bosons=[(10, "a")]  # Cavity mode with cutoff 10
)
```

**Physics:** Atom-cavity coupling (strong-coupling regime for $g$ ~ 0.1$\omega_c$)

---

## Common Use Cases & Examples

The repository includes **17 worked examples** (in `examples/` folder):

| Example | Physics | Features |
|---------|---------|----------|
| `example_static_qubit.py` | Two-level system | Basic Hamiltonian |
| `example_time_dependent_drive.py` | Rabi oscillations | Time-dependent envelope |
| `example_collapse_ops.py` | Damped Rabi | Dissipation, decay rates |
| `example_boson_number.py` | Harmonic oscillator | Fock space, cutoffs |
| `example_custom_subsystem.py` | Spin-1 system | Custom operators |
| `example_qutip_brme_dsl.py` | Non-Markovian | BRME solver |
| `example_jax_autodiff_workflow.py` | Pulse optimization | Gradient-based learning |
| `example_numpy_backend.py` | Dense matrices | NumPy arrays |
| + 9 more | Various | See documentation |

**Run any example:**
```bash
python examples/example_static_qubit.py
python examples/example_time_dependent_drive.py
```

**[Full example documentation with LaTeX equations and explanations](https://latex-parser.readthedocs.io/en/latest/examples.html)**

---

## Known Limitations

Before you start, be aware of these constraints:

### DSL Restrictions

1. **Operator functions can't contain operator sums inside**
   - ✅ OK: `\cos(\sigma_z)`, `\sin(n_1^2)`
   - ❌ Rejected: `\cos(\sigma_x + \sigma_y)` (operator sum inside function)
   - **Workaround:** Expand manually or use the IR directly

2. **Time-dependent collapse operators must be single monomials**
   - ✅ OK: `\sqrt{\gamma} \exp(-t) \sigma_-`
   - ❌ Rejected: `\sqrt{\gamma_1} \sigma_{-,1} + \sqrt{\gamma_2} \sigma_{-,2}` (sum)
   - **Workaround:** Use separate collapse operator strings

3. **Symbolic expansions capped at 512 terms**
   - Large powers like $(a + b)^{100}$ are blocked to prevent memory blow-up
   - **Workaround:** Expand manually

### Backend Limitations

- **QuTiP** — Dense-matrix only; slow for >15 qubits
- **JAX** — No built-in dissipation solvers (open systems)
- **NumPy** — Slowest backend; no optimization
- **Time-dependent collapse ops** — QuTiP only

### Parameter Handling

- Parameter names are matched with aliases (braces/spaces/asterisks stripped)
  - `\omega_c`, `\omega_{c}`, `omega_c`, `\omega c` all refer to the same parameter
  - Ambiguous aliases are warned but first match wins

### Quantum Numbers

- Boson cutoffs are **not automatically validated**
  - You can create under-truncated systems; check convergence by increasing cutoff
  - No automatic dimension warnings

### Compilation Speed

- Large systems (many qubits + complex time-dependence) can take seconds to compile
- Use static systems while developing; add time-dependence once validated

---

## Architecture

The library is organized into logical layers:

- **DSL layer** (`dsl.py`) — Physics-LaTeX canonicalization & operator extraction
- **IR layer** (`ir.py`) — LaTeX → Intermediate Representation pipeline
- **Backend layer** (`backend_*.py`) — IR → QuTiP/JAX/NumPy objects
- **API layer** (`latex_api.py`) — User-facing compilation functions

The **IR is transparent** — you can inspect it at any stage to debug or customize.

---

## Extending the Library

### Custom Backend

```python
from latex_parser.backend_base import BackendBase

class MyBackend(BackendBase):
    def compile_static_from_ir(self, ir, config, params):
        # Build operators from IR
        ...
        return my_result

from latex_parser.compile_core import register_backend
register_backend("mybackend", MyBackend)

# Use it
model = compile_model(H_latex=H, params=p, backend="mybackend")
```

See `examples/custom_backend.py` for a complete example.

### Custom LaTeX Macros

```python
from latex_parser.dsl import register_operator_macro

# Map \foo_{1} → \Jx_{1}
register_operator_macro("foo", "Jx")

H = r"g \foo_{1}"  # Now \foo is recognized as a qubit operator
```

### Custom Operator Functions

```python
from latex_parser.dsl import register_operator_function

register_operator_function("sinh")  # Add to allowed function list

H = r"\sinh(\sigma_z)"  # Now allowed
```

---

## Reference Materials

- **[Welcome & Overview](https://latex-parser.readthedocs.io/en/latest/welcome.html)** — Start here
- **[Installation Guide](https://latex-parser.readthedocs.io/en/latest/install.html)** — Setup instructions
- **[Usage Guide](https://latex-parser.readthedocs.io/en/latest/usage.html)** — Detailed tutorials
- **[DSL Reference](DSL_REFERENCE.md)** — Authoritative grammar spec
- **[API Documentation](https://latex-parser.readthedocs.io/en/latest/api.html)** — Complete API reference
- **[Examples](https://latex-parser.readthedocs.io/en/latest/examples.html)** — 17 worked examples with explanations

---

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `DSLValidationError: Missing numeric value for ...` | Parameter name mismatch | Check spelling: `\omega_c` → `omega_c`, with underscores |
| `DSLValidationError: Operator function not allowed ...` | Operator sum inside function | Rewrite: `\cos(\sigma_x + \sigma_y)` → expand manually |
| `DSLValidationError: Time-dependent collapse must be single monomial` | Sum in collapse operator | Split into separate strings: `[r"...\sigma_{-,1}", r"...\sigma_{-,2}"]` |
| `ImportError: No module named 'jax'` | JAX not installed | `pip install jax jaxlib` |
| Compilation is slow | Large system or complex time-dependence | Use static system while prototyping; switch backends |

**[See Usage Guide for more details](https://latex-parser.readthedocs.io/en/latest/usage.html)**

---

## Development

Clone and install for development:

```bash
git clone https://github.com/yourusername/latex_parser.git
cd latex_parser
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
pytest --cov=latex_parser tests/  # With coverage
```

Build documentation:

```bash
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

---

## License & Citation

See [LICENSE](LICENSE) file.

If you use this library in research, please cite:

```bibtex
@software{latex_parser,
  title={latex\_parser: Physics LaTeX to Quantum Simulation},
  author={...},
  year={2024},
  url={https://github.com/yourusername/latex_parser}
}
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/my-feature`)
3. Write tests and update documentation
4. Ensure `pytest` and `black` pass
5. Submit a pull request

---

**Ready to get started?** Go to the [Welcome page](https://latex-parser.readthedocs.io/en/latest/welcome.html) or jump to the [Quick Start](https://latex-parser.readthedocs.io/en/latest/install.html).


- **Physics-style LaTeX as input**

  Use familiar notation:

  - Qubits: \(\sigma_{x,1}, \sigma_{z,2}, \sigma_{\pm, j}\)
  - Bosons: \(a_{j}, a_{j}^{\dagger}, \hat{n}_{j}\)
  - Custom spins: \(J_{x,1}, J_{y,1}, J_{z,1}, J_{\pm,1}\)
  - Scalar parameters and time-dependent envelopes \(\cos(\omega t)\), \(\exp(-t)\), etc.

- **Structured IR instead of ad-hoc string hacking**

  LaTeX is parsed into a small intermediate representation:

  - Each term is “scalar expression × ordered operator product”.
  - Time dependence is tracked at the scalar (envelope) level.
  - Operator powers and products are explicit and validated.

- **Open-system support**

  - Static and time-dependent Hamiltonians.
  - Static and time-dependent collapse channels (with a controlled subset
    for time-dependent channels).
  - Output is directly usable as QuTiP `H` / `c_ops` objects.

- **f-deformed bosons and nonlinear optics**

  - Boson specs can carry a deformation \(f(n)\) for number operator eigenvalues.
  - DSL exposes \(\tilde{a}_{j}\) and \(\tilde{a}_{j}^{\dagger}\) that map to
    \(a f(n)\) and \(f(n) a^{\dagger}\).

- **Multiple backends**

  - QuTiP backend for open-system simulation.
  - NumPy / JAX backends for dense Hamiltonian matrices and custom workflows.

- **Escape hatches**

  - Custom finite-dimensional subsystems via `CustomSpec`.
  - IR-level APIs when the LaTeX DSL is too restrictive.
  - Full access to the underlying QuTiP objects for manual tweaks.

For the exact operator alphabet and grammar, see
[`DSL_REFERENCE.md`](DSL_REFERENCE.md), which is the canonical language spec.
For backend selection, you can also use `compile_model(..., backend="qutip"|"numpy"|"jax")`.

### Extending the DSL and backends
- Add your own operator macros without regex via `register_operator_macro("foo", "Jx")` (maps `\foo_{1}` → `\Jx_{1}`).
- Permit extra operator-valued functions with `register_operator_function("sinh")`.
- Walk/transform IR with `map_ir_terms(ir, fn)` if you need custom passes.
- Implement new backends by subclassing `BackendBase`; see `examples/custom_backend.py` for a minimal NumPy-based backend.
- Optional warnings can be toggled via `enable_warnings()` / `warn_once()`, and shared operator-function math lives in `operator_functions.apply_operator_function` for consistent results across backends.

### Extending the DSL/IR and backends
- Register custom LaTeX rewrites with `register_latex_pattern(pattern, replacement)` (e.g., map `\foo_{1}` to your custom operator name).
- Allow extra operator-valued functions with `register_operator_function("sinh")`.
- Build/transform IR with `map_ir_terms(ir, fn)` for advanced passes.
- Implement new backends by subclassing `BackendBase`; existing backends (QuTiP/JAX) consume only IR + HilbertConfig and can be used as references. Backend dispatch goes through `compile_model_core`, which consults a registry (`register_backend`) and validates required parameters early.

### Compilation pipeline (end-to-end)
1. Physics-style LaTeX → `canonicalize_physics_latex` (rewrites operators/macros).
2. SymPy parse → scalar/operator symbols.
3. IR build (`latex_to_ir`): each term = scalar SymPy expr × ordered operator product; time dependence detected from scalars/operator-function scalars.
4. Backend selection (`compile_model_core`): validates params against IR, then dispatches to QuTiP/NumPy/JAX (or a registered backend).
5. Backend compilation: IR → backend objects (`Qobj` H-list, dense arrays, etc.) using shared `BaseOperatorCache`.

### Parameter resolution rules (common to all backends)
- User params are matched via aliases: braces removed, asterisks/spaces stripped (e.g., `omega_{c}` ↔ `omega_c` ↔ `omega c`).
- Missing params raise `DSLValidationError` before backend dispatch (handled in `compile_model_core`).
- Operator symbols are never treated as params; they are filtered out when collecting required names.

### Debugging and common pitfalls
- If you see “Missing numeric value...” errors, check for mismatched parameter keys and remember aliases above.
- Time-dependent collapse ops must have exactly one operator term; scalar-only collapses are rejected.
- Large expansions like `(a_1 + a_2)^k` with big `k` are guarded by `MAX_EXPANDED_TERMS`; rewrite explicitly if needed.
- For JAX, ensure the package is installed and set `JAX_PLATFORM_NAME=cpu` if you want CPU-only runs; compiled JAX models now expose a `parameters` set so you can verify dependency tracking.
- NumPy backend is a dense-matrix wrapper; use it as a template when adding lightweight custom backends.

---

## Examples (8 validated snippets)

All imports use `latex_parser`; QuTiP examples need `pip install lat-dsl[qutip]`. Tensor ordering is always [all qubits, all bosons, all custom subsystems].

1) **Quick start (closed system)**

Hamiltonian: $H = A \cos(\omega t) \sigma_{x,1} + \cos(\sigma_{z,1})$
```python
from latex_parser import compile_model

H = r"A \cos(\omega t) \sigma_{x,1} + \cos(\sigma_{z,1})"
model = compile_model(H_latex=H, params={"A": 0.5, "omega": 1.0}, qubits=1, t_name="t")
```

2) **Open system with expectation (QuTiP)**

Hamiltonian: $H(t) = \tfrac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}$, collapse: $c = \sqrt{\gamma}\,\sigma_{-,1}$
```python
from qutip import mesolve, sigmaz, basis, expect
from latex_parser import compile_model

H_latex = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}"
c_ops = [r"\sqrt{\gamma} \sigma_{-,1}"]
params = {"omega_0": 2.0, "A": 0.3, "omega": 1.5, "gamma": 0.05}

model = compile_model(H_latex=H_latex, params=params, c_ops_latex=c_ops, qubits=1, t_name="t")
psi0 = basis(2, 0)
tlist = [0.0, 1.0, 2.0, 3.0]
result = mesolve(model.H, psi0, tlist, c_ops=model.c_ops, args=model.args)
expect_sz = [expect(sigmaz(), state) for state in result.states]
```

3) **Static Jaynes–Cummings**

Hamiltonian: $H = \omega_c n_{1} + \tfrac{\omega_q}{2} \sigma_{z,1} + g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})$
```python
from latex_parser import compile_model

H = r"\omega_c n_{1} + \frac{\omega_q}{2} \sigma_{z,1} + g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})"
model = compile_model(H_latex=H, params={"omega_c": 1.5, "omega_q": 2.0, "g": 0.1}, qubits=1, bosons=[3])
```

4) **Driven qubit (time-dependent envelope)**

Hamiltonian: $H(t) = \tfrac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}$
```python
from latex_parser import compile_model

H = r"\frac{\omega_0}{2} \sigma_{z,1} + A \cos(\omega t) \sigma_{x,1}"
model = compile_model(H_latex=H, params={"omega_0": 2.0, "A": 0.3, "omega": 1.5}, qubits=1, t_name="t")
```

5) **f-deformed boson ladder**

Hamiltonian: $H = g (\tilde{a}_{1} + \tilde{a}_{1}^{\dagger})$ with deformation $f(n) = \sqrt{n+1}$
```python
from latex_parser import compile_model, make_config

cfg = make_config(bosons=[(6, "a", r"\sqrt{n+1}")])
H = r"g (\tilde{a}_{1} + \tilde{a}_{1}^{\dagger})"
model = compile_model(H, params={"g": 0.1}, config=cfg)
```

6) **Custom subsystem (spin-1) + qubit + boson**

Hamiltonian: $H = \omega_c n_{1} + \tfrac{\omega_q}{2}\sigma_{z,1} + \omega_J J_{z,1} + g_{bq}(a_{1}\sigma_{+,1} + a_{1}^{\dagger}\sigma_{-,1}) + A \cos(\omega_d t) J_{x,1}$
```python
import numpy as np
from qutip import Qobj
from latex_parser import compile_model, make_config

sqrt2 = np.sqrt(2.0)
Jp = Qobj(np.array([[0, sqrt2, 0],[0,0,sqrt2],[0,0,0]], dtype=complex))
Jm = Jp.dag()
Jz = Qobj(np.diag([1.0, 0.0, -1.0]))
Jx = 0.5 * (Jp + Jm)

cfg = make_config(qubits=1, bosons=[3], customs=[("c", 1, 3, {"Jx": Jx, "Jz": Jz, "Jp": Jp, "Jm": Jm})])

H = r"""
    \omega_c n_{1}
    + \frac{\omega_q}{2} \sigma_{z,1}
    + \omega_J J_{z,1}
    + g_{bq} (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})
    + A \cos(\omega_d t) J_{x,1}
"""
model = compile_model(
    H_latex=H,
    params={"omega_c":1.0,"omega_q":0.7,"omega_J":0.3,"g_bq":0.05,"A":0.5,"omega_d":0.9},
    config=cfg,
    t_name="t",
)
```

7) **Two-mode beamsplitter + Gaussian pulse**

Hamiltonian: $H(t) = \omega_{c1} n_{1} + \omega_{c2} n_{2} + J (a_{1}^{\dagger} a_{2} + a_{2}^{\dagger} a_{1}) + A e^{-(t/\tau)^2} (a_{1} + a_{1}^{\dagger})$
```python
from latex_parser import compile_model

H = r"""
    \omega_{c1} n_{1} + \omega_{c2} n_{2}
    + J (a_{1}^{\dagger} a_{2} + a_{2}^{\dagger} a_{1})
    + A \exp(-(t/\tau)^2) (a_{1} + a_{1}^{\dagger})
"""
model = compile_model(
    H_latex=H,
    params={"omega_c1":1.0,"omega_c2":1.1,"J":0.02,"A":0.3,"tau":2.0},
    bosons=[3, 3],
    t_name="t",
)
```

8) **Collapse operators (static + time dependent)**

Hamiltonian: $H = \omega_c n_{1}$, collapses: $c_1 = \sqrt{\kappa}\, a_{1}$, $c_2(t) = \sqrt{\gamma}\, e^{-t/2} \sigma_{-,1}$
```python
from latex_parser import compile_model

H = r"\omega_c n_{1}"
c_ops = [r"\sqrt{\kappa} a_{1}", r"\sqrt{\gamma} \exp(-t/2) \sigma_{-,1}"]
model = compile_model(
    H_latex=H,
    c_ops_latex=c_ops,
    params={"omega_c":1.0, "kappa":0.1, "gamma":0.2},
    qubits=1,
    bosons=[4],
    t_name="t",
)
```

---

## Backends

The project is structured so that the DSL and IR are backend-agnostic.

Currently supported backends include:

* **QuTiP backend** (`backend_qutip.py`)

  * Full open-system support: static/time-dependent Hamiltonians and collapse channels.
  * Used by `compile_model` and `compile_open_system_from_latex`.
  * Produces `CompiledOpenSystemQutip` objects with:

    * `H`: static `Qobj` or QuTiP `H` list,
    * `c_ops`: list of static or time-dependent collapse operators,
    * `args`: parameter dictionary for envelopes,
    * `config`: `HilbertConfig`,
    * `time_dependent`: boolean.

* **NumPy backend** (`backend_numpy.py`)

  * Compiles IR into dense NumPy arrays for static Hamiltonians.
  * Suitable for custom eigensolvers or simple time-independent analysis.

* **JAX backend** (`backend_jax.py`)

  * Compiles IR into JAX arrays for differentiable simulation or custom JAX-based workflows.
  * Useful for optimization and machine-learning loops where JAX is already in use.

The interface is intentionally thin: all backends accept the same IR objects and `HilbertConfig`. You can also use `compile_model(..., backend="qutip"|"numpy"|"jax")` to pick a backend explicitly.

---

## Escape hatches and advanced usage

If the DSL cannot easily express what you want, there are several extension points.

1. **Custom subsystems via `CustomSpec`**

   * You can attach arbitrary operator matrices (from QuTiP, NumPy, scqubits, etc.)
     to a finite-dimensional subsystem.
   * The DSL will treat symbols like (A_{1}) or (J_{x,1}) as operators on that
     subsystem, depending on the `operators` mapping.

2. **IR-level construction**

   * You can bypass LaTeX entirely by building a `HamiltonianIR` manually and using
     backend functions such as `compile_time_dependent_hamiltonian_ir`.
   * This is useful if an external tool already generates an operator expansion.

3. **Manual backend modification**

   * `compile_model` returns a compiled model object.
   * You can treat `model.H`, `model.c_ops`, and `model.args` as starting points,
     modify them directly, and then feed the result into `mesolve` or your own
     solvers.

The intent is that the DSL is the “nice” front door, but you never get trapped in it.

---

## Troubleshooting (common errors)
- **“Operator function not allowed / mixed operator args”**: Only `exp`, `cos`, `sin` of a single operator (optional integer power) are supported. Rewrite sums/products outside the function.
- **“Time-dependent collapse must be single monomial”**: Split sums into separate collapse strings, e.g., `[ "...sigma_{-,1}", "...sigma_{-,2}" ]`.
- **“Missing numeric value for scalar symbol …”**: Check parameter names (`\omega_{c}` expects `omega_c`). Ambiguous keys log a warning and pick the first match.
- **Custom operator errors**: Ensure custom operators are `qutip.Qobj` with dims matching the declared `dim`.
- **Large expansion guard hit**: Rewrite large powers of sums explicitly; expansions beyond 512 terms are rejected.

## Developer notes

* The codebase is organized into layers:

  * `dsl.py`: physics-LaTeX canonicalization and operator extraction.
  * `ir.py`: LaTeX to IR pipeline and IR data structures.
  * `backend_qutip.py`, `backend_numpy.py`, `backend_jax.py`: backends.
  * `simple_api.py` / `latex_api.py`: public user-facing APIs.
  * `example.py`: end-to-end example (e.g. vacuum Rabi model).

* There are extensive self-tests in `dsl.py`, `ir.py`, `backend_qutip.py`,
  and `simple_api.py`. Running these directly is a good sanity check:

  ```bash
  python dsl.py
  python ir.py
  python backend_qutip.py
  python simple_api.py
  ```

* `DSL_REFERENCE.md` should be treated as the authoritative language spec;
  any change to the DSL must be reflected there.

## Development

- Use a Conda environment (e.g., `conda create -n latex python=3.13 black pytest pytest-cov jax jaxlib qutip sympy scipy numpy antlr4-python3-runtime`) and `conda activate latex` before running tests to keep the LaTeX/JAX stack consistent.
- Run `tox -e py313` for the full test suite with coverage, and `tox -e black -e flake8` for formatting/linting (Black is configured for 88-column width).

---

## Status and scope

This package is intentionally narrow:

* It focuses on **local operators** on tensor-product Hilbert spaces.
* It does **not** attempt symbolic manipulation of arbitrary operator expressions.
* It supports a controlled subset of operator-valued constructs suitable
  for typical quantum optics / circuit QED / spin-boson models.

If you need features outside this scope, the recommended path is to:

* Implement them as custom subsystems or IR builders, and
* Keep the LaTeX DSL restricted to the subset documented in `DSL_REFERENCE.md`.

Contributions that preserve this philosophy (clear DSL, explicit IR, narrow but
well-tested backends) are easier to integrate.
