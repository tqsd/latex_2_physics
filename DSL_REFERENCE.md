# DSL Reference (LaTeX-ready)

**Authoritative grammar and operator reference for the `latex_parser` quantum DSL.**

This document specifies **exactly** what LaTeX syntax is allowed and what will be rejected. Rules are organized by feature (qubits, bosons, time-dependence, etc.) with concrete examples from the repository's example files.

Copy/paste these patterns directly into your Hamiltonian or collapse-operator strings; standard math renderers will display them correctly.

---

## Core Principles

1. **Tensor order is fixed:** [all qubits] ⊗ [all bosons] ⊗ [all custom subsystems]
   - Enforced by `BaseOperatorCache` in all backends
   - Ensures consistent Kronecker products across backends

2. **Indices use commas:** `\sigma_{x,1}`, `a_{2}`, `J_{z,1}` (not `\sigma_x^{(1)}`)

3. **Strict operator alphabet** — only the symbols below are allowed

4. **Operator functions are whitelisted** — only `\exp`, `\cos`, `\sin` are allowed

5. **Time dependence lives in scalar envelopes** — not inside operator products
   - ✅ OK: `A(t) \sigma_x` where `A(t)` is a scalar envelope
   - ❌ Rejected: `\sigma_x(t) + \sigma_y(t)` (operator sums with time)

---

## Qubit Operators

**Pauli matrices and ladder operators for two-level systems.**

| Symbol | Name | Matrix | Notes |
|--------|------|--------|-------|
| `\sigma_{x,j}` | Pauli X | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip |
| `\sigma_{y,j}` | Pauli Y | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ | Rotation about y |
| `\sigma_{z,j}` | Pauli Z | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Population inversion |
| `\sigma_{+,j}` | Raising | $\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ | Excitation |
| `\sigma_{-,j}` | Lowering | $\begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$ | De-excitation |

**Example usage:**
```latex
H = \omega_0 \sigma_{z,1}                                    % Static qubit
H = A \cos(\omega t) \sigma_{x,1}                            % Driven qubit
c_ops = [r"\sqrt{\gamma} \sigma_{-,1}"]                      % Decay channel
```

---

## Bosonic Operators

**Ladder operators for quantized fields (cavity modes, vibrations).**

| Symbol | Name | Interpretation | Notes |
|--------|------|-----------------|-------|
| `a_{j}` | Lowering (annihilation) | Removes one quantum | Action: $\|n\rangle \to \sqrt{n}\|n-1\rangle$ |
| `a_{j}^{\dagger}` | Raising (creation) | Adds one quantum | Action: $\|n\rangle \to \sqrt{n+1}\|n+1\rangle$ |
| `\hat{n}_{j}` | Number operator | Photon/qubit count | Equal to $a^{\dagger} a$; eigenvalues 0, 1, 2, ... |
| `n_{j}` | Number operator | (shorthand for $\hat{n}$) | Equivalent to `\hat{n}_{j}` |
| `\tilde{a}_{j}` | f-deformed lowering | Deformed annihilation | Maps to $a f(n)$ where $f$ is a user-defined function |
| `\tilde{a}_{j}^{\dagger}` | f-deformed raising | Deformed creation | Maps to $f(n) a^{\dagger}$ |

**Cutoff specification:**
```python
# Declare a boson with cutoff 10 (Hilbert space dimension 10)
model = compile_model(..., bosons=[(10, "a")])

# Multiple bosons:
model = compile_model(..., bosons=[(5, "a"), (8, "b")])
```

**Example usage:**
```latex
H = \omega_c a_{1}^{\dagger} a_{1}                           % Energy of cavity mode
H = g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})   % Jaynes-Cummings coupling
c_ops = [r"\sqrt{\kappa} a_{1}"]                             % Photon loss
```

**Deformed bosons:**
```python
# Create boson with f(n) = sqrt(n+1)
cfg = make_config(bosons=[(10, "a", r"\sqrt{n+1}")])

H = r"g (\tilde{a}_{1} + \tilde{a}_{1}^{\dagger})"  # Uses deformed ladder operators
```

---

## Custom Subsystems

**User-defined operators on arbitrary finite-dimensional systems.**

Any operator can be attached to a custom subsystem:

```python
from qutip import Qobj
import numpy as np

# Define custom operators (e.g., spin-1 system)
sqrt2 = np.sqrt(2)
Jp = Qobj(np.array([[0, sqrt2, 0], [0, 0, sqrt2], [0, 0, 0]]))
Jm = Jp.dag()
Jz = Qobj(np.diag([1, 0, -1]))

# Register them in a custom subsystem
cfg = make_config(customs=[("J", 1, 3, {"Jp": Jp, "Jm": Jm, "Jz": Jz})])

# Use in LaTeX
H = r"\omega_0 J_{z,1} + g (J_{+,1} \sigma_{-,1} + J_{-,1} \sigma_{+,1})"
```

**Key points:**
- `"J"` is the operator prefix (use in LaTeX as `J_{x,1}`, etc.)
- `1` is the subsystem index
- `3` is the Hilbert space dimension
- `{"Jp": Qobj(...), ...}` maps operator names to `Qobj` instances

---

## Operator-Valued Functions

**Allowed functions: `\exp`, `\cos`, `\sin` applied to operators.**

### Rules

1. Function must contain **exactly one operator** (possibly raised to an integer power)
2. Scalar factors are allowed: `A \cos(\omega t) \sigma_{z,1}` ✅
3. No operator sums inside functions: `\cos(\sigma_x + \sigma_y)` ❌
4. No products of operators inside functions: `\exp(\sigma_x \sigma_z)` ❌

### Allowed patterns

```latex
\cos(\sigma_{z,1})                      % Single operator
\sin(n_{1})                             % Number operator
\exp(\sigma_{z,1}^2)                    % Operator to integer power
\exp(-\gamma t) \sigma_{-,1}            % Scalar envelope × operator
A \cos(\omega t) \sigma_{x,1}           % General envelope: scalar × function × operator
```

### Rejected patterns

```latex
\cos(\sigma_{x,1} + \sigma_{y,1})       % ❌ Operator sum inside function
\sin(\sigma_{x,1} \sigma_{z,1})         % ❌ Operator product inside function
\cos(a_{1}^2 + b_{1}^2)                 % ❌ Sum of powers inside function
\exp(\omega \sigma_z + \gamma n)         % ❌ Sum inside function
```

### Examples from the repo

From `example_ir_debugging.py`:
```python
H = r"\exp(\sigma_{z,1})"                        # Allowed
```

From `example_jax_autodiff_workflow.py`:
```python
H = r"A \cos(\omega t) \sigma_{x,1}"             # Allowed
```

---

## Scalars, Parameters, and Time-Dependence

### Parameter Naming & Aliases

Parameter names are **normalized** — braces, spaces, and asterisks are ignored:

| Your LaTeX | Valid parameter names (all equivalent) |
|-----------|------------------------------------------|
| `\omega_{c}` | `omega_c`, `omega c`, `\omega_c`, `\omega_{c}` |
| `g_{JC}` | `g_JC`, `g JC` |
| `\Omega_R` | `Omega_R`, `Omega R` |

**Important:** If your params dict has both `omega_c` and `omega_c1`, the first match wins (ambiguous!).

### Time Symbols

The **time variable** is specified with the `t_name` parameter (default: `"t"`):

```python
# Time variable is t (default)
model = compile_model(H_latex=H, ..., t_name="t")

# Time variable is tau instead
model = compile_model(H_latex=H, ..., t_name="tau")
```

**Additional time-like symbols** can be declared with `time_symbols`:

```python
# Treat both t and s as time variables
ir = latex_to_ir(H_latex, config, t_name="t", time_symbols=("t", "s"))
```

### Time-Dependence Detection

Any free symbol matching `t_name` or in `time_symbols` marks a **scalar as time-dependent:**

```latex
A \cos(\omega t) \sigma_{x,1}            % Time-dependent (uses t)
\sin(t/\tau) \sigma_{z,1}                % Time-dependent (uses t/tau)
\exp(-\gamma t) a_{1}                    % Time-dependent (exponential decay)
\omega_0 \sigma_z                        % Static (no t)
```

---

## Collapse Operators

**Lindblad jump operators for dissipation and decoherence.**

### Syntax

Each collapse operator is a separate LaTeX string in the `c_ops_latex` list:

```python
c_ops_latex = [
    r"\sqrt{\kappa} a_{1}",                           # Photon loss
    r"\sqrt{\gamma} \sigma_{-,1}",                    # Spontaneous decay
    r"\sqrt{\gamma_\phi} \sigma_{z,1}",               # Dephasing
]
```

### Static vs. Time-Dependent

**Static collapse operators** (independent of time):
```latex
\sqrt{\kappa} a_{1}                      # Photon loss
\sqrt{\gamma} \sigma_{-,1}               # Spontaneous decay
```

**Time-dependent collapse operators** (QuTiP only):
```latex
\sqrt{\gamma(t)} \sigma_{-,1}            % Decay with time-modulated rate
\sqrt{\gamma} \exp(-t/\tau) \sigma_{-,1} % Decay with turn-off envelope
```

### Restrictions

1. **Static operators can be sums:**
   ```latex
   \sqrt{\kappa} a_{1} + \sqrt{\gamma} \sigma_{-,1}   % ✅ Allowed (sum of monomials)
   ```

2. **Time-dependent operators must be single monomials:**
   ```latex
   \sqrt{\gamma(t)} \sigma_{-,1}                      % ✅ Allowed (single monomial with time-dep scalar)
   \sqrt{\gamma_1(t)} \sigma_{-,1} + \sqrt{\gamma_2(t)} \sigma_{-,2}  % ❌ Rejected (sum with time-dep)
   ```

   **Workaround:** Use separate collapse operator strings:
   ```python
   c_ops_latex = [
       r"\sqrt{\gamma_1(t)} \sigma_{-,1}",
       r"\sqrt{\gamma_2(t)} \sigma_{-,2}",
   ]
   ```

---

## f-Deformed Bosons

**Non-linear bosonic systems where the ladder operators are modified by a function.**

### Declaration

```python
# Cutoff=10, label="a", deformation function f(n) = sqrt(n+1)
cfg = make_config(bosons=[(10, "a", r"\sqrt{n+1}")])
```

### Deformation Inputs

The deformation can be specified as:

1. **LaTeX string** (parsed to callable):
   ```python
   deformation = r"\sqrt{n+1}"           # f(n) = sqrt(n+1)
   deformation = r"n+1"                  # f(n) = n+1
   ```

2. **Python callable**:
   ```python
   import numpy as np
   deformation = lambda n: np.sqrt(n+1)  # f(n) = sqrt(n+1)
   ```

3. **SymPy expression**:
   ```python
   from sympy import sqrt, symbols
   n = symbols('n')
   deformation = sqrt(n+1)               # f(n) = sqrt(n+1)
   ```

### Usage in LaTeX

In LaTeX, use the **tilde notation** for deformed operators (regardless of the actual function):

```latex
H = g (\tilde{a}_{1} + \tilde{a}_{1}^{\dagger})   % Deformed ladder operators
```

Internally, this maps to:
- $\tilde{a} \to a \cdot f(n)$  (deformed annihilation)
- $\tilde{a}^{\dagger} \to f(n) \cdot a^{\dagger}$  (deformed creation)

---

## Symbolic Expansion Guard

**Large symbolic expansions are blocked to prevent memory blow-up.**

```python
MAX_EXPANDED_TERMS = 512  # Hard limit
```

If your Hamiltonian would expand to more than 512 symbolic terms, it's rejected:

```latex
(a_{1} + a_{2} + ... + a_{10})^{10}    % ❌ Rejected (too many terms)
(a_{1} + b_{1})^8                      % Depends on context; may hit limit
```

**Workaround:** Rewrite explicitly without large powers:
```python
# Instead of (a+b)^3, write:
H = r"a_{1} a_{1} a_{1} + a_{1} a_{1} b_{1} + ... "  # Expand manually
```

---

## Quick Reference: Allowed vs. Rejected

### Hamiltonian Terms

| Pattern | Status | Notes |
|---------|--------|-------|
| `\omega_0 \sigma_{z,1}` | ✅ OK | Static qubit |
| `A \cos(\omega t) \sigma_{x,1}` | ✅ OK | Time-dependent drive |
| `\omega_c a_{1}^{\dagger} a_{1}` | ✅ OK | Cavity energy |
| `g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})` | ✅ OK | Jaynes-Cummings |
| `\exp(\sigma_{z,1})` | ✅ OK | Exponential of operator |
| `\cos(\sigma_{x,1} + \sigma_{y,1})` | ❌ Rejected | Operator sum inside function |
| `\sin(\sigma_{x,1} \sigma_{z,1})` | ❌ Rejected | Operator product inside function |
| `\omega t \sigma_{x,1}` | ❌ Rejected | Time inside scalar × operator product (not allowed; use envelope) |

### Collapse Operators

| Pattern | Status | Notes |
|---------|--------|-------|
| `\sqrt{\kappa} a_{1}` | ✅ OK | Photon loss |
| `\sqrt{\gamma} \sigma_{-,1}` | ✅ OK | Spontaneous decay |
| `\sqrt{\gamma} \exp(-t) \sigma_{-,1}` | ✅ OK | Time-dependent decay |
| `\sqrt{\gamma_1} \sigma_{-,1} + \sqrt{\gamma_2} \sigma_{-,2}` | ✅ OK | Sum (static only) |
| `\sqrt{\gamma(t)} \sigma_{-,1} + \sqrt{\gamma(t)} \sigma_{-,2}` | ❌ Rejected | Sum with time-dependent terms (split into separate strings) |

---

## Backend Assumptions

All backends enforce these rules:

1. **Tensor product ordering:** Qubits, then bosons, then customs (enforced by `BaseOperatorCache`)
2. **Operator lookup key:** `(kind, label, index, op_name, power)` — must match exactly
3. **Custom operators:** Must be `Qobj` with valid `.dims` attribute
4. **Time detection:** Free symbol matching `t_name` or `time_symbols` in scalar expression
5. **Operator function scalars:** Mixed time/static factors are rejected in static compile path

---

## Construction Recipes

### Static Two-Qubit Ising Model
```python
H = r"J \sigma_{z,1} \sigma_{z,2}"
model = compile_model(H_latex=H, params={"J": 0.5}, qubits=2)
```

### Jaynes-Cummings with Detuning
```python
H = r"\Delta \sigma_{+,1} \sigma_{-,1} + \omega_c a_{1}^{\dagger} a_{1} + g (a_{1} \sigma_{+,1} + a_{1}^{\dagger} \sigma_{-,1})"
model = compile_model(
    H_latex=H,
    params={"Delta": 0.2, "omega_c": 5.0, "g": 0.1},
    qubits=1,
    bosons=[(10, "a")]
)
```

### Damped Two-Photon Cavity with Loss
```python
H = r"\omega_c n_{1} + \kappa (a_{1}^{2} + a_{1}^{\dagger,2})"
c_ops = [r"\sqrt{\gamma} a_{1}"]
model = compile_model(
    H_latex=H,
    c_ops_latex=c_ops,
    params={"omega_c": 2.0, "kappa": 0.05, "gamma": 0.1},
    bosons=[(10, "a")]
)
```

### Multi-Mode Boson System
```python
H = r"\omega_1 n_{1} + \omega_2 n_{2} + J (a_{1}^{\dagger} a_{2} + a_{2}^{\dagger} a_{1})"
model = compile_model(
    H_latex=H,
    params={"omega_1": 1.0, "omega_2": 1.1, "J": 0.05},
    bosons=[(5, "a"), (5, "b")]
)
```

---

## Common Pitfalls

| Issue | Cause | Solution |
|-------|-------|----------|
| `DSLValidationError: Missing numeric value for ...` | Parameter name doesn't match | Check aliases: `\omega_c` → `omega_c` (braces/spaces ignored) |
| `DSLValidationError: Operator function not allowed ...` | Operator sum inside function | Rewrite: expand `\cos(\sigma_x + \sigma_y)` manually |
| `DSLValidationError: Time-dependent collapse must be single monomial` | Sum in collapse with time-dep terms | Split into separate strings: `[r"...\sigma_{-,1}", r"...\sigma_{-,2}"]` |
| `DSLValidationError: Expansion limit (512 terms) exceeded` | Large powers in symbolic expansion | Expand manually without large powers |
| `ValueError: Custom operator missing .dims` | Custom operator not a valid `Qobj` | Ensure all custom operators are `qutip.Qobj` with proper `.dims` |
| Compilation is slow | Large system or many time-dependent terms | Use static system while prototyping; add time-dependence once validated |

---

## Version & Changelog

**Current version:** 1.0 (stable)

**Key features:**
- Full qubit, boson, and custom subsystem support
- Operator functions: `\exp`, `\cos`, `\sin`
- Time-dependent Hamiltonians and collapse operators
- f-deformed bosonic ladder operators
- Three backends: QuTiP, JAX, NumPy
- Transparent intermediate representation (IR)

**Known limitations:** See main documentation for details.


