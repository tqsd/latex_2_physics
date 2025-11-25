from __future__ import annotations

from typing import Any

import numpy as _np
from scipy.linalg import expm as _np_expm, sqrtm as _np_sqrtm  # type: ignore

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as _jnp  # type: ignore
    from jax.scipy.linalg import expm as _jax_expm  # type: ignore
    from jax.scipy.linalg import sqrtm as _jax_sqrtm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    jax = None
    _jnp = None
    _jax_expm = None
    _jax_sqrtm = None

__all__ = ["apply_operator_function"]


def _apply_numpy(mat: _np.ndarray, func_name: str) -> _np.ndarray:
    r"""Apply an operator function using SciPy/NumPy implementations."""
    if func_name == "exp":
        return _np_expm(mat)
    if func_name == "cos":
        i = 1.0j
        return 0.5 * (_np_expm(i * mat) + _np_expm(-i * mat))
    if func_name == "sin":
        i = 1.0j
        return (_np_expm(i * mat) - _np_expm(-i * mat)) / (2.0j)
    if func_name == "cosh":
        return 0.5 * (_np_expm(mat) + _np_expm(-mat))
    if func_name == "sinh":
        return 0.5 * (_np_expm(mat) - _np_expm(-mat))
    if func_name == "sqrtm":
        return _np_sqrtm(mat)
    raise ValueError(f"Unsupported operator function '{func_name}'.")


def _apply_jax(mat: Any, func_name: str) -> Any:
    r"""Apply an operator function using JAX implementations."""
    if _jnp is None or _jax_expm is None:
        raise RuntimeError("jax is not available.")
    if func_name == "exp":
        return _jax_expm(mat)
    if func_name == "cos":
        i = 1.0j
        return 0.5 * (_jax_expm(i * mat) + _jax_expm(-i * mat))
    if func_name == "sin":
        i = 1.0j
        return (_jax_expm(i * mat) - _jax_expm(-i * mat)) / (2.0j)
    if func_name == "cosh":
        return 0.5 * (_jax_expm(mat) + _jax_expm(-mat))
    if func_name == "sinh":
        return 0.5 * (_jax_expm(mat) - _jax_expm(-mat))
    if func_name == "sqrtm":
        if _jax_sqrtm is None:
            raise RuntimeError("jax scipy sqrtm unavailable.")
        return _jax_sqrtm(mat)
    raise ValueError(f"Unsupported operator function '{func_name}'.")


def apply_operator_function(mat: Any, func_name: str, backend: str) -> Any:
    r"""
    Apply an operator-valued function using a chosen backend.

    backend: "numpy" | "jax"
    """
    if backend == "numpy":
        return _apply_numpy(_np.asarray(mat, dtype=complex), func_name)
    if backend == "jax":
        return _apply_jax(mat, func_name)
    raise ValueError(f"Unsupported backend '{backend}'.")
