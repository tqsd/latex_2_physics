import numpy as np
from scipy.linalg import expm

from latex_parser.operator_functions import apply_operator_function


def test_operator_functions_numpy_shapes():
    mat = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    for func in ("exp", "cos", "sin", "cosh", "sinh", "sqrtm"):
        out = apply_operator_function(mat, func, backend="numpy")
        assert out.shape == mat.shape


def test_operator_functions_numpy_values():
    mat = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    exp_val = apply_operator_function(mat, "exp", backend="numpy")
    assert np.allclose(exp_val, expm(mat))
