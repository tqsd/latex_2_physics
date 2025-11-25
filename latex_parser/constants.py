"""
Shared symbolic constants and built-in names used across backends.
"""

# TODO (kareem): Expand this with more physics constants as needed.
# Currently, these are just placeholders with default numeric values.
# For example, one might want to define "hbar", "c", "epsilon_0", etc.
# These can be mapped to more accurate values or symbolic representations
# in specific backends or configurations.
# The key idea is that these constants are recognized by the parser
# and do not require user-supplied parameters.
# They can be overridden or extended in backend implementations as needed.
# The current set is minimal for demonstration purposes.
# Next steps to make `dsl.py` `ir.py` recognize these constants directly.
# This would allow users to write LaTeX expressions using these names
# without needing to define them in the `params` dictionary.

from __future__ import annotations

import sympy as sp

# Mapping of recognized built-in names to SymPy objects. These are treated as
# intrinsic constants and never require user-supplied parameters.
BUILTIN_SYMBOL_MAP: dict[str, sp.Expr] = {
    "I": sp.I,
    "E": sp.E,
    "pi": sp.pi,
    "Pi": sp.pi,
    "oo": sp.oo,
    "zoo": sp.zoo,
    "nan": sp.nan,
    "NaN": sp.nan,
    # Common physics constants (placeholder numeric defaults).
    "epsilon": sp.Float(1.0),
    "mu": sp.Float(1.0),
}

BUILTIN_NAMES = set(BUILTIN_SYMBOL_MAP.keys())
BUILTIN_SYMS = set(BUILTIN_SYMBOL_MAP.values())
