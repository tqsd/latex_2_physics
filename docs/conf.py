"""
Sphinx configuration for the latex_parser project.

This minimal config enables autodoc and sets up the import path so the package
can be documented without installation.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

project = "latex_parser"
author = "latex_parser developers"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__call__",
    "inherited-members": False,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

try:
    import sphinx_book_theme  # type: ignore  # noqa: F401

    html_theme = "sphinx_book_theme"
except Exception:
    try:
        import furo  # type: ignore  # noqa: F401

        html_theme = "furo"
    except Exception:
        try:
            import sphinx_rtd_theme  # type: ignore  # noqa: F401

            html_theme = "sphinx_rtd_theme"
            html_theme_path = [
                __import__("sphinx_rtd_theme").themes.get_html_theme_path()
            ]
        except Exception:
            html_theme = "alabaster"

html_static_path = ["_static"]

# Keep theme options minimal to avoid unsupported keys across themes.
html_theme_options = {}

# Suppress duplicate object warnings from autodoc
# These occur when the same object is documented multiple times (resolved by removing
# duplicates)
suppress_warnings = [
    "autodoc.duplicate_object",
    "autosummary.duplicate_object",
]
