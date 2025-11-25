r"""
latex_parser package: LaTeX DSL to IR and backends.

Primary user-facing symbols are re-exported from latex_api for convenience.
"""

from .latex_api import (  # noqa: F401
    BosonSpec,
    CompiledOpenSystemQutip,
    CustomSpec,
    HilbertConfig,
    QubitSpec,
    compile_model,
    make_config,
)
from .backend_base import BackendBase, BackendOptions  # noqa: F401
from .operator_functions import apply_operator_function  # noqa: F401
from .errors import (  # noqa: F401
    LatexParserError,
    BackendUnavailableError,
    enable_warnings,
    warn_once,
)
from .dsl import deformation_callable_from_latex  # noqa: F401

__all__ = [
    "BosonSpec",
    "CompiledOpenSystemQutip",
    "CustomSpec",
    "BackendBase",
    "BackendOptions",
    "HilbertConfig",
    "QubitSpec",
    "compile_model",
    "make_config",
    "apply_operator_function",
    "LatexParserError",
    "BackendUnavailableError",
    "enable_warnings",
    "warn_once",
    "deformation_callable_from_latex",
]
