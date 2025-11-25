from __future__ import annotations

import warnings

__all__ = [
    "LatexParserError",
    "BackendUnavailableError",
    "enable_warnings",
    "warn_once",
]


class LatexParserError(Exception):
    r"""Base exception for latex_parser."""


class BackendUnavailableError(LatexParserError):
    r"""Raised when an optional backend is unavailable."""


_WARN_ENABLED = True
_WARNED: set[str] = set()


def enable_warnings(enabled: bool = True) -> None:
    r"""Enable or disable module-level runtime warnings."""
    global _WARN_ENABLED
    _WARN_ENABLED = enabled


def warn_once(message: str) -> None:
    r"""Emit a warning only once per unique message."""
    if not _WARN_ENABLED:
        return
    if message in _WARNED:
        return
    _WARNED.add(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
