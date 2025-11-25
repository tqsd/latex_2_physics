import warnings

import pytest

from latex_parser.errors import (
    BackendUnavailableError,
    LatexParserError,
    enable_warnings,
    warn_once,
)


def test_warn_once_toggle():
    enable_warnings(True)
    with warnings.catch_warnings(record=True) as w:
        warn_once("msg")
        warn_once("msg")
        assert len(w) == 1
    enable_warnings(False)
    with warnings.catch_warnings(record=True) as w:
        warn_once("msg2")
        assert len(w) == 0
    enable_warnings(True)


def test_custom_exceptions():
    with pytest.raises(LatexParserError):
        raise LatexParserError("fail")
    with pytest.raises(BackendUnavailableError):
        raise BackendUnavailableError("backend missing")
