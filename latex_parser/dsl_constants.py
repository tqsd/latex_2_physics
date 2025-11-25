# Centralized DSL constants and shared error fragments.

ALLOWED_OPERATOR_FUNCTIONS = {"exp", "cos", "sin", "cosh", "sinh", "sqrtm"}


def register_operator_function(name: str) -> None:
    r"""
    Register an additional operator-valued function (e.g., "sinh").
    """
    if not name or not isinstance(name, str):
        raise ValueError("Operator function name must be a non-empty string.")
    ALLOWED_OPERATOR_FUNCTIONS.add(name)


ERROR_HINT_OPERATOR_FUNC = (
    "Operator-valued functions must be one of "
    f"{sorted(ALLOWED_OPERATOR_FUNCTIONS)} with a single operator argument "
    "(optional positive integer power) and no sums or extra scalars."
)

ERROR_HINT_TIME_DEP_COLLAPSE = (
    "Time-dependent collapse operators must be a single monomial. "
    "Split sums into separate collapse strings, e.g., "
    r"['\\sqrt{\\gamma_1} e^{-t/2} \\sigma_{-,1}', "
    r"'\\sqrt{\\gamma_2} e^{-t/2} \\sigma_{-,2}']."
)
