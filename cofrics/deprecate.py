"""
Utilities for marking deprecated functions.
"""

import warnings
from functools import wraps


def deprecated(msg: str = None, stack_level: int = 2) -> callable:
    """Used to mark a function as deprecated.

    Parameters
    ----------
    msg: str
        The message to display in the deprecation warning.

    stack_level: int
        How far up the stack the warning needs to go, before
        showing the relevant calling lines.

    Returns
    -------
    deprecated_dec: callable
        A decorator function to invoke DeprecationWarning before calling
        decorated function.

    Usage
    -----
    @deprecated(msg='function_a is deprecated! Use function_b instead.')
    def function_a(*args, **kwargs):

    """

    def deprecated_dec(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            warnings.warn(
                msg or "Function %s is deprecated." % fn.__name__,
                category=DeprecationWarning,
                stacklevel=stack_level,
            )
            return fn(*args, **kwargs)

        return wrapper

    return deprecated_dec
