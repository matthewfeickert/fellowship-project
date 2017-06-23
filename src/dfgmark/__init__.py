from __future__ import absolute_import, division, print_function

import sys
import warnings

from dfgmark.__about__ import (
    __author__, __copyright__, __email__, __license__, __summary__, __title__,
    __uri__, __version__
)


__all__ = [
    "__title__", "__summary__", "__uri__", "__version__", "__author__",
    "__email__", "__license__", "__copyright__",
]

if sys.version_info[:2] == (2, 7):
    warnings.warn(
        "Python 2.7 is no longer supported by the Python core team, please "
        "upgrade your Python. DFGMark was written for Python 3.",
        DeprecationWarning
    )
