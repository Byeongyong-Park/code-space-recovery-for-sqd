"""Code-space recovery research package.

Public functions and result types live in their defining submodules. The
package root intentionally remains lightweight so importing version metadata
does not import optional numerical or hardware dependencies.
"""

from ._version import ALGORITHM_VERSION, PACKAGE_VERSION, __version__


__all__ = [
    "PACKAGE_VERSION",
    "ALGORITHM_VERSION",
    "__version__",
]
