from matgraph._extras import install as _install_extra_hints

_install_extra_hints()

from matgraph.sdk import MatGraphSDK  # noqa: E402  (import after hook install is intentional)

__all__ = ["MatGraphSDK"]
