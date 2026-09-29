from semantic_router.route import Route
from semantic_router.routers import HybridRouter, RouterConfig, SemanticRouter

__all__ = ["SemanticRouter", "HybridRouter", "Route", "RouterConfig"]

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("semantic-router")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0"
