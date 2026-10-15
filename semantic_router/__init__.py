from importlib.metadata import PackageNotFoundError, version

from semantic_router.config import load_router
from semantic_router.route import Route
from semantic_router.routers import Chain, DecisionRouter, IndexedRouter, Merge

__all__ = ["Route", "IndexedRouter", "DecisionRouter", "Chain", "Merge", "load_router"]

try:
    __version__ = version("semantic-router")
except PackageNotFoundError:  # running from a checkout that was never installed
    __version__ = "0.0.0"
