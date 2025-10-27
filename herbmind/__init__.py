"""Core HerbMind package."""

from importlib import import_module
from typing import Any

__all__ = ["config", "data", "models", "pipeline"]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import wrapper
    if name in __all__:
        module = import_module(f"herbmind.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'herbmind' has no attribute {name!r}")
