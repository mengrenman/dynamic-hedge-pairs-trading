# pairs/validation/__init__.py
"""
Validation utilities for pairs trading strategies.

Provides walk-forward (rolling-window) back-testing helpers that prevent
look-ahead bias and give honest out-of-sample performance estimates.
"""

from .walk_forward import walk_forward_splits, walk_forward_backtest

__all__ = [
    "walk_forward_splits",
    "walk_forward_backtest",
]
