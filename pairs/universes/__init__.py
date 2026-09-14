from __future__ import annotations
from importlib.resources import files
from pathlib import Path
from typing import Iterable

from ..utils.tickers import load_tickers as _load_tickers

__all__ = ["load_universe", "list_universes", "__UNIVERSES_VERSION__"]
__UNIVERSES_VERSION__ = "2025.08"

def _tickers_dir():
    return files(__package__) / "tickers"

def list_universes(extensions: Iterable[str] = (".txt", ".csv", ".json")) -> list[str]:
    """Names available under pairs/universes/tickers (without extension)."""
    base = _tickers_dir()
    names: set[str] = set()
    for entry in base.iterdir():
        if entry.is_file() and entry.suffix.lower() in extensions:
            names.add(entry.name[: -len(entry.suffix)])
    return sorted(names)

def load_universe(name: str):
    """
    Load a packaged ticker universe by base name (e.g., 'spx', 'ndx').
    Searches pairs/universes/tickers/<name>.(txt|csv|json).
    Returns a pandas.Index of clean, UPPERCASE tickers.
    """
    base = _tickers_dir()
    for ext in (".txt", ".csv", ".json"):
        p = base / f"{name}{ext}"
        if p.is_file():
            return _load_tickers(Path(p))
    raise FileNotFoundError(
        f"No packaged universe named {name!r}. "
        f"Available: {', '.join(list_universes()) or '(none)'}"
    )
