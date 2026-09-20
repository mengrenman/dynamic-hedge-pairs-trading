"""Process-level data registry for the web app.

The notebooks keep their intermediates in ``notebooks/cache/*.parquet``; this module loads them
once per process and hands out the frames the routes need. Nothing here computes anything — see
``webapp.compute`` for that — and nothing here writes to the cache, so the app can never corrupt a
notebook's state.

The sizes matter for how this is written: ``day_market_bars.parquet`` is 803 MB on disk and about
2.6 s to read, so it is loaded lazily on first use and then kept. The Yahoo frames are 12 MB and
200 ms. A cold first request on a day-lake page is therefore slow and every one after it is not.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "notebooks" / "cache"

# Windows are baked into the cache filenames, so the app must use the same ones the notebooks did.
YAHOO_TRAIN = "2020-01-01_2025-12-31"
YAHOO_OOS = "2026-01-01_2026-06-25"


class MissingCache(RuntimeError):
    """A cache the notebooks should have written is not there."""

    def __init__(self, path: Path, notebook: str):
        super().__init__(
            f"{path.relative_to(ROOT)} is missing — run {notebook} first, or start the app with "
            f"--source yahoo to use only the network-free Yahoo caches."
        )
        self.path, self.notebook = path, notebook


def _read(path: Path, notebook: str, **kw) -> pd.DataFrame:
    if not path.exists():
        raise MissingCache(path, notebook)
    return pd.read_parquet(path, **kw)


@dataclass(frozen=True)
class Source:
    """A data source the UI can offer, and what it costs to load."""
    key: str
    label: str
    note: str
    needs: Tuple[str, ...]           # cache stems that must exist

    def available(self) -> bool:
        return all((CACHE / f"{n}.parquet").exists() for n in self.needs)


SOURCES: Tuple[Source, ...] = (
    Source("yahoo", "OpenBB / Yahoo daily",
           "505 tickers, 2020–2025 train + 2026 hold-out. 12 MB, loads in ~0.2 s.",
           (f"nb02_train_{YAHOO_TRAIN}", f"nb02_oos_BKNG_MA_{YAHOO_OOS}")),
    Source("day_lake", "Day lake (Polygon.io / Massive.com)",
           "Every listed name 2004–2025. 803 MB, ~2.6 s on first use, then cached.",
           ("day_market_bars", "day_screen")),
)


_MISS = object()


class Registry:
    """Lazily loaded, process-wide, read-only. Thread-safe because uvicorn serves concurrently."""

    def __init__(self) -> None:
        self._guard = threading.Lock()              # protects _locks and the touch counter only
        self._locks: Dict[str, threading.Lock] = {}
        self._cache: Dict[str, object] = {}
        # how many times the user has looked at a hold-out window, shown in the UI
        self.holdout_touches: int = 0

    def _memo(self, key: str, fn):
        """Compute-once cache with a lock **per key**.

        One global lock would deadlock: a memoised function may itself call _memo for a different
        key (fit_hedge asks for prices), and threading.Lock is not reentrant. Per-key locks also
        stop a 2.6 s day-lake load from blocking an unrelated 0.2 s Yahoo request, which a single
        lock would do.
        """
        hit = self._cache.get(key, _MISS)
        if hit is not _MISS:
            return hit
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            hit = self._cache.get(key, _MISS)       # another thread may have won the race
            if hit is not _MISS:
                return hit
            val = fn()
            self._cache[key] = val
            return val

    # ── prices ────────────────────────────────────────────────────────────────
    def yahoo_train(self) -> pd.DataFrame:
        def load():
            df = _read(CACHE / f"nb02_train_{YAHOO_TRAIN}.parquet", "notebook 02")
            return df["close"].unstack("ticker")
        return self._memo("yahoo_train", load)

    def yahoo_oos(self, t1: str, t2: str) -> Optional[pd.DataFrame]:
        """Only a handful of pairs were ever downloaded for the hold-out; None if this is not one."""
        p = CACHE / f"nb02_oos_{t1}_{t2}_{YAHOO_OOS}.parquet"
        if not p.exists():
            return None
        def load():
            return pd.read_parquet(p)["close"].unstack("ticker")
        return self._memo(f"yahoo_oos_{t1}_{t2}", load)

    def day_bars(self) -> pd.DataFrame:
        def load():
            df = _read(CACHE / "day_market_bars.parquet", "notebook 09", columns=["close"])
            return df["close"].unstack("ticker")
        return self._memo("day_bars", load)

    def day_dividends(self) -> pd.DataFrame:
        def load():
            df = _read(CACHE / "day_market_bars.parquet", "notebook 09", columns=["dividend"])
            px = self.day_bars()
            return df["dividend"].unstack("ticker").reindex_like(px).fillna(0.0)
        return self._memo("day_div", load)

    def prices(self, source: str) -> pd.DataFrame:
        return self.yahoo_train() if source == "yahoo" else self.day_bars()

    # ── screens ───────────────────────────────────────────────────────────────
    def day_screen(self) -> pd.DataFrame:
        return self._memo("day_screen", lambda: _read(CACHE / "day_screen.parquet", "notebook 10"))

    def day_rule(self, rule: str) -> pd.DataFrame:
        return self._memo(f"rule_{rule}",
                          lambda: _read(CACHE / f"day_rule_{rule}.parquet", "notebook 10"))

    def yahoo_screen(self) -> pd.DataFrame:
        return self._memo("yahoo_screen",
                          lambda: _read(CACHE / "viz_screen_spx_ndx_combined.parquet",
                                        "visualize_cointegrated_pairs (notebook 06)"))

    def formations(self) -> List[pd.Timestamp]:
        return self._memo("formations",
                          lambda: sorted(pd.to_datetime(self.day_screen()["formation"].unique())))

    # ── pair menus for the explorer ───────────────────────────────────────────
    def pairs_for(self, source: str, limit: int = 300) -> List[Tuple[str, str, str]]:
        """(ticker1, ticker2, label) — the pairs the UI offers, strongest evidence first."""
        def load():
            if source == "yahoo":
                sc = self.yahoo_screen()
                sc = sc[sc["verdict"].eq("pass")].nsmallest(limit, "eg_p")
                px = self.yahoo_train()
                out = []
                for (a, b), row in sc.iterrows():
                    if a in px.columns and b in px.columns:
                        out.append((a, b, f"{a} / {b}    p={row['eg_p']:.1e}"))
                return out
            r = self.day_rule("bh_dual").sort_values("eg_p_fdr").head(limit)
            seen, out = set(), []
            for a, b, p_ in zip(r["ticker1"], r["ticker2"], r["eg_p_fdr"]):
                if (a, b) not in seen:
                    seen.add((a, b)); out.append((a, b, f"{a} / {b}    p={p_:.1e}"))
            return out
        return self._memo(f"pairs_{source}_{limit}", load)

    # ── hold-out discipline ───────────────────────────────────────────────────
    def touch_holdout(self) -> int:
        with self._guard:
            self.holdout_touches += 1
            return self.holdout_touches


REGISTRY = Registry()


def available_sources() -> List[Source]:
    return [s for s in SOURCES if s.available()]
