"""Tier-3 background jobs.

The portfolio backtest runs 39 semi-annual formations over twenty years and takes minutes, which is
far too long for a request. It runs on a worker thread; the page polls a fragment endpoint with
``hx-trigger="every 2s"`` until the job reports done.

A dict and a thread pool are the right size for this. There is one user, on one machine, and a
queue broker would be more moving parts than the thing it coordinates. What matters is that jobs
are cancellable and that a crash surfaces as a message rather than a spinner that never stops.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")
_LOCK = threading.Lock()
_JOBS: "Dict[str, Job]" = {}
_MAX_KEPT = 20


@dataclass
class Job:
    id: str
    label: str
    status: str = "queued"               # queued | running | done | failed | canceled
    progress: float = 0.0                # 0..1
    message: str = ""
    result: Any = None
    error: str = ""
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    @property
    def done(self) -> bool:
        return self.status in ("done", "failed", "canceled")

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def canceled(self) -> bool:
        return self._cancel.is_set()


class Canceled(RuntimeError):
    """Raised inside a worker when the user asks it to stop."""


def submit(label: str, fn: Callable[["Job"], Any]) -> Job:
    """Run ``fn(job)`` on a worker. The function should call ``job.tick(...)`` as it goes."""
    job = Job(id=uuid.uuid4().hex[:12], label=label)

    def tick(frac: float, msg: str = "") -> None:
        if job.canceled:
            raise Canceled()
        job.progress = max(0.0, min(1.0, frac))
        if msg:
            job.message = msg
    job.tick = tick                                        # type: ignore[attr-defined]

    def run() -> None:
        job.status = "running"
        try:
            job.result = fn(job)
            job.status = "done"
            job.progress = 1.0
            job.message = job.message or "finished"
        except Canceled:
            job.status = "canceled"
            job.message = "canceled"
        except Exception as exc:                            # surfaced in the UI, not swallowed
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
            traceback.print_exc()
        finally:
            job.finished = time.time()

    with _LOCK:
        _JOBS[job.id] = job
        # keep the table from growing without bound across a long session
        finished = sorted((j for j in _JOBS.values() if j.done), key=lambda j: j.finished or 0)
        for old in finished[:-_MAX_KEPT]:
            _JOBS.pop(old.id, None)
    _POOL.submit(run)
    return job


def get(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)


def recent(n: int = 10) -> List[Job]:
    with _LOCK:
        return sorted(_JOBS.values(), key=lambda j: j.started, reverse=True)[:n]
