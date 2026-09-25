"""FastAPI + HTMX front end for the pairs-trading research.

    uvicorn webapp.main:app --reload
    # or: python -m webapp

Three pages, matching the three compute tiers in ``webapp.compute``:

    /            explorer   — one pair; tier-1 knobs live, tier-2 refit on a button
    /screen      browser    — the 1.7M cached cointegration tests, filterable
    /portfolio   backtest   — notebook 11's engine as a cancellable background job

Every figure is rendered server-side by ``pairs.plotting`` and swapped in as an ``<img>``, so what
the app draws and what the notebooks draw are the same code. There is no client-side charting and
no build step; HTMX is a single script tag.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from . import figures, jobs
from .compute import (HedgeSpec, SignalSpec, fit_hedge, holdout, portfolio_backtest,
                      run_signals, walk_forward)
from .state import REGISTRY, MissingCache, available_sources

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Pairs trading explorer", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.filters["money"] = lambda v: f"${v:,.0f}"
templates.env.filters["pct"] = lambda v: f"{v * 100:.1f}%"
templates.env.filters["num"] = lambda v, d=2: ("—" if v is None or (isinstance(v, float) and
                                                                    not np.isfinite(v))
                                               else f"{v:,.{d}f}")


def _specs(form) -> tuple[HedgeSpec, SignalSpec]:
    g = form.get
    def f(name, default, cast=float):
        v = g(name)
        if v is None or v == "":
            return default
        try:
            return cast(v)
        except (TypeError, ValueError):
            return default
    hedge = HedgeSpec(
        source=g("source") or "yahoo", t1=g("t1") or "BKNG", t2=g("t2") or "MA",
        model=g("model") or "kalman", q=f("q", 1e-5), em_iters=f("em_iters", 5, int),
        rolling_window=f("rolling_window", 252, int))
    sig = SignalSpec(
        z_method=g("z_method") or "robust",
        z_window=f("z_window", None, int) if g("z_window") else None,
        z_entry=f("z_entry", 2.0), z_exit=f("z_exit", 0.5), z_stop=f("z_stop", 4.0),
        max_hold_bars=f("max_hold_bars", None, int) if g("max_hold_bars") else None,
        cooldown_bars=f("cooldown_bars", 0, int), capital=f("capital", 10_000.0),
        cost_bps=f("cost_bps", 5.0), borrow_bps=f("borrow_bps", 50.0))
    return hedge, sig


def _explorer_context(hedge: HedgeSpec, sig: SignalSpec) -> dict:
    fitted = fit_hedge(hedge)
    states = fitted["states"]
    res = run_signals(states, sig)
    s = res["summary"]
    key = f"{hedge.key()}|{sig}"
    wf = walk_forward(states, sig)
    return {
        "hedge": hedge, "sig": sig, "summary": s, "fitted": fitted,
        "z_window": res["z_window"], "n_bars": len(states),
        "fig_signals": figures.signals(states, res["signals"], hedge.t1, hedge.t2,
                                       sig.z_entry, sig.z_exit, sig.z_stop, key),
        "fig_equity": figures.equity(res["daily"], key + "|eq"),
        "fig_folds": figures.folds(wf, float(s["sharpe"]), key + "|wf") if len(wf) else None,
        "wf": wf, "touches": REGISTRY.holdout_touches,
        "has_holdout": hedge.source == "yahoo" and REGISTRY.yahoo_oos(hedge.t1, hedge.t2) is not None,
    }


# ── explorer ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def explorer(request: Request):
    hedge, sig = HedgeSpec(), SignalSpec()
    ctx = _explorer_context(hedge, sig)
    return templates.TemplateResponse(request, "explorer.html", {"sources": available_sources(),
        "pairs": REGISTRY.pairs_for(hedge.source), **ctx})


@app.post("/explore/run", response_class=HTMLResponse)
async def explore_run(request: Request):
    """Tier 1 (and tier 2 when the pair or hedge changed — fit_hedge memoizes either way)."""
    hedge, sig = _specs(await request.form())
    try:
        ctx = _explorer_context(hedge, sig)
    except (ValueError, MissingCache) as exc:
        return HTMLResponse(f'<div class="err">{exc}</div>')
    return templates.TemplateResponse(request, "partials/result.html", {**ctx})


@app.post("/explore/pairs", response_class=HTMLResponse)
async def explore_pairs(request: Request):
    """Swap the pair menu when the source changes."""
    form = await request.form()
    src = form.get("source") or "yahoo"
    return templates.TemplateResponse(request, "partials/pairs.html", {"pairs": REGISTRY.pairs_for(src)})


@app.post("/explore/holdout", response_class=HTMLResponse)
async def explore_holdout(request: Request):
    hedge, sig = _specs(await request.form())
    res = holdout(hedge, sig)
    if res is None:
        return HTMLResponse('<div class="err">No hold-out cache for this pair. '
                            'Only the pairs notebooks 01–05 traded were ever downloaded.</div>')
    key = f"holdout|{hedge.key()}|{sig}"
    return templates.TemplateResponse(request, "partials/holdout.html", {"summary": res["summary"], "touches": res["touches"],
        "fig": figures.signals(res["states"], res["signals"], hedge.t1, hedge.t2,
                               sig.z_entry, sig.z_exit, sig.z_stop, key)})


@app.get("/fig/{token}")
def figure(token: str):
    data = figures.get(token)
    if data is None:
        return Response(status_code=404)
    return Response(data, media_type="image/png",
                    headers={"Cache-Control": "public, max-age=3600"})


# ── screen browser ───────────────────────────────────────────────────────────

@app.get("/screen", response_class=HTMLResponse)
def screen(request: Request):
    try:
        forms = REGISTRY.formations()
    except MissingCache as exc:
        return HTMLResponse(f'<div class="err">{exc}</div>')
    return templates.TemplateResponse(request, "screen.html", {"formations": forms, "selected": forms[-1]})


@app.post("/screen/rows", response_class=HTMLResponse)
async def screen_rows(request: Request):
    form = await request.form()
    scr = REGISTRY.day_screen()
    f = pd.Timestamp(form.get("formation"))
    sub = scr[pd.to_datetime(scr["formation"]) == f]
    total = len(sub)
    if form.get("only_pass") == "on":
        sub = sub[sub["verdict"].eq("pass")]
    try:
        pmax = float(form.get("p_max") or 1.0)
    except ValueError:
        pmax = 1.0
    sub = sub[sub["eg_p"] <= pmax]
    sort = form.get("sort") or "eg_p"
    sub = sub.nsmallest(200, sort) if sort in ("eg_p", "eg_p_fdr") else sub.nlargest(200, sort)
    return templates.TemplateResponse(request, "partials/screen_rows.html", {"rows": sub.to_dict("records"),
        "shown": len(sub), "matched": int((scr["formation"] == f).sum()), "total": total,
        "kept": len(sub)})


# ── portfolio backtest ───────────────────────────────────────────────────────

@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_page(request: Request):
    return templates.TemplateResponse(request, "portfolio.html", {"jobs": jobs.recent()})


@app.post("/portfolio/run", response_class=HTMLResponse)
async def portfolio_run(request: Request):
    form = await request.form()
    _, sig = _specs(form)
    rule = form.get("rule") or "bh_dual"
    excl = form.get("exclude_levered") == "on"
    label = f"{rule} · {sig.cost_bps:g} bps" + (" · no levered ETFs" if excl else "")

    def work(job):
        out = portfolio_backtest(rule, sig, exclude_levered=excl, job=job)
        out["fig"] = figures.portfolio(out["pnl"], out["active"], f"pf|{label}|{id(out)}",
                                       f"{rule} — {out['n_folds']} pair-folds, "
                                       f"{out['n_trades']:,} trades")
        return out
    job = jobs.submit(label, work)
    return templates.TemplateResponse(request, "partials/job.html", {"job": job})


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_status(request: Request, job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return HTMLResponse('<div class="err">job not found</div>')
    return templates.TemplateResponse(request, "partials/job.html", {"job": job})


@app.post("/jobs/{job_id}/cancel", response_class=HTMLResponse)
def job_cancel(request: Request, job_id: str):
    job = jobs.get(job_id)
    if job is not None:
        job.cancel()
    return templates.TemplateResponse(request, "partials/job.html", {"job": job})


def main() -> None:                                        # pragma: no cover
    import uvicorn
    uvicorn.run("webapp.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":                                 # pragma: no cover
    main()
