"""Tests for the FastAPI + HTMX explorer.

The point of most of these is not that the app renders, but that it renders *the same numbers the
notebooks do*. An explorer that quietly diverges from the research behind it would be worse than no
explorer, and this repository has been bitten by that class of drift more than once.

Anything needing a notebook cache is skipped when the cache is absent, so a clean checkout still
passes.
"""
import warnings

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("matplotlib")
from fastapi.testclient import TestClient      # noqa: E402

from webapp import figures                      # noqa: E402
from webapp.compute import HedgeSpec, SignalSpec, fit_hedge, holdout, run_signals, walk_forward
from webapp.state import CACHE, REGISTRY, available_sources   # noqa: E402

warnings.filterwarnings("ignore")

YAHOO = pytest.mark.skipif(
    not (CACHE / "nb02_train_2020-01-01_2025-12-31.parquet").exists(),
    reason="notebook 02's price cache is not present")
HOLDOUT = pytest.mark.skipif(
    not (CACHE / "nb02_oos_BKNG_MA_2026-01-01_2026-06-25.parquet").exists(),
    reason="the 2026 hold-out cache is not present")
DAY = pytest.mark.skipif(not (CACHE / "day_screen.parquet").exists(),
                         reason="notebook 10's screen cache is not present")


@pytest.fixture(scope="module")
def client():
    from webapp.main import app
    return TestClient(app)


# ── the numbers must match the notebooks ─────────────────────────────────────

@YAHOO
@HOLDOUT
def test_holdout_reproduces_notebook_01_exactly():
    """Notebook 01 §4b prints Sharpe 2.64203 on 12 trades for BKNG/MA at 1 bp a leg-side.

    If this drifts, the explorer is showing something the research does not support.
    """
    res = holdout(HedgeSpec(), SignalSpec(cost_bps=1.0))
    s = res["summary"]
    assert int(s["n_trades"]) == 12
    assert float(s["sharpe"]) == pytest.approx(2.64203, abs=1e-5)
    assert float(s["net_pnl"]) == pytest.approx(664.302195, abs=1e-4)
    assert res["z_window"] == 30


@YAHOO
def test_tier_two_is_memoized_so_tier_one_never_pays_for_it():
    spec = HedgeSpec()
    first = fit_hedge(spec)
    assert fit_hedge(spec) is first, "an identical spec must not refit"
    assert fit_hedge(HedgeSpec(model="static")) is not first


@YAHOO
@pytest.mark.parametrize("model", ["kalman", "static", "rolling"])
def test_every_hedge_model_produces_a_tradeable_state_frame(model):
    fitted = fit_hedge(HedgeSpec(model=model))
    states = fitted["states"]
    assert {"P1", "P2", "beta", "resid"} <= set(states.columns)
    assert len(states) > 250 and states["resid"].notna().any()
    out = run_signals(states, SignalSpec())
    assert {"signals", "daily", "trades", "summary"} <= set(out)


@YAHOO
def test_widening_the_entry_threshold_cannot_increase_trade_count():
    states = fit_hedge(HedgeSpec())["states"]
    counts = [int(run_signals(states, SignalSpec(z_entry=z))["summary"]["n_trades"])
              for z in (1.5, 2.0, 3.0, 4.0)]
    assert counts == sorted(counts, reverse=True), counts


@YAHOO
def test_walk_forward_returns_one_row_per_fold_with_finite_sharpes():
    wf = walk_forward(fit_hedge(HedgeSpec())["states"], SignalSpec())
    assert len(wf) >= 5
    assert set(wf.columns) == {"start", "end", "sharpe", "net_pnl", "trades"}
    assert wf["sharpe"].notna().all()
    assert (wf["end"] > wf["start"]).all()


# ── the registry ─────────────────────────────────────────────────────────────

def test_sources_report_their_own_availability():
    for s in available_sources():
        assert s.available() and s.label and s.note


@YAHOO
def test_memo_is_reentrant_across_keys():
    """fit_hedge asks the registry for prices from inside a memoized call.

    A single non-reentrant lock deadlocks here; the per-key locks must not.
    """
    REGISTRY._cache.pop("hedge:" + HedgeSpec().key(), None)
    assert fit_hedge(HedgeSpec())["states"] is not None


# ── figures ──────────────────────────────────────────────────────────────────

def test_figure_cache_round_trips_and_evicts():
    tok = figures.cached("test-key-1", lambda: b"\x89PNG-not-really")
    assert figures.get(tok) == b"\x89PNG-not-really"
    calls = []
    figures.cached("test-key-1", lambda: calls.append(1) or b"x")
    assert not calls, "an identical key must not re-render"
    assert figures.get("nosuchtoken") is None


# ── routes ───────────────────────────────────────────────────────────────────

@YAHOO
def test_explorer_page_renders_with_figures(client):
    r = client.get("/")
    assert r.status_code == 200
    import re
    tokens = re.findall(r"/fig/([0-9a-f]{16})", r.text)
    assert len(tokens) >= 2
    for t in tokens:
        f = client.get(f"/fig/{t}")
        assert f.status_code == 200 and f.headers["content-type"] == "image/png"


def test_unknown_figure_token_is_404_not_500(client):
    assert client.get("/fig/" + "0" * 16).status_code == 404


@YAHOO
def test_tier_one_post_returns_a_fragment_not_a_page(client):
    r = client.post("/explore/run", data={
        "source": "yahoo", "t1": "BKNG", "t2": "MA", "model": "kalman",
        "z_entry": "2.5", "z_exit": "0.5", "z_stop": "4", "cost_bps": "5"})
    assert r.status_code == 200
    assert "<html" not in r.text.lower(), "an HTMX swap target must not be a whole page"
    assert "Sharpe" in r.text


@YAHOO
def test_a_bad_ticker_is_reported_not_raised(client):
    r = client.post("/explore/run", data={"source": "yahoo", "t1": "NOSUCH", "t2": "MA"})
    assert r.status_code == 200 and "err" in r.text


@YAHOO
def test_changing_the_source_swaps_the_pair_menu(client):
    r = client.post("/explore/pairs", data={"source": "yahoo"})
    assert r.status_code == 200 and "<option" in r.text
    assert "selected" in r.text


@DAY
def test_screen_rows_filter_and_stay_bounded(client):
    forms = REGISTRY.formations()
    r = client.post("/screen/rows", data={
        "formation": str(forms[len(forms) // 2].date()), "p_max": "0.05",
        "only_pass": "on", "sort": "eg_p"})
    assert r.status_code == 200
    assert r.text.count("<tr>") <= 201, "the table must never stream the whole screen"


def test_portfolio_page_renders(client):
    assert client.get("/portfolio").status_code == 200


def test_jobs_report_missing_ids_without_raising(client):
    r = client.get("/jobs/deadbeefcafe")
    assert r.status_code == 200 and "not found" in r.text


def test_job_lifecycle_runs_and_cancels():
    from webapp import jobs
    import time

    def slow(job):
        for i in range(200):
            job.tick(i / 200, f"step {i}")
            time.sleep(0.01)
        return "finished"

    j = jobs.submit("test", slow)
    for _ in range(200):
        if j.status == "running":
            break
        time.sleep(0.01)
    j.cancel()
    for _ in range(300):
        if j.done:
            break
        time.sleep(0.01)
    assert j.status == "canceled" and j.done

    ok = jobs.submit("quick", lambda job: 42)
    for _ in range(300):
        if ok.done:
            break
        time.sleep(0.01)
    assert ok.status == "done" and ok.result == 42


def test_a_failing_job_surfaces_the_error_rather_than_hanging():
    from webapp import jobs
    import time

    def boom(job):
        raise ValueError("deliberate")

    j = jobs.submit("boom", boom)
    for _ in range(300):
        if j.done:
            break
        time.sleep(0.01)
    assert j.status == "failed" and "deliberate" in j.error
