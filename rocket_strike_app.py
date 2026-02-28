"""
Web interface: live P(strike) in 5, 15, 60 minutes from true past data.
Fetches data every 5 minutes and updates probabilities.
Run: python rocket_strike_app.py  then open http://127.0.0.1:5050  (or set PORT=5001 for another port)
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Import from the hazard NN module (same logic, no CLI)
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Suppress tqdm in web context to avoid log noise
import os
_env_tqdm = os.environ.get("TQDM_DISABLE")
os.environ["TQDM_DISABLE"] = "1"

from rocket_strike_hawkes_process import fit_hawkes, hawkes_predict, hawkes_backtest_intensity

from rocket_strike_hazard_nn import (
    FEATURE_COLS_MINUTE,
    build_minute_timeline,
    hazard_features_minute,
    train_test_split_by_minutes,
    build_sequences_hazard_minute,
    train_hazard_nn,
    predict_proba_strike,
    _derive_horizon_probs_from_current,
    _next_hour_features,
)

if _env_tqdm is not None:
    os.environ["TQDM_DISABLE"] = _env_tqdm
else:
    os.environ.pop("TQDM_DISABLE", None)

# Horizons for web: 5, 15, 60 minutes
WEB_HORIZONS = [5, 15, 60]

# Shared state (updated by background thread and read by API)
_state = {
    "model": None,
    "scaler": None,
    "probs": None,           # {"5": float, "15": float, "60": float, "1": float}
    "updated_at": None,      # ISO datetime
    "error": None,
    "ready": False,
    "backtest_5h": None,     # {"times": [iso...], "actual": [0|1...], "pred": [float...]}
    "next_15_probs": None,   # [p1, p2, ..., p15] for next 15 minutes
    "hawkes_params": None,   # (mu, alpha, beta) fitted Hawkes parameters
}
_lock = threading.Lock()

# Use fewer rows on low-memory hosts (e.g. Render free tier 512MB). Set MAX_MINUTE_ROWS=500000 locally for full data.
MAX_MINUTE_ROWS = int(os.environ.get("MAX_MINUTE_ROWS", "80000"))

# Static state: train locally, export live_state.json, then deploy with USE_STATIC_STATE=1 to only serve graphs (no training on server)
CACHE_DIR = _PROJECT_ROOT / "data_cache"
LIVE_STATE_FILE = CACHE_DIR / "live_state.json"
# Pretrained model: train locally, save with joblib; server loads it and only fetches recent data to update probs/graphs
MODEL_FILE = CACHE_DIR / "model.joblib"
# When we loaded a pretrained model, refresh uses only last N CSV rows (low memory)
_REFRESH_MAX_ROWS = None

try:
    import joblib
except ImportError:
    joblib = None


def _load_pretrained_model() -> bool:
    """Load model and scaler from MODEL_FILE. Returns True if loaded. Server then uses recent-only data to update probs."""
    if joblib is None or not MODEL_FILE.exists():
        return False
    try:
        obj = joblib.load(MODEL_FILE)
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            model, scaler = obj[0], obj[1]
        else:
            return False
        with _lock:
            _state["model"] = model
            _state["scaler"] = scaler
        return True
    except Exception:
        return False


def _load_static_state() -> bool:
    """Load pre-computed state from live_state.json. Returns True if loaded."""
    import json
    if not LIVE_STATE_FILE.exists():
        return False
    try:
        with open(LIVE_STATE_FILE) as f:
            data = json.load(f)
        with _lock:
            _state["probs"] = {
                "1": data["probs"]["1"],
                5: data["probs"]["5"],
                15: data["probs"]["15"],
                60: data["probs"]["60"],
            }
            _state["backtest_5h"] = data.get("backtest_5h")
            _state["next_15_probs"] = data.get("next_15_probs")
            _state["updated_at"] = data.get("updated_at", "")
            _state["ready"] = True
            _state["error"] = None
        return True
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
        return False


def _extend_backtest_to_now(backtest_5h: dict) -> dict:
    """If backtest_5h ends before current minute (UTC), append minutes so the graph extends to 'now'."""
    if not backtest_5h or not backtest_5h.get("times"):
        return backtest_5h
    times = backtest_5h["times"]
    last_iso = times[-1]
    try:
        last_ts = pd.Timestamp(last_iso)
    except Exception:
        return backtest_5h
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(timezone.utc)
    now_utc = datetime.now(tz=timezone.utc)
    now_floor = now_utc.replace(second=0, microsecond=0)
    now_ts = pd.Timestamp(now_floor)
    if last_ts.tzinfo is not None and now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize(timezone.utc)
    if last_ts >= now_ts:
        return backtest_5h
    # Cap extension at 2 hours to avoid runaway padding
    extra = pd.date_range(start=last_ts + pd.Timedelta(minutes=1), end=now_ts, freq="min")
    if len(extra) > 120:
        extra = extra[-120:]
    if len(extra) == 0:
        return backtest_5h
    last_pred = backtest_5h["pred"][-1] if backtest_5h.get("pred") else 0.0
    extra_iso = [t.isoformat() for t in extra]
    return {
        "times": list(times) + extra_iso,
        "actual": list(backtest_5h["actual"]) + [0] * len(extra),
        "pred": list(backtest_5h["pred"]) + [float(last_pred)] * len(extra),
    }


def _load_all_strike_times_minutes() -> list:
    """Load ALL historical strike times (minutes from epoch) directly from the alerts CSV.
    This gives 12,000+ events across 11 years — far more than the in-memory window.
    Used for Hawkes parameter fitting only (not for current intensity).
    """
    from rocket_strike_hazard_nn import GITHUB_ALERTS_CACHE
    if not GITHUB_ALERTS_CACHE.exists():
        return []
    try:
        raw = pd.read_csv(GITHUB_ALERTS_CACHE)
        # find the timestamp column
        date_col = None
        for c in raw.columns:
            if str(c).strip().lower() == "alertdate":
                date_col = c
                break
        if date_col is None:
            for c in raw.columns:
                if str(c).strip().lower() in ("alert_date", "datetime", "date"):
                    date_col = c
                    break
        if date_col is None:
            return []
        # Parse as UTC, deduplicate to minute-level (multiple alerts in same minute = 1 event)
        ts = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dropna()
        # Round to minute, deduplicate
        ts_min = ts.dt.floor("min").drop_duplicates().sort_values()
        # Use last 2 years — ~7k events, fast MLE, still covers multiple conflict episodes
        cutoff = ts_min.max() - pd.Timedelta(days=730)
        ts_min = ts_min[ts_min >= cutoff]
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        minutes = [(t - epoch).total_seconds() / 60.0 for t in ts_min]
        print(f"  Hawkes training: {len(minutes):,} strike-minutes "
              f"({ts_min.min().date()} → {ts_min.max().date()}, last 2 years)", flush=True)
        return minutes
    except Exception as e:
        print(f"  Warning: could not load full alert history for Hawkes: {e}", flush=True)
        return []


def _run_hawkes(df: pd.DataFrame) -> dict:
    """Fit a Hawkes process on ALL historical strike data and return predictions.

    Fitting uses the full alert CSV (12,000+ events, 11 years) for the best
    parameter estimates.  Current intensity uses only recent strikes from `df`
    (exponential decay means old events contribute negligibly anyway).
    """
    strike_col = "strike" if "strike" in df.columns else "strikes"
    dt_col     = "datetime" if "datetime" in df.columns else "dt"

    strike_rows = df[df[strike_col] == 1]
    if len(strike_rows) < 3:
        return {}

    now_ts  = pd.Timestamp(df[dt_col].iloc[-1])
    epoch   = pd.Timestamp("1970-01-01", tz="UTC")

    def to_epoch_min(t):
        ts = pd.Timestamp(t)
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Jerusalem")
        return (ts.tz_convert("UTC") - epoch).total_seconds() / 60.0

    now_min = to_epoch_min(now_ts)

    # ---- fit on ALL historical data (11 years, 12k+ events) ------------------
    all_strike_min = _load_all_strike_times_minutes()
    if len(all_strike_min) < 10:
        # fallback: use what's in df
        all_strike_min = [to_epoch_min(t) for t in strike_rows[dt_col]]

    mu, alpha, beta = fit_hawkes(all_strike_min, verbose=True)

    # ---- current intensity: use recent strikes from df (last 7 days) ---------
    # Old strikes have exp(-beta*(t_now - t_i)) ≈ 0, so only recent ones matter.
    cutoff = now_ts - pd.Timedelta(days=7)
    recent = strike_rows[pd.to_datetime(strike_rows[dt_col]) >= cutoff]
    if len(recent) < 1:
        recent = strike_rows

    recent_min = [to_epoch_min(t) for t in recent[dt_col]]
    past_relative = [t - now_min for t in recent_min]   # negative = past

    per_min, cumulative = hawkes_predict(past_relative, mu, alpha, beta, horizon=60)

    probs = {
        "1":  cumulative[0],
        5:    cumulative[4],
        15:   cumulative[14],
        60:   cumulative[59],
    }
    next_15 = per_min[:15]

    # ---- backtest: Hawkes intensity over last 5 h ----------------------------
    n_5h     = 5 * 60
    bt_start = max(0, len(df) - n_5h)
    bt_df    = df.iloc[bt_start:]
    bt_times_iso = [pd.Timestamp(t).isoformat() for t in bt_df[dt_col]]
    bt_actual    = [int(v) for v in bt_df[strike_col]]

    # Use all_strike_min (full history) as "past events" for causal backtest
    bt_query_min = [to_epoch_min(t) for t in bt_df[dt_col]]
    bt_pred = hawkes_backtest_intensity(all_strike_min, bt_query_min, mu, alpha, beta)

    return {
        "hawkes_params": (mu, alpha, beta),
        "probs":         probs,
        "next_15_probs": next_15,
        "backtest_5h": {
            "times":  bt_times_iso,
            "actual": bt_actual,
            "pred":   bt_pred,
        },
    }


def _compute_backtest_and_next_15(df, model, scaler):
    """Compute last 5h backtest (actual vs pred) and next 15 min curve. Returns (backtest_5h, next_15_probs)."""
    last_idx = len(df) - 1
    n_5h = 5 * 60
    start = max(1, last_idx - n_5h + 1)
    X_5h = df.iloc[start - 1 : last_idx][FEATURE_COLS_MINUTE].values.astype(float)
    pred_5h = predict_proba_strike(model, scaler, X_5h)
    actual_5h = df.iloc[start : last_idx + 1]["strike"].values
    times_5h = df.iloc[start : last_idx + 1]["datetime"]
    times_iso = [pd.Timestamp(t).isoformat() for t in times_5h]
    # Clamp probabilities to [0, 1] (should never exceed 1; guard against numerical glitches)
    pred_clamped = [float(max(0.0, min(1.0, x))) for x in pred_5h]
    backtest_5h = {
        "times": times_iso,
        "actual": [int(x) for x in actual_5h],
        "pred": pred_clamped,
    }
    next_15_X = _next_hour_features(df, last_idx, n_minutes=15)
    next_15_probs = [float(max(0.0, min(1.0, x))) for x in predict_proba_strike(model, scaler, next_15_X)]
    return backtest_5h, next_15_probs


def _train_and_predict(max_rows: Optional[int] = None, keep_last_minutes: Optional[int] = None,
                       train_on_all: bool = False):
    """Build timeline, train 1-min model, compute current probs. Uses true data only.
    keep_last_minutes: trim timeline to this many minutes so train and inference use the same distribution.
    train_on_all: if True, train on ALL data (no test holdout) – use this for export so the model
                  learns from the full current window including the most recent active period."""
    try:
        timeline = build_minute_timeline(max_rows=max_rows, keep_last_minutes=keep_last_minutes)
        if timeline is None or len(timeline) < 1000:
            with _lock:
                _state["error"] = "Not enough minute-level data (need GitHub/Kaggle timeline)."
                _state["ready"] = False
            return
        if max_rows is None and keep_last_minutes is None and len(timeline) > MAX_MINUTE_ROWS:
            timeline = timeline.iloc[-MAX_MINUTE_ROWS:].reset_index(drop=True)
        df = hazard_features_minute(timeline)
        if train_on_all:
            # For export: train on ALL rows so the model sees the full current active period.
            # The train/test split would otherwise put recent strikes in the test set (they'd be
            # invisible to the model), causing near-zero predictions during sudden escalations.
            train_df = df
        else:
            if max_rows is not None:
                test_minutes = max(300, int(0.2 * len(df)))
            else:
                test_minutes = 7 * 24 * 60
            train_df, test_df = train_test_split_by_minutes(df, test_minutes=test_minutes)
            if len(test_df) < 2:
                n = len(df) - 1
                split = int(0.8 * n)
                train_df = df.iloc[: split + 1]
        X_train, y_train = build_sequences_hazard_minute(train_df, horizon_minutes=1)
        if X_train.size == 0 or y_train.sum() == 0:
            with _lock:
                _state["error"] = "No training sequences (no positives?)."
                _state["ready"] = False
            return
        model, scaler = train_hazard_nn(X_train, y_train, class_weight_balanced=True)
        # Hawkes process: uses raw strike times, no feature engineering needed.
        # Overrides MLP probabilities — more principled for self-exciting events.
        hk = _run_hawkes(df)
        if hk:
            probs        = hk["probs"]
            backtest_5h  = hk["backtest_5h"]
            next_15_probs = hk["next_15_probs"]
        else:
            # Fall back to MLP if Hawkes fails (too few strikes)
            last_idx = len(df) - 1
            last_X = df.iloc[last_idx][FEATURE_COLS_MINUTE].values.astype(float).reshape(1, -1)
            p_1 = float(predict_proba_strike(model, scaler, last_X)[0])
            probs = _derive_horizon_probs_from_current(p_1, WEB_HORIZONS)
            probs["1"] = p_1
            backtest_5h, next_15_probs = _compute_backtest_and_next_15(df, model, scaler)
            if next_15_probs:
                prod_no_strike = 1.0
                for p in next_15_probs:
                    prod_no_strike *= max(0.0, min(1.0, 1.0 - p))
                probs[15] = min(1.0, 1.0 - prod_no_strike)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with _lock:
            _state["model"] = model
            _state["scaler"] = scaler
            _state["probs"] = probs
            _state["updated_at"] = now_iso
            _state["backtest_5h"] = backtest_5h
            _state["next_15_probs"] = next_15_probs
            _state["hawkes_params"] = hk.get("hawkes_params") if hk else None
            _state["error"] = None
            _state["ready"] = True
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
            _state["ready"] = False


def _refresh_data_only(max_rows: Optional[int] = None, keep_last_minutes: Optional[int] = None):
    """Re-fetch timeline and features, recompute probs with existing model (no retrain).
    When max_rows is set, only last max_rows CSV rows are loaded (low memory, live update).
    keep_last_minutes: when set, keep this many minutes (e.g. 30*24*60) so features match full-data-trained model."""
    with _lock:
        model, scaler = _state.get("model"), _state.get("scaler")
        use_recent = max_rows if max_rows is not None else _REFRESH_MAX_ROWS
    if model is None or scaler is None:
        _train_and_predict()
        return
    try:
        timeline = build_minute_timeline(max_rows=use_recent, keep_last_minutes=keep_last_minutes)
        if timeline is None or len(timeline) < 100:
            return
        if use_recent is None and len(timeline) > MAX_MINUTE_ROWS:
            timeline = timeline.iloc[-MAX_MINUTE_ROWS:].reset_index(drop=True)
        df = hazard_features_minute(timeline)
        last_idx = len(df) - 1
        last_X = df.iloc[last_idx][FEATURE_COLS_MINUTE].values.astype(float).reshape(1, -1)
        p_1 = float(predict_proba_strike(model, scaler, last_X)[0])
        # OOD check: if all predictions in recent window are numerically zero, the pretrained model
        # is out of distribution (e.g. trained on quiet-period data, now there is high activity).
        # Trigger a full retrain so the scaler calibrates on current data.
        if p_1 < 1e-100:
            sample_X = df.iloc[max(0, last_idx - 59) : last_idx][FEATURE_COLS_MINUTE].values.astype(float)
            sample_preds = predict_proba_strike(model, scaler, sample_X)
            if sample_preds.max() < 1e-100:
                print("     Pretrained model gives near-zero predictions (likely out of distribution). Retraining on current window...", flush=True)
                _train_and_predict(max_rows=_REFRESH_MAX_ROWS, keep_last_minutes=keep_last_minutes, train_on_all=True)
                return
        # Hawkes overrides MLP for all predictions
        hk = _run_hawkes(df)
        if hk:
            probs         = hk["probs"]
            backtest_5h   = hk["backtest_5h"]
            next_15_probs = hk["next_15_probs"]
        else:
            probs = _derive_horizon_probs_from_current(p_1, WEB_HORIZONS)
            probs["1"] = p_1
            backtest_5h, next_15_probs = _compute_backtest_and_next_15(df, model, scaler)
            if next_15_probs:
                prod_no_strike = 1.0
                for p in next_15_probs:
                    prod_no_strike *= max(0.0, min(1.0, 1.0 - p))
                probs[15] = min(1.0, 1.0 - prod_no_strike)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with _lock:
            _state["probs"] = probs
            _state["updated_at"] = now_iso
            _state["backtest_5h"] = backtest_5h
            _state["next_15_probs"] = next_15_probs
            _state["hawkes_params"] = hk.get("hawkes_params") if hk else None
            _state["error"] = None
            _state["ready"] = True
    except Exception as e:
        with _lock:
            _state["error"] = str(e)


def _background_refresh(max_rows: Optional[int] = None, keep_last_minutes: Optional[int] = None):
    """Run data refresh every 5 minutes. When max_rows set, use recent-only fetch (for pretrained model)."""
    while True:
        time.sleep(300)  # 5 minutes
        _refresh_data_only(max_rows=max_rows, keep_last_minutes=keep_last_minutes)


def create_app():
    from flask import Flask, jsonify, render_template_string

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(INDEX_HTML)

    @app.route("/bg")
    def background_image():
        from flask import send_file, abort
        img = _PROJECT_ROOT / "background.jpg"
        if not img.exists():
            abort(404)
        return send_file(img, mimetype="image/jpeg")

    @app.route("/api/probs")
    def api_probs():
        with _lock:
            if not _state["ready"]:
                return jsonify({
                    "ok": False,
                    "error": _state.get("error") or "No model yet. Run: python export_live_state.py then add data_cache/model.joblib to the project.",
                    "updated_at": _state.get("updated_at"),
                }), 503
            out = {
                "ok": True,
                "p_1": _state["probs"]["1"],
                "p_5": _state["probs"][5],
                "p_15": _state["probs"][15],
                "p_60": _state["probs"][60],
                "updated_at": _state["updated_at"],
                "now": datetime.now(tz=timezone.utc).isoformat(),
            }
            if _state.get("backtest_5h"):
                out["backtest_5h"] = _extend_backtest_to_now(_state["backtest_5h"])
            probs15 = _state.get("next_15_probs") or []
            out["next_15_probs"] = probs15
            # Compute first minute where cumulative P(≥1 strike) crosses 50%
            if probs15:
                cum, prod = 0.0, 1.0
                next_alarm_min = None
                for k, p in enumerate(probs15, start=1):
                    prod *= max(0.0, 1.0 - p)
                    cum = 1.0 - prod
                    if cum >= 0.50 and next_alarm_min is None:
                        next_alarm_min = k
                out["next_alarm_min"] = next_alarm_min  # None if <50% in 15 min
            return jsonify(out)

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>🇮🇱 Israel vs Iran 🇮🇷 Round 2</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0a0a0f;
      --surface: #111118;
      --card: rgba(18,18,26,0.82);
      --card-border: rgba(255,255,255,0.07);
      --text: #e2e0dd;
      --muted: #666;
      --muted2: #999;
      --accent: #ff4444;
      --warn: #f59e0b;
      --ok: #22c55e;
      --blue: #3b82f6;
      --radius: 14px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
      background: url('/bg') center top / cover fixed no-repeat;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.72);
      z-index: 0;
      pointer-events: none;
    }
    .header, .main-content, #nextAlarmBanner, .status-row, .section-header, .chart-card {
      position: relative;
      z-index: 1;
    }
    /* Header */
    .header {
      background: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.65));
      border-bottom: 1px solid rgba(255,60,60,0.25);
      padding: 1.5rem 2rem 1.25rem;
    }
    .header-inner { max-width: 960px; margin: 0 auto; }
    .header h1 {
      font-size: 1.2rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: #fff;
    }
    .header h1 span { color: var(--accent); }
    .header .subtitle {
      font-size: 0.8rem;
      color: var(--muted2);
      margin-top: 0.25rem;
    }
    /* Main content */
    .main { max-width: 960px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }
    /* Threat level bar */
    .threat-section { margin-bottom: 1.5rem; }
    .threat-label-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 0.4rem;
    }
    .threat-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }
    .threat-badge {
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.15rem 0.6rem;
      border-radius: 99px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .threat-badge.low    { background: rgba(34,197,94,0.15);  color: var(--ok);   border: 1px solid rgba(34,197,94,0.3); }
    .threat-badge.mid    { background: rgba(245,158,11,0.15); color: var(--warn); border: 1px solid rgba(245,158,11,0.3); }
    .threat-badge.high   { background: rgba(255,68,68,0.15);  color: var(--accent); border: 1px solid rgba(255,68,68,0.3); }
    .threat-badge.critical { background: rgba(255,68,68,0.25); color: #ff8080; border: 1px solid rgba(255,100,100,0.5); }
    .threat-track {
      height: 6px;
      background: rgba(255,255,255,0.07);
      border-radius: 99px;
      overflow: hidden;
    }
    .threat-fill {
      height: 100%;
      border-radius: 99px;
      transition: width 0.6s ease, background 0.6s ease;
      background: var(--ok);
    }
    /* Probability cards */
    .cards {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.5rem;
    }
    @media (max-width: 480px) { .cards { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 1.1rem 1rem;
      text-align: center;
      position: relative;
      overflow: hidden;
      transition: border-color 0.4s;
    }
    .card::before {
      content: '';
      position: absolute;
      inset: 0;
      opacity: 0;
      transition: opacity 0.4s;
      pointer-events: none;
    }
    .card.high::before  { background: radial-gradient(ellipse at 50% 0%, rgba(255,68,68,0.12) 0%, transparent 70%); opacity: 1; }
    .card.high  { border-color: rgba(255,68,68,0.25); }
    .card.mid::before   { background: radial-gradient(ellipse at 50% 0%, rgba(245,158,11,0.10) 0%, transparent 70%); opacity: 1; }
    .card.mid   { border-color: rgba(245,158,11,0.2); }
    .card-horizon { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 0.5rem; }
    .card-value {
      font-size: 2.2rem;
      font-weight: 800;
      line-height: 1;
      transition: color 0.4s;
      color: var(--text);
    }
    .card.high .card-value  { color: var(--accent); }
    .card.mid  .card-value  { color: var(--warn); }
    .card.low  .card-value  { color: var(--ok); }
    .card-sub { font-size: 0.7rem; color: var(--muted); margin-top: 0.4rem; }
    /* Status row */
    .status-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-bottom: 1.75rem;
      font-size: 0.78rem;
      color: var(--muted2);
    }
    .live-dot {
      display: inline-block;
      width: 7px; height: 7px;
      border-radius: 50%;
      background: var(--ok);
      margin-right: 5px;
      vertical-align: middle;
      animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%,100%{ opacity:1; box-shadow:0 0 0 0 rgba(34,197,94,0.5); }
                       50%{ opacity:0.7; box-shadow:0 0 0 4px rgba(34,197,94,0); } }
    /* Section headers */
    .section-header {
      display: flex;
      align-items: baseline;
      gap: 0.6rem;
      margin: 1.75rem 0 0.35rem;
    }
    .section-header h2 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); font-weight: 600; }
    .section-header .section-note { font-size: 0.75rem; color: var(--muted); }
    /* Charts */
    .chart-card {
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 1rem 1rem 0.75rem;
      margin-bottom: 1rem;
    }
    .chart-card canvas { display: block; max-height: 200px; }
    /* Error */
    .err-box {
      background: rgba(255,68,68,0.08);
      border: 1px solid rgba(255,68,68,0.25);
      border-radius: 10px;
      padding: 0.75rem 1rem;
      color: #ff8080;
      font-size: 0.85rem;
      margin-bottom: 1rem;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="header-inner">
      <h1>🇮🇱 ISRAEL VS. IRAN 🇮🇷 <span>ROUND 2</span></h1>
      <div class="subtitle">AI-predicted P(rocket strike) in the next 5 / 15 / 60 minutes &mdash; from live Tzeva Adom data, refreshed every 5 minutes.</div>
    </div>
  </div>

  <div class="main">
    <div id="errBox" class="err-box" style="display:none"></div>

    <!-- Threat level -->
    <div class="threat-section">
      <div class="threat-label-row">
        <span class="threat-title">Threat Level (next 15 min)</span>
        <span class="threat-badge low" id="threatBadge">—</span>
      </div>
      <div class="threat-track"><div class="threat-fill" id="threatFill" style="width:0%"></div></div>
    </div>

    <!-- Probability cards -->
    <div class="cards">
      <div class="card" id="card5">
        <div class="card-horizon">Next 5 min</div>
        <div class="card-value" id="p5">—</div>
        <div class="card-sub">P(at least one strike)</div>
      </div>
      <div class="card" id="card15">
        <div class="card-horizon">Next 15 min</div>
        <div class="card-value" id="p15">—</div>
        <div class="card-sub">P(at least one strike)</div>
      </div>
      <div class="card" id="card60">
        <div class="card-horizon">Next 60 min</div>
        <div class="card-value" id="p60">—</div>
        <div class="card-sub">P(at least one strike)</div>
      </div>
    </div>

    <!-- Status row -->
    <div class="status-row">
      <span><span class="live-dot"></span><span id="updated">Loading&hellip;</span></span>
      <span id="nextRefresh"></span>
    </div>

    <!-- Next alarm banner -->
    <div id="nextAlarmBanner" style="display:none; max-width:960px; margin:0.75rem auto; padding:0.75rem 1.25rem;
         border-radius:8px; background:rgba(255,68,68,0.12); border:1px solid rgba(255,68,68,0.35);
         font-size:0.95rem; font-weight:600; color:#ff6060; text-align:center; letter-spacing:0.03em;">
    </div>

    <!-- Backtest chart -->
    <div class="section-header">
      <h2>Strike timeline &amp; next strike forecast</h2>
      <span class="section-note" id="chart5hEnd">Last 20 strikes + forecast of when the next one may occur</span>
    </div>
    <div class="chart-card"><canvas id="chartBacktest"></canvas></div>

  </div>

  <script>
    var GRID = 'rgba(255,255,255,0.05)';
    var TICK = '#555';

    function fmtP(p) {
      if (p == null) return '—';
      var pct = p * 100;
      return pct < 0.01 ? '<0.01%' : pct.toFixed(pct >= 10 ? 1 : 2) + '%';
    }
    function levelOf(p) {
      if (p == null) return 'low';
      if (p >= 0.6) return 'critical';
      if (p >= 0.3) return 'high';
      if (p >= 0.1) return 'mid';
      return 'low';
    }
    function levelLabel(l) {
      return { low: 'MINIMAL', mid: 'MODERATE', high: 'HIGH', critical: 'CRITICAL' }[l] || '—';
    }
    // Use LOCAL time for chart labels so they match the user's clock
    function fmtLocal(iso) {
      if (!iso) return '';
      var d = new Date(iso);
      return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
    }
    function formatUTC(iso) {
      if (!iso) return '';
      var d = new Date(iso);
      return d.getUTCHours().toString().padStart(2,'0') + ':' + d.getUTCMinutes().toString().padStart(2,'0');
    }
    function colorFor(l) {
      return { low: '#22c55e', mid: '#f59e0b', high: '#ff4444', critical: '#ff6666' }[l] || '#22c55e';
    }

    var chartBacktest = null, chartNext15 = null;

    function updateCharts(data) {
      var probs15 = (data.next_15_probs || []).map(Number);

      // --- Strike timeline: last 20 strikes + "when is the next one?" forecast ---
      if (data.backtest_5h && data.backtest_5h.times && data.backtest_5h.times.length) {
        var times = data.backtest_5h.times;
        var actData = (data.backtest_5h.actual || []).map(Number);

        // Find indices of last 20 actual strikes
        var strikeIdxs = [];
        for (var i = actData.length - 1; i >= 0 && strikeIdxs.length < 20; i--) {
          if (actData[i] === 1) strikeIdxs.unshift(i);
        }
        // Slice history from a bit before the first of those strikes
        var histStart = strikeIdxs.length > 0 ? Math.max(0, strikeIdxs[0] - 10) : Math.max(0, times.length - 60);
        var histTimes  = times.slice(histStart);
        var histActual = actData.slice(histStart);

        // Future timestamps (+1m … +15m from NOW)
        var lastTimeMs = new Date(times[times.length - 1]).getTime();
        var futureTimes = probs15.map(function(_, i) {
          return new Date(lastTimeMs + (i + 1) * 60000).toISOString();
        });

        // All times for x-axis
        var allTimes = histTimes.concat(futureTimes);
        var nowIdx   = histTimes.length - 1; // index of "NOW" in allTimes

        // Dataset 1: strike markers (tall bar = 1 at each past strike, null elsewhere)
        var strikeBars = histActual.map(function(v) { return v === 1 ? 1.0 : null; })
                                   .concat(futureTimes.map(function() { return null; }));

        // Dataset 2: cumulative P(next strike has happened BY minute k after now)
        // Starts at 0 right after "now", rises to the 15-min card value by +15m
        var cumFore = [], cprod2 = 1.0;
        for (var i = 0; i < probs15.length; i++) {
          cprod2 *= Math.max(0, 1 - probs15[i]);
          cumFore.push(1 - cprod2);
        }
        var cdfData = histTimes.map(function() { return null; }).concat(cumFore);

        // Labels: absolute time in history, "NOW ▶" at boundary, +Nm in future
        var labels = allTimes.map(function(t, i) {
          if (i === nowIdx) return 'NOW';
          if (i > nowIdx)   return '+' + (i - nowIdx) + 'm';
          var minsFromNow = nowIdx - i;
          return (minsFromNow % 15 === 0) ? fmtLocal(t) : '';
        });

        // How many strikes are in this window?
        var nStrikes = histActual.reduce(function(s, v) { return s + v; }, 0);
        document.getElementById('chart5hEnd').textContent =
          nStrikes + ' strikes shown  ·  orange line = cumulative P(next strike has occurred by that minute)';

        if (chartBacktest) chartBacktest.destroy();
        chartBacktest = new Chart(document.getElementById('chartBacktest'), {
          type: 'bar',
          data: {
            labels: labels,
            datasets: [
              {
                type: 'bar',
                label: 'Strike',
                data: strikeBars,
                backgroundColor: 'rgba(34,197,94,0.85)',
                borderColor: 'rgba(34,197,94,1)',
                borderWidth: 0,
                borderRadius: 2,
                categoryPercentage: 1.0,
                barPercentage: 1.0,
                yAxisID: 'y'
              },
              {
                type: 'line',
                label: 'P(next strike by this minute)',
                data: cdfData,
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245,158,11,0.10)',
                borderWidth: 2.5,
                borderDash: [5, 4],
                fill: true,
                tension: 0.3,
                pointRadius: 0,
                yAxisID: 'y',
                spanGaps: false
              }
            ]
          },
          options: {
            responsive: true, maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { labels: { color: '#888', font: { size: 11 }, boxWidth: 14, padding: 14 } },
              tooltip: {
                backgroundColor: '#1e1e28', titleColor: '#aaa', bodyColor: '#ddd',
                borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
                filter: function(item) { return item.parsed.y !== null && item.parsed.y > 0; },
                callbacks: {
                  title: function(items) {
                    var idx = items[0].dataIndex;
                    if (idx === nowIdx) return 'NOW';
                    if (idx > nowIdx)   return 'Forecast +' + (idx - nowIdx) + ' min';
                    return fmtLocal(allTimes[idx]);
                  },
                  label: function(ctx) {
                    if (ctx.parsed.y === null) return null;
                    if (ctx.datasetIndex === 0) return 'Strike occurred';
                    return 'P(next strike by +' + (ctx.dataIndex - nowIdx) + 'm): ' + (ctx.parsed.y * 100).toFixed(1) + '%';
                  }
                }
              }
            },
            scales: {
              x: { ticks: { color: TICK, maxRotation: 0, font: { size: 10 } }, grid: { color: GRID } },
              y: { type: 'linear', min: 0, max: 1,
                   ticks: { color: TICK, font: { size: 10 }, callback: function(v) {
                     return v === 1 ? 'Strike' : (v * 100).toFixed(0) + '%';
                   }},
                   grid: { color: GRID } }
            }
          }
        });
      }

      // --- Next 15 min chart: per-minute bars + cumulative line ---
      if (probs15.length > 0) {
        // Cumulative: P(at least one strike by minute k) = 1 - product(1-p[0..k])
        var cumulative = [], prod = 1.0;
        for (var i = 0; i < probs15.length; i++) {
          prod *= Math.max(0, 1 - probs15[i]);
          cumulative.push(1 - prod);
        }
        var maxBar = Math.max.apply(null, probs15);
        var yMax = Math.max(0.05, Math.max(maxBar, cumulative[cumulative.length-1]) * 1.05);
        var barColors = probs15.map(function(p) {
          return colorFor(levelOf(p)) + 'bb';
        });
        var lbls = Array.from({length: probs15.length}, function(_, i) { return '+' + (i+1) + 'm'; });

        if (chartNext15) chartNext15.destroy();
        chartNext15 = new Chart(document.getElementById('chartNext15'), {
          type: 'bar',
          data: {
            labels: lbls,
            datasets: [
              {
                type: 'bar',
                label: 'P(strike in this min)',
                data: probs15,
                backgroundColor: barColors,
                borderColor: 'transparent',
                borderRadius: 3,
                borderWidth: 0,
                yAxisID: 'y'
              },
              {
                type: 'line',
                label: 'Cumulative P(≥1 strike by this min)',
                data: cumulative,
                borderColor: '#f59e0b',
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 3,
                pointBackgroundColor: '#f59e0b',
                tension: 0.3,
                yAxisID: 'y'
              }
            ]
          },
          options: {
            responsive: true, maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { display: true, labels: { color: '#888', font: { size: 10 }, boxWidth: 12, padding: 12 } },
              tooltip: {
                backgroundColor: '#1e1e28', titleColor: '#aaa', bodyColor: '#ddd',
                borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
                callbacks: {
                  label: function(ctx) {
                    return ctx.dataset.label + ': ' + (ctx.parsed.y * 100).toFixed(1) + '%';
                  }
                }
              }
            },
            scales: {
              x: { ticks: { color: TICK, font: { size: 10 } }, grid: { color: GRID } },
              y: { min: 0, max: yMax,
                   ticks: { color: TICK, font: { size: 10 }, callback: function(v) { return (v*100).toFixed(0)+'%'; } },
                   grid: { color: GRID } }
            }
          }
        });
      }
    }

    function updateUI(data) {
      var errBox = document.getElementById('errBox');
      if (!data.ok) {
        errBox.textContent = data.error || 'Error loading data.';
        errBox.style.display = 'block';
        return;
      }
      errBox.style.display = 'none';

      // Probabilities
      var vals = { p5: data.p_5, p15: data.p_15, p60: data.p_60 };
      var ids  = { p5: ['p5','card5'], p15: ['p15','card15'], p60: ['p60','card60'] };
      Object.keys(vals).forEach(function(k) {
        var p = vals[k];
        var l = levelOf(p);
        document.getElementById(ids[k][0]).textContent = fmtP(p);
        document.getElementById(ids[k][1]).className = 'card ' + l;
      });

      // Threat level bar (based on 15 min)
      var lv = levelOf(data.p_15);
      var pct = Math.min(100, (data.p_15 || 0) * 100 / 0.6 * 100);  // 60% = full bar
      pct = Math.max(2, Math.min(100, (data.p_15 || 0) * 166.7));
      var fill = document.getElementById('threatFill');
      var badge = document.getElementById('threatBadge');
      fill.style.width = pct.toFixed(1) + '%';
      fill.style.background = colorFor(lv);
      badge.textContent = levelLabel(lv);
      badge.className = 'threat-badge ' + lv;

      // Next alarm banner
      var banner = document.getElementById('nextAlarmBanner');
      var alarmMin = data.next_alarm_min;
      if (alarmMin != null) {
        var alarmTime = new Date(new Date().getTime() + alarmMin * 60000);
        var hh = alarmTime.getHours().toString().padStart(2,'0');
        var mm = alarmTime.getMinutes().toString().padStart(2,'0');
        banner.textContent = '⚠ 50% chance of next strike reached at +' + alarmMin + ' min  (' + hh + ':' + mm + ' local)';
        banner.style.display = 'block';
      } else {
        banner.style.display = 'none';
      }

      // Status
      var updatedAt = (data.updated_at || '').replace('T', ' ').slice(0, 19);
      document.getElementById('updated').textContent = 'Updated ' + updatedAt + ' UTC';
      var next = new Date();
      next.setMinutes(next.getMinutes() + (5 - next.getMinutes() % 5));
      next.setSeconds(0, 0);
      document.getElementById('nextRefresh').textContent = 'Next refresh ~' + next.toLocaleTimeString();

      updateCharts(data);
    }

    function fetchProbs() {
      fetch('/api/probs').then(function(r) { return r.json(); }).then(updateUI).catch(function(e) {
        updateUI({ ok: false, error: e.message });
      });
    }
    fetchProbs();
    setInterval(fetchProbs, 30000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    use_static = os.environ.get("USE_STATIC_STATE", "").strip().lower() in ("1", "true", "yes")
    # On Render (512MB), never train – require pretrained model or static state to avoid OOM
    pretrained_only = os.environ.get("REQUIRE_PRETRAINED", "").strip().lower() in ("1", "true", "yes")
    if pretrained_only and not (use_static and LIVE_STATE_FILE.exists()) and not (MODEL_FILE.exists() and joblib):
        print("ERROR: REQUIRE_PRETRAINED is set but no model or static state found.", flush=True)
        print("  Run locally: python export_live_state.py", flush=True)
        print("  Then commit and push: data_cache/model.joblib (and optionally data_cache/live_state.json)", flush=True)
        print("  Ensure data_cache/ or model.joblib is not in .gitignore.", flush=True)
        sys.exit(1)
    if use_static and LIVE_STATE_FILE.exists():
        print("Rocket strike web app – loading pre-computed state (no training)...", flush=True)
        if _load_static_state():
            print("  Loaded live_state.json. Serving graphs only.", flush=True)
        else:
            print("  Failed to load live_state.json.", _state.get("error"), flush=True)
    elif MODEL_FILE.exists() and joblib is not None:
        # Load pretrained model; fetch recent data and update live every 5 min
        _REFRESH_MAX_ROWS = 120000
        _KEEP_LAST_MINUTES = 30 * 24 * 60  # 30 days so feature distribution matches full-data-trained model (avoids zero preds)
        print("Rocket strike web app – loading pretrained model (no training)...", flush=True)
        if _load_pretrained_model():
            print("  Model loaded. Fetching recent data (30d window) and updating every 5 min.", flush=True)
            _refresh_data_only(max_rows=_REFRESH_MAX_ROWS, keep_last_minutes=_KEEP_LAST_MINUTES)
            t = threading.Thread(target=_background_refresh, daemon=True, kwargs={"max_rows": _REFRESH_MAX_ROWS, "keep_last_minutes": _KEEP_LAST_MINUTES})
            t.start()
        else:
            if pretrained_only:
                print("  Failed to load model. Exiting (REQUIRE_PRETRAINED set).", flush=True)
                sys.exit(1)
            print("  Failed to load model. Falling back to training.", flush=True)
            _train_and_predict()
            if _state["ready"]:
                t = threading.Thread(target=_background_refresh, daemon=True)
                t.start()
    else:
        if pretrained_only:
            print("ERROR: No model.joblib found. Run export_live_state.py and push data_cache/model.joblib.", flush=True)
            sys.exit(1)
        print("Rocket strike web app – loading data and training model (may take a minute)...", flush=True)
        _train_and_predict()
        if not _state["ready"]:
            print("Warning: model not ready.", _state.get("error"), flush=True)
        else:
            print("Model ready. Starting server and 5-minute data refresh.", flush=True)
        t = threading.Thread(target=_background_refresh, daemon=True)
        t.start()
    app = create_app()
    print(f"  Listening on 0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
