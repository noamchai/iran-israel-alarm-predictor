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
            out["next_15_probs"] = _state.get("next_15_probs") or []
            return jsonify(out)

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Israel Strike Risk Monitor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0a0a0f;
      --surface: #111118;
      --card: #16161e;
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
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    /* Header */
    .header {
      background: linear-gradient(135deg, #1a0a0a 0%, #0f0f18 60%, #0a0a0f 100%);
      border-bottom: 1px solid rgba(255,60,60,0.15);
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
      <h1>Israel Strike Risk Monitor <span>&#x26A0;</span></h1>
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

    <!-- Backtest chart -->
    <div class="section-header">
      <h2>Last 5 hours</h2>
      <span class="section-note" id="chart5hEnd">Actual strikes vs predicted probability</span>
    </div>
    <div class="chart-card"><canvas id="chartBacktest"></canvas></div>

    <!-- Next 15 min chart -->
    <div class="section-header">
      <h2>Next 15 minutes</h2>
      <span class="section-note">P(strike in that single minute) &mdash; bars sum into the 15 min probability above</span>
    </div>
    <div class="chart-card"><canvas id="chartNext15"></canvas></div>
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
      // --- Backtest chart ---
      if (data.backtest_5h && data.backtest_5h.times && data.backtest_5h.times.length) {
        var times = data.backtest_5h.times;
        var lastDate = new Date(times[times.length - 1]);
        var endStr = 'Through ' + formatUTC(times[times.length - 1]) + ' UTC';
        document.getElementById('chart5hEnd').textContent = endStr;

        var step = 30;
        var labels = times.map(function(t, i) {
          return (i % step === 0 || i === times.length - 1) ? formatUTC(t) : '';
        });
        var predData = (data.backtest_5h.pred || []).map(function(p) { return Math.max(0, Math.min(1, Number(p))); });
        var actData = data.backtest_5h.actual || [];

        if (chartBacktest) chartBacktest.destroy();
        chartBacktest = new Chart(document.getElementById('chartBacktest'), {
          type: 'line',
          data: {
            labels: labels,
            datasets: [
              {
                label: 'Actual strike',
                data: actData,
                borderColor: 'rgba(34,197,94,0.7)',
                backgroundColor: 'rgba(34,197,94,0.08)',
                fill: true, yAxisID: 'y0', tension: 0, pointRadius: 0, borderWidth: 1.5
              },
              {
                label: 'Predicted P(strike)',
                data: predData,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.07)',
                fill: true, yAxisID: 'y1', tension: 0.25, pointRadius: 0, borderWidth: 2
              }
            ]
          },
          options: {
            responsive: true, maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: { labels: { color: '#888', font: { size: 11 }, boxWidth: 14, padding: 14 } },
              tooltip: { backgroundColor: '#1e1e28', titleColor: '#aaa', bodyColor: '#ddd', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 }
            },
            scales: {
              x: { ticks: { color: TICK, maxRotation: 0, font: { size: 10 } }, grid: { color: GRID } },
              y0: { type: 'linear', position: 'left', min: -0.05, max: 1.05,
                    ticks: { color: TICK, stepSize: 0.5, font: { size: 10 } }, grid: { color: GRID } },
              y1: { type: 'linear', position: 'right', min: 0, max: 1,
                    ticks: { color: TICK, font: { size: 10 }, callback: function(v) { return (v*100).toFixed(0)+'%'; } },
                    grid: { drawOnChartArea: false } }
            }
          }
        });
      }

      // --- Next 15 min chart ---
      var probs15 = data.next_15_probs;
      if (probs15 && probs15.length > 0) {
        var values = probs15.map(Number);
        var maxP = Math.max.apply(null, values);
        var yMax = Math.max(0.005, maxP * 1.25);
        var barColors = values.map(function(p) {
          var l = levelOf(p);
          return colorFor(l) + 'cc';
        });

        if (chartNext15) chartNext15.destroy();
        chartNext15 = new Chart(document.getElementById('chartNext15'), {
          type: 'bar',
          data: {
            labels: Array.from({length: values.length}, function(_, i) { return i + 1; }),
            datasets: [{
              label: 'P(strike this minute)',
              data: values,
              backgroundColor: barColors,
              borderColor: 'transparent',
              borderRadius: 4,
              borderWidth: 0
            }]
          },
          options: {
            responsive: true, maintainAspectRatio: true,
            plugins: {
              legend: { display: false },
              tooltip: { backgroundColor: '#1e1e28', titleColor: '#aaa', bodyColor: '#ddd',
                         borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1,
                         callbacks: { label: function(ctx) { return 'P = ' + (ctx.parsed.y*100).toFixed(2)+'%'; } } }
            },
            scales: {
              x: { title: { display: true, text: 'Minutes from now', color: '#666', font: { size: 11 } },
                   ticks: { color: TICK, font: { size: 10 } }, grid: { color: GRID } },
              y: { min: 0, max: yMax,
                   ticks: { color: TICK, font: { size: 10 }, callback: function(v) { return (v*100).toFixed(1)+'%'; } },
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
