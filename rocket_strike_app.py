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
    MAX_MINUTE_ROWS,
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


def _train_and_predict():
    """Build timeline, train 1-min model, compute current probs. Uses true data only."""
    try:
        timeline = build_minute_timeline()
        if timeline is None or len(timeline) < 1000:
            with _lock:
                _state["error"] = "Not enough minute-level data (need GitHub/Kaggle timeline)."
                _state["ready"] = False
            return
        if len(timeline) > MAX_MINUTE_ROWS:
            timeline = timeline.iloc[-MAX_MINUTE_ROWS:].reset_index(drop=True)
        df = hazard_features_minute(timeline)
        train_df, test_df = train_test_split_by_minutes(df, test_minutes=7 * 24 * 60)
        if len(test_df) < 2:
            n = len(df) - 1
            split = int(0.8 * n)
            train_df = df.iloc[: split + 1]
            test_df = df.iloc[split + 1 :]
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


def _refresh_data_only():
    """Re-fetch timeline and features, recompute probs with existing model (no retrain)."""
    with _lock:
        model, scaler = _state.get("model"), _state.get("scaler")
    if model is None or scaler is None:
        _train_and_predict()
        return
    try:
        timeline = build_minute_timeline()
        if timeline is None or len(timeline) < 100:
            return
        if len(timeline) > MAX_MINUTE_ROWS:
            timeline = timeline.iloc[-MAX_MINUTE_ROWS:].reset_index(drop=True)
        df = hazard_features_minute(timeline)
        last_idx = len(df) - 1
        last_X = df.iloc[last_idx][FEATURE_COLS_MINUTE].values.astype(float).reshape(1, -1)
        p_1 = float(predict_proba_strike(model, scaler, last_X)[0])
        probs = _derive_horizon_probs_from_current(p_1, WEB_HORIZONS)
        probs["1"] = p_1
        backtest_5h, next_15_probs = _compute_backtest_and_next_15(df, model, scaler)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with _lock:
            _state["probs"] = probs
            _state["updated_at"] = now_iso
            _state["backtest_5h"] = backtest_5h
            _state["next_15_probs"] = next_15_probs
            _state["error"] = None
    except Exception as e:
        with _lock:
            _state["error"] = str(e)


def _background_refresh():
    """Run data refresh every 5 minutes."""
    while True:
        time.sleep(300)  # 5 minutes
        _refresh_data_only()


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
                    "error": _state.get("error") or "Model not ready yet. Wait for first training.",
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
                out["backtest_5h"] = _state["backtest_5h"]
            if _state.get("next_15_probs"):
                out["next_15_probs"] = _state["next_15_probs"]
            return jsonify(out)

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IRAN vs ISRAEL round 2 alarm predictor</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root { --bg: #0f0f12; --card: #1a1a20; --text: #e8e6e3; --muted: #888; --accent: #e74c3c; --ok: #2ecc71; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; min-height: 100vh; padding: 1.5rem; }
    h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 0.5rem 0; }
    h2 { font-size: 1rem; font-weight: 600; margin: 1.5rem 0 0.5rem 0; color: var(--muted); }
    .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
    .card { background: var(--card); border-radius: 12px; padding: 1.25rem; text-align: center; border: 1px solid rgba(255,255,255,0.06); }
    .card .label { font-size: 0.85rem; color: var(--muted); margin-bottom: 0.35rem; }
    .card .value { font-size: 1.75rem; font-weight: 700; }
    .card .value.high { color: var(--accent); }
    .card .value.mid { color: #f39c12; }
    .card .value.low { color: var(--ok); }
    .meta { color: var(--muted); font-size: 0.8rem; }
    .meta .live { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--ok); margin-right: 6px; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{ opacity:1 } 50%{ opacity:0.5 } }
    .err { color: var(--accent); margin-top: 1rem; }
    #next { margin-top: 1rem; font-size: 0.85rem; color: var(--muted); }
    .chart-wrap { background: var(--card); border-radius: 12px; padding: 1rem; margin-top: 0.5rem; border: 1px solid rgba(255,255,255,0.06); max-width: 900px; height: 220px; }
    .chart-wrap canvas { max-height: 220px; }
  </style>
</head>
<body>
  <h1>IRAN vs ISRAEL round 2 alarm predictor</h1>
  <p class="sub">P(strike) in the next 5, 15, 60 minutes (from true past data). Data refreshed every 5 minutes.</p>
  <div class="cards">
    <div class="card">
      <div class="label">Next 5 min</div>
      <div class="value" id="p5">—</div>
    </div>
    <div class="card">
      <div class="label">Next 15 min</div>
      <div class="value" id="p15">—</div>
    </div>
    <div class="card">
      <div class="label">Next 60 min</div>
      <div class="value" id="p60">—</div>
    </div>
  </div>
  <p class="meta"><span class="live"></span> <span id="updated">Waiting for data…</span></p>
  <p id="next" class="meta"></p>
  <p id="err" class="err" style="display:none"></p>

  <h2>Last 5 hours: data vs prediction</h2>
  <div class="chart-wrap">
    <canvas id="chartBacktest"></canvas>
  </div>
  <h2>Next 15 minutes: P(strike) per minute</h2>
  <div class="chart-wrap">
    <canvas id="chartNext15"></canvas>
  </div>

  <script>
    function fmtP(p) {
      if (p == null) return '—';
      return (p * 100).toFixed(2) + '%';
    }
    var chartBacktest = null, chartNext15 = null;
    function formatTime(iso) {
      if (!iso) return '';
      var d = new Date(iso);
      return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
    }
    function updateCharts(data) {
      if (data.backtest_5h && data.backtest_5h.times && data.backtest_5h.times.length) {
        var times = data.backtest_5h.times;
        var step = Math.max(1, Math.floor(times.length / 6));
        var labels = times.map(function(_, i) {
          return (i % step === 0 || i === times.length - 1) ? formatTime(times[i]) : '';
        });
        if (chartBacktest) chartBacktest.destroy();
        var predData = (data.backtest_5h.pred || []).map(function(p) { return Math.max(0, Math.min(1, Number(p))); });
        chartBacktest = new Chart(document.getElementById('chartBacktest'), {
          type: 'line',
          data: {
            labels: labels,
            datasets: [
              { label: 'Actual strike', data: data.backtest_5h.actual, borderColor: '#2ecc71', backgroundColor: 'rgba(46,204,113,0.2)', fill: true, yAxisID: 'y0', tension: 0, pointRadius: 0, borderWidth: 1 },
              { label: 'Predicted P(strike)', data: predData, borderColor: '#3498db', backgroundColor: 'rgba(52,152,219,0.1)', fill: true, yAxisID: 'y1', tension: 0.2, pointRadius: 0, borderWidth: 1.5 }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { labels: { color: '#e8e6e3' } } },
            scales: {
              x: { ticks: { color: '#888', maxRotation: 0 }, grid: { color: 'rgba(255,255,255,0.06)' } },
              y0: { type: 'linear', position: 'left', min: -0.05, max: 1, ticks: { color: '#888', stepSize: 0.25 }, grid: { color: 'rgba(255,255,255,0.06)' } },
              y1: { type: 'linear', position: 'right', min: 0, max: 1, ticks: { color: '#888', callback: function(v) { return (v*100).toFixed(0)+'%'; } }, grid: { drawOnChartArea: false } }
            }
          }
        });
      }
      var probs15 = data.next_15_probs;
      if (probs15 && Array.isArray(probs15) && probs15.length > 0) {
        var values = probs15.map(Number);
        var maxP = Math.max.apply(null, values);
        var yMax = Math.max(0.1, maxP * 1.2);
        if (chartNext15) chartNext15.destroy();
        chartNext15 = new Chart(document.getElementById('chartNext15'), {
          type: 'bar',
          data: {
            labels: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15].slice(0, values.length),
            datasets: [{ label: 'P(strike in this minute)', data: values, backgroundColor: 'rgba(52,152,219,0.8)', borderColor: '#3498db', borderWidth: 1 }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false } },
            scales: {
              x: { title: { display: true, text: 'Minutes from now', color: '#888' }, ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.06)' } },
              y: { min: 0, max: yMax, ticks: { color: '#888', callback: function(v) { return (v*100).toFixed(1)+'%'; } }, grid: { color: 'rgba(255,255,255,0.06)' } }
            }
          }
        });
      }
    }
    function updateUI(data) {
      if (!data.ok) {
        document.getElementById('err').textContent = data.error || 'Error';
        document.getElementById('err').style.display = 'block';
        return;
      }
      document.getElementById('err').style.display = 'none';
      var p5 = document.getElementById('p5'), p15 = document.getElementById('p15'), p60 = document.getElementById('p60');
      p5.textContent = fmtP(data.p_5); p15.textContent = fmtP(data.p_15); p60.textContent = fmtP(data.p_60);
      p5.className = 'value ' + (data.p_5 >= 0.5 ? 'high' : data.p_5 >= 0.2 ? 'mid' : 'low');
      p15.className = 'value ' + (data.p_15 >= 0.5 ? 'high' : data.p_15 >= 0.2 ? 'mid' : 'low');
      p60.className = 'value ' + (data.p_60 >= 0.5 ? 'high' : data.p_60 >= 0.2 ? 'mid' : 'low');
      document.getElementById('updated').textContent = 'Updated: ' + (data.updated_at || '').replace('T', ' ').slice(0, 19) + ' UTC';
      var next = new Date();
      next.setMinutes(next.getMinutes() + (5 - next.getMinutes() % 5));
      next.setSeconds(0, 0);
      document.getElementById('next').textContent = 'Next data refresh: ~' + next.toLocaleTimeString();
      updateCharts(data);
    }
    function fetchProbs() {
      fetch('/api/probs').then(function(r) { return r.json(); }).then(updateUI).catch(function(e) {
        updateUI({ ok: false, error: e.message });
      });
    }
    fetchProbs();
    setInterval(fetchProbs, 30 * 1000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    # One-time train on startup (then background only refreshes data every 5 min)
    print("Rocket strike web app – loading data and training model (may take a minute)...", flush=True)
    _train_and_predict()
    if not _state["ready"]:
        print("Warning: model not ready.", _state.get("error"), flush=True)
    else:
        print("Model ready. Starting server and 5-minute data refresh.", flush=True)
    t = threading.Thread(target=_background_refresh, daemon=True)
    t.start()
    app = create_app()
    print(f"  Open http://127.0.0.1:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
