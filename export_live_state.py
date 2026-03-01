"""
Train locally and export model + live_state for the web app.
Uses a 30-day window so the model sees recent alarms; excludes the last 60 minutes
from training (they become the held-out evaluation window).

Run locally:
  python export_live_state.py

Then commit data_cache/model.joblib and push. On Render, set REQUIRE_PRETRAINED=1.
"""

import json
import os
import sys
from pathlib import Path

# Import app module so we use its _train_and_predict and _state
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["TQDM_DISABLE"] = "1"

from rocket_strike_app import (
    _train_and_predict,
    _state,
    _lock,
    LIVE_STATE_FILE,
    MODEL_FILE,
    CACHE_DIR,
    joblib,
)

# Load last 120k CSV rows; keep 30-day minute window so the model sees recent active periods.
# If there are alarms in the last 30 days, trains on everything up to 60 min ago
# (last 60 min held out as evaluation window). Falls back to 7-day test split if quiet.
APP_WINDOW_ROWS = 120000
KEEP_LAST_MINUTES = 30 * 24 * 60  # 30 days

def main():
    print(f"Training on last {APP_WINDOW_ROWS} CSV rows ({KEEP_LAST_MINUTES // 1440}d window; "
          f"last 60 min excluded if recent alarms)...", flush=True)
    _train_and_predict(max_rows=APP_WINDOW_ROWS, keep_last_minutes=KEEP_LAST_MINUTES, train_on_all=False)
    with _lock:
        err = _state.get("error") or ""
        need_fallback = not _state["ready"] and "no positives" in err.lower()
    if need_fallback:
        print("  No strikes in recent window; retrying with full data...", flush=True)
        _train_and_predict(max_rows=None, train_on_all=False)
    with _lock:
        if not _state["ready"]:
            print("Error:", _state.get("error"), flush=True)
            return 1
        model, scaler = _state["model"], _state["scaler"]
        out = {
            "probs": {
                "1": _state["probs"]["1"],
                "5": _state["probs"][5],
                "15": _state["probs"][15],
                "60": _state["probs"][60],
            },
            "backtest_5h": _state.get("backtest_5h"),
            "next_15_probs": _state.get("next_15_probs"),
            "updated_at": _state.get("updated_at", ""),
        }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if joblib is not None and model is not None and scaler is not None:
        joblib.dump((model, scaler), MODEL_FILE)
        print(f"Wrote {MODEL_FILE} (pretrained model; server will load it and update from live data)", flush=True)
    with open(LIVE_STATE_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {LIVE_STATE_FILE}", flush=True)
    print("Next: commit and push data_cache/model.joblib (and optionally live_state.json).", flush=True)
    print("On Render: deploy without USE_STATIC_STATE; app will load model and fetch recent data every 5 min.", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
