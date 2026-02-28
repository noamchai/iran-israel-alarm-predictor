#!/usr/bin/env python3
"""
Live prediction dashboard server.

Trains the returns predictor on historical data at startup, then streams
live 1-minute prices via WebSocket and makes hourly predictions.

Usage:
    python live_server.py [--port 5000] [--source auto] [--years 1.0]
"""
from __future__ import annotations

import argparse
import json
import pickle
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from data_loader import fetch_stock_prices
from correlation_and_predict import (
    split_into_slices,
    split_into_overlapping_slices,
    slice_averaged_covariances_returns,
    predict_next_returns,
    fit_var_returns,
    predict_next_returns_arp,
    prices_to_returns,
)
from nn_predictor import (
    train_factor_transformer,
    predict_factor_transformer,
    save_factor_transformer,
    load_factor_transformer,
)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V", "JNJ", "WMT",
    "PG", "MA", "HD", "DIS", "BAC", "ADBE", "XOM", "NFLX", "CRM", "PEP",
    "KO", "COST", "AVGO", "ORCL", "MCD", "CSCO", "ACN", "ABT", "NEE", "TMO",
    "DHR", "LIN", "WFC", "INTC", "AMD", "QCOM", "TXN", "UNP", "HON", "UPS",
    "RTX", "LOW", "CAT", "DE", "GE", "IBM", "AMAT", "INTU", "NOW", "PLD",
]

USE_HOURLY = True    # train & test on hourly bars; 1 point = 1 hour
PREDICTION_LAG = 1   # 1 bar (1 hour) ahead
# Test = last 10 days (240h); FT needs slice >= W+H+1 → 265 for W=24, H=240
WINDOW_LENGTH = 265  # bars (hours): ~11 days test, enough for FT W=24 + predict 10 days
WINDOW_STEP = 1
AR_ORDER = 10
NN_WINDOW = 24       # 1 day of context (24 hours)
NN_HORIZON = 24      # FT predicts 1 day (24h); we propagate 10× for 10-day extension
FT_K = 10
REG = 1e-5
CACHE_DIR = Path("data_cache")
MODELS_DIR = CACHE_DIR / "models"
# Buffer: need enough minutes to build hourly bars when USE_HOURLY (25h = 1500 min)
PRICE_BUFFER_MAXLEN = 1600 if USE_HOURLY else 480

app = Flask(__name__)
app.config["SECRET_KEY"] = "financeexp-live"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── Shared state ─────────────────────────────────────────────────────────────

state = {
    "status": "initializing",        # initializing | training | ready | error
    "error": None,
    "tickers": [],
    "mu_ret": None,                  # (N,) mean return
    "cov_ret_0": None,               # (N,N)
    "cov_ret_all": None,             # list of (N,N) for lags 0..max_lag
    "var_model": None,               # VAR(p) model dict from fit_var_returns
    "ft_model": None,                # Factor Transformer dict
    "actual_lag": 0,                 # actual max lag used
    "price_buffer": deque(maxlen=PRICE_BUFFER_MAXLEN),
    "latest_prices": {},             # {ticker: price}
    "prev_prices": {},               # previous minute's prices for return calc
    "active_prediction": None,       # {timestamp_ms, trajectory: {ticker: [p1..p60]}, ...}
    "prediction_history": deque(maxlen=200),
    "train_info": {},                # metadata about training
    "backtest": None,                # {timestamps, actual, predicted} from test slice
    "source": "auto",
    "years": 1.0,
}
state_lock = threading.Lock()


def _ts_ms(dt=None):
    """Current UTC timestamp in milliseconds."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    return int(dt.timestamp() * 1000)


def _is_market_open():
    """Rough check: NYSE 9:30-16:00 ET, Mon-Fri."""
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    t = now.time()
    from datetime import time as dtime
    return dtime(9, 30) <= t <= dtime(16, 0)


# ── Data cache ────────────────────────────────────────────────────────────────

def _cache_path(interval: str) -> Path:
    return CACHE_DIR / f"prices_{interval}.csv"


def _save_cache(prices: pd.DataFrame, interval: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(interval)
    prices.to_csv(path)
    print(f"  Cached {len(prices)} bars to {path}")


def _load_cache(interval: str) -> pd.DataFrame | None:
    path = _cache_path(interval)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  Loaded {len(df)} bars from cache ({path})")
        return df
    except Exception as e:
        print(f"  Cache load failed: {e}")
        return None


# ── Model persistence ────────────────────────────────────────────────────────

def _model_config_key(tickers_list=None):
    """Config dict used to decide if saved models are still valid."""
    return {
        "use_hourly": USE_HOURLY,
        "window_length": WINDOW_LENGTH,
        "window_step": WINDOW_STEP,
        "ar_order": AR_ORDER,
        "nn_window": NN_WINDOW,
        "nn_horizon": NN_HORIZON,
        "ft_k": FT_K,
        "reg": REG,
        "tickers": sorted(tickers_list or TICKERS),
    }


def _save_models(var_model, ft_model, valid_tickers):
    """Save VAR and FT models to MODELS_DIR."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config = _model_config_key(valid_tickers)
    config_path = MODELS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=0)
    with open(MODELS_DIR / "var_model.pkl", "wb") as f:
        pickle.dump(var_model, f)
    save_factor_transformer(ft_model, MODELS_DIR / "ft")
    print(f"  Models saved to {MODELS_DIR}")


def _load_models(require_config_match=True):
    """
    Load VAR and FT from MODELS_DIR. If require_config_match, return None when config differs.
    Returns (var_model, ft_model, config) or None.
    """
    config_path = MODELS_DIR / "config.json"
    var_path = MODELS_DIR / "var_model.pkl"
    ft_dir = MODELS_DIR / "ft"
    if not config_path.exists() or not var_path.exists() or not (ft_dir / "ft_state.pt").exists():
        return None
    with open(config_path) as f:
        saved_config = json.load(f)
    if require_config_match:
        current = _model_config_key()
        if saved_config != current:
            return None
    with open(var_path, "rb") as f:
        var_model = pickle.load(f)
    ft_model = load_factor_transformer(ft_dir)
    print(f"  Loaded saved models from {MODELS_DIR} (config match)")
    return (var_model, ft_model, saved_config)


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(tickers, years, source, interval="1m", fetch_fresh=False, force_retrain=False):
    """Fetch historical data (or load from cache). Train or load models, then run backtest."""
    with state_lock:
        state["status"] = "training"
    socketio.emit("status", {"status": "training"})

    try:
        prices = None
        if not fetch_fresh:
            prices = _load_cache(interval)

        if prices is None:
            print("  Fetching fresh data...")
            prices = fetch_stock_prices(tickers, years=years, interval=interval, source=source)
            _save_cache(prices, interval)
        else:
            print("  Using cached data (pass --fetch-fresh to re-download)")

        prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if USE_HOURLY and interval == "1m":
            prices = prices.resample("1h").last().dropna(how="all")
            interval = "1h"
            print(f"  Resampled to hourly: {len(prices)} bars")
        valid_tickers = list(prices.columns)
        if len(valid_tickers) < 2:
            raise ValueError("Need at least 2 stocks with valid data")

        n_rows = len(prices)
        max_lag = min(max(PREDICTION_LAG, AR_ORDER), WINDOW_LENGTH - 2)
        if max_lag < 1:
            max_lag = 1
        actual_lag = min(PREDICTION_LAG, max_lag)

        train_part = prices.iloc[:n_rows - WINDOW_LENGTH]
        test_slice = prices.iloc[n_rows - WINDOW_LENGTH:]
        train_slices = split_into_overlapping_slices(train_part, WINDOW_LENGTH, WINDOW_STEP)

        loaded = None
        if not force_retrain:
            loaded = _load_models(require_config_match=True)

        if loaded is not None:
            var_model, ft_model, _ = loaded
            mu_ret = var_model["mu_ret"]
            cov0 = np.dot(var_model["B"], var_model["B"].T)
            cov_ret = [cov0] * (max_lag + 1)
            print(f"  Using saved models (backtest on {len(test_slice)} bars)")
        else:
            if force_retrain:
                print("  Retrain requested (--retrain)")
            else:
                print("  No saved models or config changed; training from scratch.")
            print(f"  Training: {len(train_slices)} overlapping slices (window={WINDOW_LENGTH}, step={WINDOW_STEP}), {n_rows} total bars")
            mu_ret, cov_ret = slice_averaged_covariances_returns(train_slices, max_lag)
            print("Fitting VAR model (Yule-Walker solve)...")
            var_model = fit_var_returns(mu_ret, cov_ret, AR_ORDER, reg=REG)
            print(f"VAR({var_model['p']}) fitted, spectral radius = {var_model['spectral_radius']:.4f}")
            print(f"Training Factor Transformer (factor space, W={NN_WINDOW}, H=24)...")
            ft_model = train_factor_transformer(
                train_slices, W=NN_WINDOW, H=24, K=FT_K,
                use_price_space=False,
            )
            _save_models(var_model, ft_model, valid_tickers)

        print("Running backtest on held-out test slice...")

        # ── Backtest on held-out test slice ──
        test_df = test_slice
        P_test = test_df.values.astype(float)
        R_test = np.zeros_like(P_test)
        R_test[1:] = (P_test[1:] - P_test[:-1]) / np.where(P_test[:-1] != 0, P_test[:-1], np.nan)
        R_test = np.nan_to_num(R_test, nan=0.0, posinf=0.0, neginf=0.0)
        T_test, N = P_test.shape
        tau = actual_lag
        p_order = var_model["p"]
        start_i = max(1, p_order, NN_WINDOW + 1)  # need enough history for both VAR(p) and NN
        n_pred = T_test - tau - start_i

        bt_timestamps = []
        bt_actual = {t: [] for t in valid_tickers}
        bt_onestep = {t: [] for t in valid_tickers}
        bt_ft = {t: [] for t in valid_tickers}
        extended_timestamps = []
        extended_ft = {t: [] for t in valid_tickers}
        extended_hours = 0
        propagated_ft_timestamps = []
        propagated_ft = {t: [] for t in valid_tickers}

        # FT trajectory over test using ONLY train + predicted data (no true test data)
        if ft_model and T_test > 0 and len(train_part) >= NN_WINDOW + 1:
            P_train = np.asarray(train_part.values, dtype=float)
            R_train = np.zeros_like(P_train)
            R_train[1:] = np.where(P_train[:-1] != 0, (P_train[1:] - P_train[:-1]) / P_train[:-1], 0.0)
            R_train = np.nan_to_num(R_train, nan=0.0, posinf=0.0, neginf=0.0)
            P_curr = P_train[-1].copy()
            R_window = np.asarray(R_train[-NN_WINDOW:], dtype=float)
            for k in range(T_test):
                ts_k = test_df.index[k]
                ts_val = int(ts_k.timestamp()) if hasattr(ts_k, 'timestamp') else 0
                propagated_ft_timestamps.append(ts_val)
                traj = predict_factor_transformer(ft_model, R_window=R_window, P_t=P_curr)
                next_P = np.asarray(traj[0])
                next_P = np.nan_to_num(next_P, nan=P_curr, posinf=P_curr, neginf=P_curr)
                for j, tk in enumerate(valid_tickers):
                    propagated_ft[tk].append(round(float(next_P[j]), 4))
                r_new = np.where(P_curr != 0, (next_P - P_curr) / P_curr, 0.0)
                r_new = np.nan_to_num(r_new, nan=0.0, posinf=0.0, neginf=0.0)
                R_window = np.vstack([R_window[1:], r_new[np.newaxis, :]])
                P_curr = next_P

        if n_pred > 0:
            anchor_ts = test_df.index[start_i]
            anchor_val = int(anchor_ts.timestamp()) if hasattr(anchor_ts, 'timestamp') else 0
            bt_timestamps.append(anchor_val)
            for j, tk in enumerate(valid_tickers):
                p0 = round(float(P_test[start_i, j]), 4)
                bt_actual[tk].append(p0)
                bt_onestep[tk].append(p0)
                bt_ft[tk].append(p0)


            ft_traj_remaining = 0
            ft_traj_prices = None

            for i in range(start_i, start_i + n_pred):
                t_idx = i + tau
                ts_idx = test_df.index[t_idx]
                ts_val = int(ts_idx.timestamp()) if hasattr(ts_idx, 'timestamp') else 0
                bt_timestamps.append(ts_val)

                real_window = [R_test[i - k] for k in range(min(p_order, i + 1))]
                P_onestep = predict_next_returns_arp(
                    real_window, mu_ret, var_model, P_test[i]
                )

                if ft_traj_remaining <= 0 and i >= NN_WINDOW and (i + NN_HORIZON) <= T_test:
                    if ft_model.get("use_price_space"):
                        P_win = P_test[i - NN_WINDOW:i + 1]  # (W+1, N)
                        ft_traj_prices = predict_factor_transformer(
                            ft_model, P_window=P_win, P_t=P_test[i],
                        )
                    else:
                        R_win_chrono = R_test[i - NN_WINDOW:i]
                        ft_traj_prices = predict_factor_transformer(
                            ft_model, R_window=R_win_chrono, P_t=P_test[i],
                        )
                    ft_traj_remaining = NN_HORIZON
                    ft_step_idx = 0

                if ft_traj_prices is not None and ft_step_idx < NN_HORIZON:
                    P_ft = ft_traj_prices[ft_step_idx]
                    ft_step_idx += 1
                    ft_traj_remaining -= 1
                else:
                    P_ft = P_test[t_idx]

                for j, tk in enumerate(valid_tickers):
                    bt_actual[tk].append(round(float(P_test[t_idx, j]), 4))
                    bt_onestep[tk].append(round(float(P_onestep[j]), 4))
                    bt_ft[tk].append(round(float(P_ft[j]), 4))

            # Extended prediction: continue the Transformer (10 days = 240h when hourly)
            step_sec = 3600 if USE_HOURLY else 60
            ext_steps = 240 if USE_HOURLY else 60
            last_i = start_i + n_pred - 1
            last_ts = bt_timestamps[-1] if bt_timestamps else 0
            H_ft = ft_model.get("H", 1) if ft_model else 1
            if last_i >= NN_WINDOW and ft_model and ext_steps > 0:
                P_curr = P_test[last_i].copy().astype(float)
                R_window = np.asarray(R_test[last_i - NN_WINDOW : last_i], dtype=float)
                n_blocks = (ext_steps + H_ft - 1) // H_ft
                for block in range(n_blocks):
                    n_step = min(H_ft, ext_steps - block * H_ft)
                    if ft_model.get("use_price_space"):
                        P_win = np.vstack([P_test[last_i - NN_WINDOW : last_i], P_curr[np.newaxis, :]]) if block == 0 else np.vstack([R_window[-NN_WINDOW:], P_curr[np.newaxis, :]])
                        traj = predict_factor_transformer(ft_model, P_window=P_win, P_t=P_curr)
                    else:
                        traj = predict_factor_transformer(ft_model, R_window=R_window, P_t=P_curr)
                    traj = np.asarray(traj)
                    new_returns = []
                    prev = P_curr.copy()
                    for h in range(n_step):
                        next_P = np.nan_to_num(traj[h], nan=prev, posinf=prev, neginf=prev)
                        for j, tk in enumerate(valid_tickers):
                            extended_ft[tk].append(round(float(next_P[j]), 4))
                        extended_timestamps.append(last_ts + step_sec * (block * H_ft + h + 1))
                        r_new = np.where(prev != 0, (next_P - prev) / prev, 0.0)
                        r_new = np.nan_to_num(r_new, nan=0.0, posinf=0.0, neginf=0.0)
                        new_returns.append(r_new)
                        prev = next_P
                    new_returns = np.array(new_returns)
                    R_window = np.vstack([R_window[n_step:], new_returns])
                    P_curr = prev

                # Prepend last FT backtest point so extended line continues from Transformer (cyan) graph
                if bt_timestamps and extended_timestamps and all(extended_ft[tk] for tk in valid_tickers):
                    for tk in valid_tickers:
                        extended_ft[tk] = [bt_ft[tk][-1]] + list(extended_ft[tk])
                    extended_timestamps = [bt_timestamps[-1]] + list(extended_timestamps)
                extended_hours = len(extended_timestamps) if extended_timestamps else 0

            tk0 = valid_tickers[0]
            print(f"Backtest: {n_pred} steps, start_i={start_i}")
            print(f"  {tk0} anchor: {bt_actual[tk0][0]} (shared start)")
            for di in range(1, min(4, len(bt_actual[tk0]))):
                print(f"  {tk0} t={di}: actual={bt_actual[tk0][di]}, "
                      f"1step={bt_onestep[tk0][di]}, ft={bt_ft[tk0][di]}")

        with state_lock:
            state["tickers"] = valid_tickers
            state["mu_ret"] = mu_ret
            state["cov_ret_0"] = cov_ret[0]
            state["cov_ret_all"] = cov_ret
            state["var_model"] = var_model
            state["ft_model"] = ft_model
            state["actual_lag"] = actual_lag
            state["status"] = "ready"
            state["error"] = None
            state["train_info"] = {
                "n_tickers": len(valid_tickers),
                "n_bars": n_rows,
                "n_slices": len(train_slices),
                "window_length": WINDOW_LENGTH,
                "step": WINDOW_STEP,
                "lag": actual_lag,
                "ar_order": var_model["p"],
                "spectral_radius": round(var_model["spectral_radius"], 4),
                "ft_window": NN_WINDOW,
                "ft_horizon": NN_HORIZON,
                "ft_factors": FT_K,
                "interval": interval,
                "use_hourly": USE_HOURLY,
                "trained_at": _ts_ms(),
            }
            state["backtest"] = {
                "timestamps": bt_timestamps,
                "actual": bt_actual,
                "onestep": bt_onestep,
                "ft": bt_ft,
                "propagated_ft_timestamps": propagated_ft_timestamps,
                "propagated_ft": propagated_ft,
                "extended_timestamps": extended_timestamps,
                "extended_ft": extended_ft,
                "extended_hours": extended_hours,
            }
            last_row = prices.iloc[-1]
            state["latest_prices"] = {t: float(last_row[t]) for t in valid_tickers if pd.notna(last_row[t])}
            state["prev_prices"] = {}
            if len(prices) >= 2:
                prev_row = prices.iloc[-2]
                state["prev_prices"] = {t: float(prev_row[t]) for t in valid_tickers if pd.notna(prev_row[t])}

        print(f"Model ready: {len(valid_tickers)} tickers, lag={actual_lag}")
        socketio.emit("status", {"status": "ready", "tickers": valid_tickers})
        socketio.emit("train_info", state["train_info"])

        # Make an initial prediction immediately so the dashboard has something to show
        prediction = _make_prediction()
        if prediction:
            prediction["base_prices"] = dict(state["latest_prices"])
            with state_lock:
                state["active_prediction"] = prediction
            socketio.emit("new_prediction", prediction)
            print(f"Initial prediction made for {len(prediction['prices'])} stocks")

    except Exception as e:
        traceback.print_exc()
        with state_lock:
            state["status"] = "error"
            state["error"] = str(e)
        socketio.emit("status", {"status": "error", "error": str(e)})


# ── Live price fetching ──────────────────────────────────────────────────────

def _fetch_latest_prices(tickers):
    """Fetch the most recent 1m bar for all tickers via yfinance."""
    import yfinance as yf
    try:
        data = yf.download(tickers, period="1d", interval="1m",
                           group_by="ticker", auto_adjust=True,
                           progress=False, threads=True)
        if data is None or len(data) == 0:
            return None, None

        if isinstance(data.columns, pd.MultiIndex):
            result = {}
            for t in tickers:
                try:
                    sub = data[t] if t in data.columns.get_level_values(0) else None
                    if sub is not None and len(sub) > 0:
                        col = "Adj Close" if "Adj Close" in sub.columns else "Close"
                        val = sub[col].dropna().iloc[-1]
                        if pd.notna(val):
                            result[t] = float(val)
                except (KeyError, IndexError):
                    continue
            ts = data.index[-1]
        else:
            col = "Adj Close" if "Adj Close" in data.columns else "Close"
            val = data[col].dropna().iloc[-1]
            result = {tickers[0]: float(val)} if pd.notna(val) else {}
            ts = data.index[-1]

        ts_ms = _ts_ms(ts.to_pydatetime().replace(tzinfo=timezone.utc)
                        if ts.tzinfo is None else ts.to_pydatetime())
        return result, ts_ms
    except Exception as e:
        print(f"  Fetch error: {e}")
        return None, None


def _buffer_to_hourly_prices(buf, tickers):
    """Convert minute price buffer to DataFrame of hourly close prices. Returns (df, None) or (None, err)."""
    if len(buf) < 60:
        return None, "need at least 60 minutes"
    rows = []
    for b in buf:
        ts = b.get("timestamp_ms")
        if ts is None:
            continue
        prices = b.get("prices") or {}
        dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        row = {t: prices.get(t) for t in tickers}
        row["_dt"] = dt
        rows.append(row)
    if not rows:
        return None, "no rows"
    df = pd.DataFrame(rows)
    df = df.set_index("_dt")
    df.index = pd.DatetimeIndex(df.index)
    hourly = df.resample("1h").last().dropna(how="all")
    return hourly, None


def _make_prediction():
    """
    Predict price ahead using VAR(p) + Factor Transformer.
    When USE_HOURLY, builds hourly bars from minute buffer and predicts next hour.
    """
    with state_lock:
        if state["status"] != "ready":
            return None
        tickers = state["tickers"]
        mu_ret = state["mu_ret"]
        vm = state["var_model"]
        ft_mod = state["ft_model"]
        actual_lag = state["actual_lag"]
        latest = state["latest_prices"]
        prev = state["prev_prices"]
        buf = list(state["price_buffer"])
        use_hourly = state.get("train_info", {}).get("use_hourly", False)

    N = len(tickers)
    hourly_df = None
    P_curr_safe = None
    if len(latest) < N * 0.5 or len(prev) < N * 0.5:
        return None

    P_t = np.array([latest.get(t, np.nan) for t in tickers])
    P_prev = np.array([prev.get(t, np.nan) for t in tickers])
    mask = np.isfinite(P_t) & np.isfinite(P_prev) & (P_prev != 0)
    if mask.sum() < 2:
        return None

    P_t_safe = np.where(mask, P_t, 0.0)
    P_prev_safe = np.where(mask, P_prev, 1.0)

    if use_hourly:
        hourly_df, err = _buffer_to_hourly_prices(buf, tickers)
        if hourly_df is None or len(hourly_df) < vm["p"] + 1:
            return None
        P_hourly = hourly_df[tickers].values.astype(float)
        P_hourly = np.nan_to_num(P_hourly, nan=0.0, posinf=0.0, neginf=0.0)
        # Use last bar as "current" price; VAR needs p_order returns ending at current
        r_window = []
        for k in range(vm["p"]):
            i = len(P_hourly) - 1 - k
            if i < 1:
                break
            denom = np.where(P_hourly[i - 1] != 0, P_hourly[i - 1], 1.0)
            r_k = (P_hourly[i] - P_hourly[i - 1]) / denom
            r_k = np.nan_to_num(r_k, nan=0.0, posinf=0.0, neginf=0.0)
            r_window.append(r_k)
        if len(r_window) < vm["p"]:
            return None
        # r_window is already most recent first (r_window[0] = last hour return)
        P_curr = P_hourly[-1]
        P_curr_safe = np.where(np.isfinite(P_curr) & (P_curr != 0), P_curr, P_t_safe)
        P_pred = predict_next_returns_arp(r_window, mu_ret, vm, P_curr_safe)
        now = datetime.now(timezone.utc)
        target_ts = _ts_ms(now + timedelta(hours=1))
    else:
        # Minute mode: build return window from buffer (most recent first)
        r_window = []
        r_curr = (P_t_safe - P_prev_safe) / P_prev_safe
        r_curr = np.nan_to_num(r_curr, nan=0.0, posinf=0.0, neginf=0.0)
        r_window.append(r_curr)
        if len(buf) >= 2:
            for k in range(min(vm["p"] - 1, len(buf) - 1)):
                idx = len(buf) - 1 - k
                if idx < 1:
                    break
                p_now = np.array([buf[idx]["prices"].get(t, np.nan) for t in tickers])
                p_prv = np.array([buf[idx - 1]["prices"].get(t, np.nan) for t in tickers])
                valid = np.isfinite(p_now) & np.isfinite(p_prv) & (p_prv != 0)
                r_k = np.where(valid, (p_now - p_prv) / p_prv, 0.0)
                r_k = np.nan_to_num(r_k, nan=0.0, posinf=0.0, neginf=0.0)
                r_window.append(r_k)
        P_pred = predict_next_returns_arp(r_window, mu_ret, vm, P_t_safe)
        now = datetime.now(timezone.utc)
        target_ts = _ts_ms(now + timedelta(minutes=actual_lag))

    tau = actual_lag
    pred_dict = {t: round(float(P_pred[i]), 4) for i, t in enumerate(tickers) if mask[i]}
    base_dict = {t: round(float(P_t[i]), 4) for i, t in enumerate(tickers) if mask[i]}

    # Factor Transformer trajectory
    ft_trajectory = {}
    ft_timestamps_ms = []
    W_ft = ft_mod["W"] if ft_mod else 0
    H_ft = ft_mod["H"] if ft_mod else 0
    if ft_mod is not None:
        if use_hourly and hourly_df is not None and len(hourly_df) >= W_ft + 1:
            P_win = hourly_df[tickers].values.astype(float)[-(W_ft + 1):]
            P_win = np.nan_to_num(P_win, nan=1.0, posinf=1.0, neginf=1.0)
            P_ft_traj = predict_factor_transformer(ft_mod, P_window=P_win, P_t=P_curr_safe)
            for h in range(H_ft):
                ft_timestamps_ms.append(_ts_ms(now + timedelta(hours=h + 1)))
            for i_t, tk in enumerate(tickers):
                if mask[i_t]:
                    ft_trajectory[tk] = [round(float(P_ft_traj[h, i_t]), 4) for h in range(H_ft)]
        elif not use_hourly:
            if ft_mod.get("use_price_space") and len(buf) >= W_ft + 1:
                P_win = []
                for k in range(-(W_ft + 1), 0):
                    row = np.array([buf[k]["prices"].get(t, np.nan) for t in tickers])
                    P_win.append(row)
                P_win = np.array(P_win)
                P_win = np.nan_to_num(P_win, nan=1.0, posinf=1.0, neginf=1.0)
                P_ft_traj = predict_factor_transformer(ft_mod, P_window=P_win, P_t=P_t_safe)
            else:
                ft_r_window = []
                for k in range(W_ft - 1, -1, -1):
                    idx = len(buf) - 1 - k
                    if idx < 1:
                        ft_r_window.append(np.zeros(N))
                        continue
                    p_now = np.array([buf[idx]["prices"].get(t, np.nan) for t in tickers])
                    p_prv = np.array([buf[idx - 1]["prices"].get(t, np.nan) for t in tickers])
                    valid = np.isfinite(p_now) & np.isfinite(p_prv) & (p_prv != 0)
                    r_k = np.where(valid, (p_now - p_prv) / p_prv, 0.0)
                    r_k = np.nan_to_num(r_k, nan=0.0, posinf=0.0, neginf=0.0)
                    ft_r_window.append(r_k)
                ft_r_window = np.array(ft_r_window)
                P_ft_traj = predict_factor_transformer(ft_mod, R_window=ft_r_window, P_t=P_t_safe) if len(ft_r_window) >= W_ft else None
            if P_ft_traj is not None:
                for h in range(H_ft):
                    ft_timestamps_ms.append(_ts_ms(now + timedelta(minutes=h + 1)))
                for i_t, tk in enumerate(tickers):
                    if mask[i_t]:
                        ft_trajectory[tk] = [round(float(P_ft_traj[h, i_t]), 4) for h in range(H_ft)]
        else:
            P_ft_traj = None

    return {
        "timestamp_ms": _ts_ms(now),
        "target_ms": target_ts,
        "prices": pred_dict,
        "base_prices": base_dict,
        "trajectory": {t: [pred_dict[t]] for t in pred_dict},
        "timestamps_ms": [target_ts],
        "ft_trajectory": ft_trajectory,
        "ft_timestamps_ms": ft_timestamps_ms,
    }


def _score_prediction(prediction, actual_prices):
    """Compare a past prediction's final price to actual prices."""
    if prediction is None:
        return None
    scores = {}
    base = prediction.get("base_prices", {})
    for t, pred_price in prediction.get("prices", {}).items():
        if t in actual_prices and actual_prices[t] > 0:
            actual = actual_prices[t]
            err_pct = abs(pred_price - actual) / actual * 100
            base_price = base.get(t, actual)
            direction_pred = "up" if pred_price > base_price else "down"
            direction_actual = "up" if actual > base_price else "down"
            scores[t] = {
                "predicted": round(pred_price, 2),
                "actual": round(actual, 2),
                "error_pct": round(err_pct, 2),
                "direction_correct": direction_pred == direction_actual,
            }
    return scores


# ── Background loop ──────────────────────────────────────────────────────────

def background_loop():
    """Main background loop: fetch prices every 60s, predict every minute."""
    fetch_interval = 60  # seconds

    while True:
        time.sleep(fetch_interval)

        with state_lock:
            if state["status"] != "ready":
                continue
            tickers = list(state["tickers"])

        if not tickers:
            continue

        market_open = _is_market_open()

        new_prices, ts_ms = _fetch_latest_prices(tickers)
        if new_prices is None or len(new_prices) == 0:
            socketio.emit("market_status", {"open": market_open, "message": "No new data"})
            continue

        # Score previous prediction against this new actual data
        with state_lock:
            prev_pred = state["active_prediction"]
        if prev_pred is not None:
            scores = _score_prediction(prev_pred, new_prices)
            if scores:
                result_entry = {
                    "timestamp_ms": prev_pred["timestamp_ms"],
                    "target_ms": prev_pred["target_ms"],
                    "scores": scores,
                }
                with state_lock:
                    state["prediction_history"].append(result_entry)
                socketio.emit("prediction_scored", result_entry)

        with state_lock:
            state["prev_prices"] = dict(state["latest_prices"])
            state["latest_prices"].update(new_prices)
            if ts_ms:
                state["price_buffer"].append({"timestamp_ms": ts_ms, "prices": new_prices})

        socketio.emit("price_update", {
            "timestamp_ms": ts_ms or _ts_ms(),
            "prices": new_prices,
            "market_open": market_open,
        })
        socketio.emit("market_status", {"open": market_open})

        # Make new prediction for the next minute
        prediction = _make_prediction()
        if prediction:
            with state_lock:
                state["active_prediction"] = prediction
            socketio.emit("new_prediction", prediction)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    """Full current state for initial page load."""
    with state_lock:
        history = list(state["price_buffer"])
        pred_hist = list(state["prediction_history"])
        return jsonify({
            "status": state["status"],
            "error": state["error"],
            "tickers": state["tickers"],
            "latest_prices": state["latest_prices"],
            "active_prediction": state["active_prediction"],
            "prediction_history": pred_hist,
            "price_history": history[-240:],
            "train_info": state["train_info"],
            "market_open": _is_market_open(),
            "backtest": state.get("backtest"),
        })


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    """Force retrain models (ignores saved models). Pass ?fresh=1 to re-fetch data from source."""
    fresh = request.args.get("fresh", "0") == "1"
    with state_lock:
        tickers = state["tickers"] or TICKERS
        source = state["source"]
        years = state["years"]
    threading.Thread(
        target=train_model,
        args=(tickers, years, source, "1m"),
        kwargs={"fetch_fresh": fresh, "force_retrain": True},
        daemon=True,
    ).start()
    msg = "Retraining with fresh data..." if fresh else "Retraining from cache (saved models ignored)..."
    return jsonify({"ok": True, "message": msg})


@socketio.on("connect")
def handle_connect():
    """Send current state to newly connected client."""
    with state_lock:
        socketio.emit("status", {"status": state["status"], "tickers": state["tickers"]})
        if state["latest_prices"]:
            socketio.emit("price_update", {
                "timestamp_ms": _ts_ms(),
                "prices": state["latest_prices"],
                "market_open": _is_market_open(),
            })
        if state["active_prediction"]:
            socketio.emit("new_prediction", state["active_prediction"])
        if state["train_info"]:
            socketio.emit("train_info", state["train_info"])


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FinanceExp Live Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--source", default="auto", choices=("auto", "yfinance", "alpaca"))
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--fetch-fresh", action="store_true",
                        help="Force re-download of historical data (ignores cache)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain models (ignore saved models even if config matches)")
    args = parser.parse_args()

    state["source"] = args.source
    state["years"] = args.years

    print("=" * 60)
    print("  FinanceExp Live Dashboard")
    print("=" * 60)
    cached = _cache_path("1m").exists() and not args.fetch_fresh
    if cached:
        print("Using cached historical data (pass --fetch-fresh to re-download)")
    else:
        print(f"Fetching historical 1m data ({args.years}yr, source={args.source})...")
    print()

    train_thread = threading.Thread(
        target=train_model,
        args=(TICKERS, args.years, args.source, "1m"),
        kwargs={"fetch_fresh": args.fetch_fresh, "force_retrain": args.retrain},
        daemon=True,
    )
    train_thread.start()

    bg_thread = threading.Thread(target=background_loop, daemon=True)
    bg_thread.start()

    print(f"\nDashboard: http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
