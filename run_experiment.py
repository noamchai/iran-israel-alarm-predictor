#!/usr/bin/env python3
"""
Run the full experiment:
1. Load 1 year of (highest resolution) prices for many stocks
2. Split year into even slices; compute slice-averaged C_AB(τ) and mean
3. Predict future prices over slice horizon using correlation + mean
4. Compare to real data and report error metrics
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from pathlib import Path

from data_loader import fetch_stock_prices
from correlation_and_predict import (
    split_into_slices,
    split_into_overlapping_slices,
    slice_averaged_covariances,
    slice_averaged_covariances_returns,
    compute_quadratic_model,
    predict_with_quadratic_model,
    predict_next,
    predict_next_diagonal,
    predict_next_returns,
    fit_var_returns,
    predict_next_returns_arp,
    prices_to_returns,
    evaluate_predictions,
)
from nn_predictor import (
    train_nn_predictor,
    predict_nn,
    train_nn_anchored,
    predict_nn_anchored,
    train_nn_pairs,
    predict_nn_pairs,
    train_nn_nchoose2,
    train_nn_multistep,
    predict_nn_multistep,
    train_factor_transformer,
    predict_factor_transformer,
    predict_nn_nchoose2,
    train_per_stock_1h,
    predict_per_stock_1h_propagate,
)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# Default tickers: diverse set for correlation structure (more stocks = richer Cov matrix)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V", "JNJ", "WMT",
    "PG", "MA", "HD", "DIS", "BAC", "ADBE", "XOM", "NFLX", "CRM", "PEP",
    "KO", "COST", "AVGO", "ORCL", "MCD", "CSCO", "ACN", "ABT", "NEE", "TMO",
    "DHR", "LIN", "WFC", "INTC", "AMD", "QCOM", "TXN", "UNP", "HON", "UPS",
    "RTX", "LOW", "CAT", "DE", "GE", "IBM", "AMAT", "INTU", "NOW", "PLD",
]


def run_experiment(
    tickers: list[str] | None = None,
    years: float = 1.0,
    interval: str = "1d",
    n_slices: int = 12,
    prediction_lag: int = 1,
    regularization: float = 1e-5,
    hold_out_last_slice: bool = True,
    predictor: str = "diagonal",
    shrink: float = 1.0,
    window_length: int | None = None,
    step: int | None = None,
    alpha_quad: float = 1.0,
    max_tau1: int = 2,
    source: str = "auto",
    ar_order: int = 10,
    nn_window: int = 30,
    nn_horizon: int = 10,
) -> dict:
    """
    Run the full pipeline and return metrics.

    If hold_out_last_slice is True, we use slices 0..n_slices-2 to estimate
    μ and Cov(τ), then predict slice n_slices-1 and compare to actual.
    """
    tickers = tickers or DEFAULT_TICKERS
    print("Fetching price data...")
    prices = fetch_stock_prices(tickers, years=years, interval=interval, source=source)
    # Drop any ticker that has no data
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
    tickers = list(prices.columns)
    if len(tickers) < 2:
        raise ValueError("Need at least 2 stocks with valid data")
    print(f"Loaded {len(prices)} rows, {len(tickers)} tickers: {tickers}")

    n_rows = len(prices)
    if window_length is not None:
        # Overlapping windows: reserve last window_length bars for test; need at least window_length left for train
        max_window = n_rows // 2  # need train_part length >= window_length
        if window_length < 2 or window_length > n_rows - 1:
            raise ValueError(
                f"window_length must be between 2 and {n_rows - 1} (have {n_rows} rows)."
            )
        if window_length > max_window:
            raise ValueError(
                f"window_length={window_length} is too large: with {n_rows} rows we reserve the last {window_length} for test, "
                f"leaving only {n_rows - window_length} for training. We need at least {window_length} rows to form one train window. "
                f"Use window_length <= {max_window}. (With 1h data, Yahoo only provides ~60 days, so total bars are limited.)"
            )
        step_val = step if step is not None else max(1, window_length // 2)
        train_part = prices.iloc[: n_rows - window_length]
        test_slice = prices.iloc[n_rows - window_length :]
        train_slices = split_into_overlapping_slices(train_part, window_length, step_val)
        slice_len = window_length
        n_slices_actual = len(train_slices)
        print(f"Overlapping slices: window_length={window_length}, step={step_val}, train slices={n_slices_actual}")
    else:
        all_slices = split_into_slices(prices, n_slices)
        slice_len = min(len(s) for s in all_slices)
        if hold_out_last_slice:
            train_slices = all_slices[:-1]
            test_slice = all_slices[-1]
        else:
            train_slices = all_slices
            test_slice = all_slices[-1]
        n_slices_actual = len(train_slices)

    max_lag = min(prediction_lag, slice_len - 1)
    if predictor == "returns_arp":
        max_lag = min(max(prediction_lag, ar_order), slice_len - 2)
    if max_lag < 1:
        max_lag = 1

    print(f"Train slices: {n_slices_actual}, slice length = {slice_len} bars")
    print(f"Correlation window: {slice_len} bars per slice (Cov(τ) estimated in each slice, then averaged)")
    # Fail fast if test slice will be too short (only when not using overlapping)
    if window_length is None:
        tau_eff = min(prediction_lag, slice_len - 1) if slice_len > 1 else 1
        need_for_returns = tau_eff + 2
        if predictor == "returns" and slice_len < need_for_returns:
            raise ValueError(
                f"Slice length is {slice_len} but return predictor needs at least {need_for_returns} bars per slice "
                f"(τ+2). Use --window-length and --step for overlapping slices, or fewer n_slices / smaller --lag."
            )
    print(f"Predictor: {predictor}, shrink: {shrink}")
    nn_model, scaler_X, scaler_y = None, None, None
    nn_anchored_model, nn_anchored_scaler = None, None
    nn_pairs_model, nn_pairs_scaler_X, nn_pairs_scaler_y = None, None, None
    nn_nchoose2_models = None
    nn_ms_model, nn_ms_scaler_X, nn_ms_scaler_Y = None, None, None
    ft_dict = None
    per_stock_1h_models = None
    cov_at_lag = None
    if predictor == "per_stock_1h":
        W_1h, H_1h = 60, 60
        print(f"Training per-stock 1h-window predictor (50 models, W={W_1h}, next {H_1h} min)...")
        mu_avg = np.nanmean([s.values.mean(axis=0) for s in train_slices], axis=0)
        per_stock_1h_models = train_per_stock_1h(
            train_slices, W=W_1h, max_samples_per_stock=100_000, max_iter=100,
        )
        mu_ret_avg, cov_ret_at_lag = None, None
    elif predictor == "factor_transformer":
        print(f"Training Factor Transformer (W={nn_window}, H={nn_horizon}, K=10)...")
        mu_avg = np.nanmean([s.values.mean(axis=0) for s in train_slices], axis=0)
        ft_dict = train_factor_transformer(
            train_slices, W=nn_window, H=nn_horizon, K=10,
        )
        mu_ret_avg, cov_ret_at_lag = None, None
    elif predictor == "nn_multistep":
        print(f"Training multi-step NN predictor (W={nn_window}, H={nn_horizon})...")
        mu_avg = np.nanmean([s.values.mean(axis=0) for s in train_slices], axis=0)
        nn_ms_model, nn_ms_scaler_X, nn_ms_scaler_Y, _, _ = train_nn_multistep(
            train_slices, W=nn_window, H=nn_horizon,
        )
        mu_ret_avg, cov_ret_at_lag = None, None
    elif predictor == "nn_nchoose2":
        print("Training N choose 2 pair models (one small MLP per stock pair)...")
        mu_avg = np.nanmean([s.values.mean(axis=0) for s in train_slices], axis=0)
        tau_nn = min(prediction_lag, max_lag)
        nn_nchoose2_models = train_nn_nchoose2(train_slices, tau_nn)
        print(f"Trained {len(nn_nchoose2_models)} pair models.")
        cov_at_lag = []
        mu_ret_avg, cov_ret_at_lag = None, None
    elif predictor == "nn":
        print("Training NN predictor (one network for all stocks)...")
        print("Tip: --predictor returns often does better; use it for best results.")
        mu_avg = np.nanmean([s.values.mean(axis=0) for s in train_slices], axis=0)
        tau_nn = min(prediction_lag, max_lag)
        nn_model, scaler_X, scaler_y = train_nn_predictor(train_slices, tau_nn, predict_returns=True)
        mu_ret_avg, cov_ret_at_lag = None, None
    elif predictor == "nn_anchored":
        print("Computing slice-averaged covariances, then training NN to learn residual over linear predictor...")
        print("Tip: --predictor returns (return-space Cov) often outperforms NN variants.")
        mu_avg, cov_at_lag = slice_averaged_covariances(train_slices, max_lag)
        tau_nn = min(prediction_lag, max_lag)
        nn_anchored_model, nn_anchored_scaler = train_nn_anchored(
            train_slices, tau_nn, mu_avg, cov_at_lag[0], cov_at_lag[tau_nn], reg=regularization
        )
        mu_ret_avg, cov_ret_at_lag = None, None
    elif predictor == "nn_pairs":
        print("Training NN on pairwise (out-of-time style) features [d; upper(d d')]...")
        print("Tip: --predictor returns usually does better.")
        mu_avg = np.nanmean([s.values.mean(axis=0) for s in train_slices], axis=0)
        tau_nn = min(prediction_lag, max_lag)
        nn_pairs_model, nn_pairs_scaler_X, nn_pairs_scaler_y = train_nn_pairs(train_slices, tau_nn, mu_avg)
        cov_at_lag = []
        mu_ret_avg, cov_ret_at_lag = None, None
    else:
        mu_avg, cov_at_lag = slice_averaged_covariances(train_slices, max_lag)
        mu_ret_avg, cov_ret_at_lag = None, None
        var_model = None
        quad_model = None
        if predictor in ("returns", "returns_arp"):
            print("Computing slice-averaged return covariances...")
            mu_ret_avg, cov_ret_at_lag = slice_averaged_covariances_returns(train_slices, max_lag)
            if predictor == "returns_arp":
                var_model = fit_var_returns(mu_ret_avg, cov_ret_at_lag, ar_order, reg=regularization)
                print(f"  VAR({var_model['p']}) fitted, spectral radius = {var_model['spectral_radius']:.4f}")
        elif predictor == "quadratic":
            _tau = min(prediction_lag, max_lag)
            print("Building quadratic model (stacked multi-lag input, proper C_xx^{-1} contractions)...")
            quad_model = compute_quadratic_model(
                train_slices, tau2=_tau, max_tau1=max_tau1, reg=regularization
            )
        else:
            print(f"Computing slice-averaged covariances up to lag {max_lag}...")

    # Predict on test slice: for each t where t+τ is in range, predict P(t+τ) from P(t)
    P_test = test_slice.values.astype(float)
    T_test, N = P_test.shape
    tau = min(prediction_lag, max_lag)
    if tau > 0 and interval in ("1h", "60m"):
        print(f"Prediction horizon: {tau} bars = {tau} hours ahead")
    elif tau > 0 and interval == "1d":
        print(f"Prediction horizon: {tau} bars = {tau} days ahead")
    if cov_at_lag is not None and len(cov_at_lag) > 0:
        cov_0, cov_tau = cov_at_lag[0], cov_at_lag[tau]
    elif predictor == "nn_anchored":
        cov_0, cov_tau = cov_at_lag[0], cov_at_lag[tau]
    n_pred = T_test - tau
    if n_pred < 1:
        raise ValueError("Test slice too short for chosen prediction lag")

    # For return predictor we need r(t), so start at i=1
    # For returns_arp we need p past returns, so start at i=p
    # For quadratic we need returns [r(t), r(t-1), ..., r(t-L)], so need L+2 prices: start at i >= max_tau1+1
    if predictor == "per_stock_1h":
        start_i = 59  # need W=60 rows (0..59) as context, then predict 60..119
    elif predictor in ("nn_multistep", "factor_transformer"):
        start_i = nn_window + 1
    elif predictor == "returns":
        start_i = 1
    elif predictor == "returns_arp":
        start_i = max(1, ar_order)
    elif predictor == "quadratic":
        start_i = max_tau1 + 1
    else:
        start_i = 0
    n_pred_eff = n_pred - start_i
    if n_pred_eff < 1 and predictor not in ("nn", "nn_anchored", "nn_pairs", "nn_nchoose2", "nn_multistep", "factor_transformer", "per_stock_1h"):
        need_rows = tau + 2 if predictor == "returns" else (tau + max_tau1 + 2 if predictor == "quadratic" else tau + 1)
        raise ValueError(
            f"Test slice has {T_test} bars but need at least {need_rows} for "
            f"predictor={predictor} and τ={tau}. Use fewer slices (e.g. --n_slices 12 or 26) "
            "so slice length is larger, or use a smaller --lag or --max-tau1."
        )
    if predictor == "nn":
        start_i, n_pred_eff = 0, n_pred
    elif predictor in ("nn_anchored", "nn_pairs"):
        start_i, n_pred_eff = 1, n_pred - 1
    elif predictor == "nn_nchoose2":
        start_i, n_pred_eff = 1, n_pred - 1
    if predictor == "per_stock_1h":
        if T_test < 121:
            raise ValueError("per_stock_1h needs test slice length >= 121 (W+1 initial + H=60). Use --window-length 121 or larger.")
        n_pred_eff = 60  # one 60-step propagation

    pred = np.full((T_test, N), np.nan)
    actual = np.full((T_test, N), np.nan)
    if predictor in ("returns_arp", "nn_multistep", "factor_transformer"):
        _R_test = prices_to_returns(P_test)
        _R_test = np.nan_to_num(_R_test, nan=0.0, posinf=0.0, neginf=0.0)
    test_loop = range(start_i, n_pred)
    if predictor == "nn_nchoose2":
        test_loop = tqdm(test_loop, desc="Predicting test steps", unit="step")
    for i in test_loop:
        t_idx = i + tau
        if predictor == "nn_nchoose2":
            pred[t_idx] = predict_nn_nchoose2(nn_nchoose2_models, P_test[i], P_test[i - 1], N)
        elif predictor == "nn":
            pred[t_idx] = predict_nn(nn_model, scaler_X, scaler_y, P_test[i], predict_returns=True)
        elif predictor == "nn_anchored":
            pred[t_idx] = predict_nn_anchored(
                nn_anchored_model, nn_anchored_scaler, P_test[i], P_test[max(i-1, 0)], mu_avg, cov_0, cov_tau, reg=regularization
            )
        elif predictor == "nn_pairs":
            pred[t_idx] = predict_nn_pairs(nn_pairs_model, nn_pairs_scaler_X, nn_pairs_scaler_y, P_test[i], P_test[max(i-1, 0)])
        elif predictor == "full":
            pred[t_idx] = predict_next(
                P_test[i], mu_avg, cov_0, cov_tau, reg=regularization, shrink=shrink
            )
        elif predictor == "diagonal":
            pred[t_idx] = predict_next_diagonal(P_test[i], mu_avg, cov_0, cov_tau, shrink=shrink)
        elif predictor == "quadratic":
            # Need L+2 prices to get L+1 returns: [P(t-L-1), ..., P(t)]
            P_history = P_test[i - max_tau1 - 1 : i + 1]  # (max_tau1+2, N)
            pred[t_idx] = predict_with_quadratic_model(
                P_history, quad_model, alpha_quad=alpha_quad
            )
        elif predictor == "returns":
            r_t = (P_test[i] - P_test[i - 1]) / np.where(P_test[i - 1] != 0, P_test[i - 1], np.nan)
            r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)
            cov_0_ret = cov_ret_at_lag[0]
            cov_tau_ret = cov_ret_at_lag[tau]
            pred[t_idx] = predict_next_returns(
                r_t, mu_ret_avg, cov_0_ret, cov_tau_ret, P_test[i], reg=regularization
            )
        elif predictor == "returns_arp":
            r_window = [_R_test[i - 1 - k] for k in range(min(var_model["p"], i))]
            pred[t_idx] = predict_next_returns_arp(
                r_window, mu_ret_avg, var_model, P_test[i]
            )
        elif predictor == "nn_multistep":
            R_win = _R_test[i - nn_window:i][::-1]  # (W, N) most recent first
            P_traj = predict_nn_multistep(
                nn_ms_model, nn_ms_scaler_X, nn_ms_scaler_Y,
                R_win, P_test[i], nn_horizon,
            )
            pred[t_idx] = P_traj[min(tau - 1, nn_horizon - 1)]
        elif predictor == "factor_transformer":
            R_win = _R_test[i - nn_window:i]  # (W, N) chronological
            P_traj = predict_factor_transformer(ft_dict, R_win, P_test[i])
            pred[t_idx] = P_traj[min(tau - 1, nn_horizon - 1)]
        elif predictor == "per_stock_1h":
            # Single 60-step propagation at i=60: true data only in initial buffer (rows 0..60), then all predicted
            if i == 60:
                start_t = 60
                P_traj = np.column_stack([
                    predict_per_stock_1h_propagate(
                        per_stock_1h_models, P_test, start_t, j, H=60, W=60,
                    )
                    for j in range(N)
                ])
                pred[60:120] = P_traj
                actual[60:120] = P_test[60:120]
        else:
            actual[t_idx] = P_test[t_idx]

    valid_start = start_i + tau
    pred_valid = pred[valid_start : valid_start + n_pred_eff]
    actual_valid = actual[valid_start : valid_start + n_pred_eff]
    prior_valid = P_test[start_i : start_i + n_pred_eff]

    metrics = evaluate_predictions(
        actual_valid, pred_valid, tickers, prior_prices=prior_valid
    )

    # Baseline: predict average μ for every step (explicit use of average)
    baseline_pred = np.broadcast_to(mu_avg, (n_pred_eff, N))
    baseline_metrics = evaluate_predictions(
        actual_valid, baseline_pred, tickers, prior_prices=prior_valid
    )
    metrics["baseline_mean"] = baseline_metrics["overall"]

    # For plotting: always use the full test slice so the actual line is the same
    # regardless of predictor. Pad predictions with NaN where predictor can't start.
    full_actual = P_test[tau:]       # actual P(t+τ) for all t, length n_pred
    full_pred = pred[tau:]           # pred has NaN before valid_start
    full_prior = P_test[:n_pred]     # P(t) for all t

    return {
        "tickers": tickers,
        "n_slices": n_slices_actual,
        "slice_len": slice_len,
        "prediction_lag": tau,
        "n_test_points": int(n_pred_eff),
        "metrics": metrics,
        "mu_avg": mu_avg,
        "cov_at_lag": cov_at_lag if cov_at_lag is not None else [],
        "predictions": full_pred,
        "actual": full_actual,
        "prior": full_prior,
    }


INTERPRETATION = """
--- What the results tell us ---
• Time alignment: We use P(t) as the initial point to predict P(t+τ), then compare
  that prediction to the real value at t+τ. So prediction and actual are for the
  same time t+τ (we do NOT compare different times).
• R² = 1 - (squared errors)/(variance of actuals). Negative R² does NOT mean
  "predicted lower" — it means the model does worse than always predicting the
  average (bigger errors). For behaviour (up/down), use correlation; R² can be
  negative even when corr is high if the scale is wrong.
• We DO use the average (μ): the predictor is P_pred = μ + correction. The baseline
  "predict μ every step" R² is shown above; the model should beat it on price level.
• Price-level R² above baseline means the correlation correction adds something;
  per-stock R² > 0 means that stock beats its own mean. Negative per-stock R² =
  worse than predicting that stock’s average.
• Return prediction (R² and corr on one-step returns) is the strict test: negative
  R² or ~0 correlation means we don’t predict the next move.
• Predictor choice: diagonal = per-stock only (stable, uses μ); full = full Cov
  returns = return-space.
"""


def plot_results(result: dict, out_dir: str | Path | None = "plots") -> None:
    """Generate and save diagnostic plots."""
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    tickers = result["tickers"]
    actual = result["actual"]
    pred = result["predictions"]
    prior = result["prior"]
    m = result["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Predicted vs actual price (scatter)
    ax = axes[0, 0]
    a_flat = actual.ravel()
    p_flat = pred.ravel()
    mask = np.isfinite(a_flat) & np.isfinite(p_flat)
    ax.scatter(a_flat[mask], p_flat[mask], alpha=0.4, s=12, c="steelblue", edgecolors="none")
    lims = [min(a_flat[mask].min(), p_flat[mask].min()), max(a_flat[mask].max(), p_flat[mask].max())]
    ax.plot(lims, lims, "k--", alpha=0.7, label="Perfect prediction")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Predicted vs actual (price level)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")

    # 2. Predicted vs actual return (scatter)
    ax = axes[0, 1]
    ret_actual = (actual - prior) / np.where(prior != 0, prior, np.nan)
    ret_pred = (pred - prior) / np.where(prior != 0, prior, np.nan)
    ra = ret_actual.ravel()
    rp = ret_pred.ravel()
    mask = np.isfinite(ra) & np.isfinite(rp)
    ax.scatter(ra[mask], rp[mask], alpha=0.4, s=12, c="coral", edgecolors="none")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.axvline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Actual return")
    ax.set_ylabel("Predicted return")
    ax.set_title("Predicted vs actual (one-step return)")

    # 3. Per-stock R² (bar)
    ax = axes[1, 0]
    r2s = [m["per_stock"].get(s, {}).get("r2", np.nan) for s in tickers]
    colors = ["steelblue" if x >= 0 else "coral" for x in r2s]
    ax.bar(range(len(tickers)), r2s, color=colors, edgecolor="gray", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_ylabel("R² (price level)")
    ax.set_title("Per-stock R² (positive = better than mean)")

    # 4. Time series: actual vs predicted for multiple stocks (grid of 6)
    n_show = min(6, len(tickers))
    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=axes[1, 1].get_subplotspec(), hspace=0.4, wspace=0.3)
    fig.delaxes(axes[1, 1])
    for k in range(n_show):
        ax = fig.add_subplot(inner[k // 3, k % 3])
        t = np.arange(actual.shape[0])
        ax.plot(t, actual[:, k], "-", color="C0", linewidth=1.2, label="actual")
        ax.plot(t, pred[:, k], "--", color="C1", linewidth=1, label="pred")
        ax.set_title(tickers[k], fontsize=9)
        ax.set_xlabel("Time index in test", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", fontsize=6)

    plt.tight_layout()
    if out:
        plt.savefig(out / "experiment_results.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plots saved to {out / 'experiment_results.png'}")

        # Second figure: more stocks, one per subplot (3x4 = 12 stocks)
        n_grid = min(12, len(tickers))
        n_cols = 4
        n_rows = (n_grid + n_cols - 1) // n_cols
        fig2, ax2 = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows))
        if n_rows == 1:
            ax2 = ax2.reshape(1, -1)
        for k in range(n_grid):
            r, c = k // n_cols, k % n_cols
            a = ax2[r, c]
            t = np.arange(actual.shape[0])
            a.plot(t, actual[:, k], "-", color="C0", linewidth=1.2, label="actual")
            a.plot(t, pred[:, k], "--", color="C1", linewidth=1, label="pred")
            a.set_title(tickers[k], fontsize=9)
            a.set_xlabel("Time index in test", fontsize=7)
            a.tick_params(labelsize=7)
            a.legend(loc="best", fontsize=6)
        for k in range(n_grid, n_rows * n_cols):
            r, c = k // n_cols, k % n_cols
            ax2[r, c].set_visible(False)
        plt.tight_layout()
        plt.savefig(out / "experiment_results_stocks.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Stock time series (first {n_grid}) saved to {out / 'experiment_results_stocks.png'}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Out-of-time correlation and prediction experiment")
    p.add_argument("--tickers", nargs="+", default=None, help="Stock symbols (default: built-in list)")
    p.add_argument("--years", type=float, default=1.0, help="Years of history")
    p.add_argument("--interval", default="1d", help="Bar interval: 1d, 1h, 1m, 5m, 15m, 30m")
    p.add_argument("--source", default="auto", choices=("auto", "yfinance", "alpaca"),
    help="Data source: auto (alpaca if keys set, else yfinance), yfinance, or alpaca")
    p.add_argument("--n_slices", type=int, default=12, help="Number of contiguous slices (ignored if --window-length set)")
    p.add_argument("--window-length", type=int, default=None, metavar="W", help="Use overlapping windows of W bars; reserve last W bars for test (avoids short-slice issues)")
    p.add_argument("--step", type=int, default=None, metavar="S", help="Step between overlapping windows (default W/2 when --window-length set)")
    p.add_argument("--lag", type=int, default=1, help="Prediction lag (e.g. 1 = next time step)")
    p.add_argument("--reg", type=float, default=1e-5, help="Ridge regularization for Cov(0) inverse")
p.add_argument("--predictor", choices=("full", "diagonal", "returns", "returns_arp", "quadratic", "nn", "nn_anchored", "nn_pairs", "nn_nchoose2", "nn_multistep", "factor_transformer", "per_stock_1h"), default="returns",
        help="returns=AR(1); returns_arp=VAR(p); nn_multistep=NN multi-step; factor_transformer=PCA+Transformer; per_stock_1h=50 NNs, 1h context+current, 60-step propagate")
    p.add_argument("--ar-order", type=int, default=10,
    help="VAR(p) order for returns_arp predictor (default 10)")
    p.add_argument("--nn-window", type=int, default=30,
    help="Input window size for nn_multistep (default 30)")
    p.add_argument("--nn-horizon", type=int, default=10,
    help="Prediction horizon for nn_multistep (default 10)")
    p.add_argument("--alpha-quad", type=float, default=1.0,
    help="Weight on quadratic (3-point) correction; 1.0=exact formula, <1=shrinkage (default 1.0)")
    p.add_argument("--max-tau1", type=int, default=2,
    help="Max past lag L for stacked input x=[P(t),...,P(t-L)]; D=N*(L+1) (default 2)")
    p.add_argument("--shrink", type=float, default=1.0, help="Shrink correction toward mean (0=always predict μ, 1=full)")
    p.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    p.add_argument("--out-dir", default="plots", help="Directory to save plots (default: plots)")
    args = p.parse_args()

    # High-res + short slices: use 1h and more slices (period auto-capped to 60 days)
    if args.interval in ("1h", "60m", "30m", "15m") and args.n_slices <= 12:
        print("Tip: for intraday data, try more slices (e.g. --n_slices 15) for shorter slice windows.")

    result = run_experiment(
        tickers=args.tickers,
        years=args.years,
        interval=args.interval,
        n_slices=args.n_slices,
        prediction_lag=args.lag,
        regularization=args.reg,
        hold_out_last_slice=True,
        predictor=args.predictor,
        shrink=args.shrink,
        window_length=args.window_length,
        step=args.step,
        alpha_quad=args.alpha_quad,
        max_tau1=args.max_tau1,
        source=args.source,
        ar_order=args.ar_order,
        nn_window=args.nn_window,
        nn_horizon=args.nn_horizon,
    )

    m = result["metrics"]
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS: Prediction from slice-averaged correlation")
    print("=" * 60)
    print(f"Stocks: {result['tickers']}")
    print(f"Slices: {result['n_slices']}, slice length ~{result['slice_len']}, prediction lag τ = {result['prediction_lag']}")
    print(f"Test points: {result['n_test_points']}")
    print("(Each point: predict P(t+τ) from P(t); compare to actual P(t+τ) — same time t+τ.)")
    print("\n--- Baseline (predict average μ every step) ---")
    b = m.get("baseline_mean", {})
    print(f"  R²:          {b.get('r2', np.nan):.4f}  (model should beat this)")
    print("\n--- Overall (price level) ---")
    print(f"  RMSE:        {m['overall']['rmse']:.4f}")
    print(f"  MAE:         {m['overall']['mae']:.4f}")
    print(f"  R²:          {m['overall']['r2']:.4f}")
    print(f"  Corr(pred, actual): {m['overall']['corr']:.4f}")
    if m.get("returns"):
        r = m["returns"]
        print("\n--- Returns (next-step move from correlation model) ---")
        print(f"  R² (returns): {r['r2']:.4f}")
        print(f"  Corr(pred return, actual return): {r['corr']:.4f}")
        print(f"  RMSE (returns): {r['rmse']:.6f}")
    print("\n--- Per-stock R² and correlation ---")
    for sym in result["tickers"]:
        ps = m["per_stock"].get(sym, {})
        print(f"  {sym}: R² = {ps.get('r2', np.nan):.4f}, corr = {ps.get('corr', np.nan):.4f}")
    print(INTERPRETATION)
    print("=" * 60)

    if not args.no_plot:
        plot_results(result, out_dir=args.out_dir)
    return result


if __name__ == "__main__":
    main()
