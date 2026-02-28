"""
Out-of-time correlation C_AB(τ) = E[P_A(t) P_B(t+τ)] over slices, and prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass


def split_into_slices(
    prices: pd.DataFrame, n_slices: int
) -> list[pd.DataFrame]:
    """Split price DataFrame into n_slices contiguous slices of (approximately) equal length."""
    n = len(prices)
    if n < n_slices:
        raise ValueError(f"Not enough rows ({n}) for {n_slices} slices")
    base, extra = divmod(n, n_slices)
    lengths = [base + (1 if i < extra else 0) for i in range(n_slices)]
    start = 0
    slices = []
    for L in lengths:
        slices.append(prices.iloc[start : start + L])
        start += L
    return slices


def split_into_overlapping_slices(
    prices: pd.DataFrame,
    window_length: int,
    step: int,
) -> list[pd.DataFrame]:
    """
    Split into overlapping windows of length window_length, advancing by step each time.
    Slices are prices.iloc[i : i+window_length] for i in range(0, n-window_length+1, step).
    """
    n = len(prices)
    if n < window_length:
        raise ValueError(f"Not enough rows ({n}) for window length {window_length}")
    if step < 1:
        step = 1
    slices = []
    for start in range(0, n - window_length + 1, step):
        slices.append(prices.iloc[start : start + window_length])
    return slices


def covariance_at_lag(P: np.ndarray, tau: int) -> np.ndarray:
    """
    Compute Cov_AB(τ) = E[(P_A(t)-μ_A)(P_B(t+τ)-μ_B)] for all A,B.
    P shape (T, N) with T time steps, N stocks.
    """
    T, N = P.shape
    if tau >= T or tau < 0:
        return np.full((N, N), np.nan)
    mu = P.mean(axis=0)
    X = P[:-tau] - mu if tau > 0 else P - mu
    Y = P[tau:] - mu if tau > 0 else P - mu
    return (X.T @ Y) / (T - tau)


def raw_moment_at_lag(P: np.ndarray, tau: int) -> np.ndarray:
    """C_AB(τ) = E[P_A(t) P_B(t+τ)] (raw product moment)."""
    T, N = P.shape
    if tau >= T or tau < 0:
        return np.full((N, N), np.nan)
    X = P[:-tau] if tau > 0 else P
    Y = P[tau:] if tau > 0 else P
    return (X.T @ Y) / (T - tau)


def third_order_at_lag(P: np.ndarray, tau: int) -> np.ndarray:
    """
    Three-point correlation K_{cAB} = E[(P_A(t)-μ_A)(P_B(t)-μ_B)(P_c(t+τ)-μ_c)].
    Returns array of shape (N, N, N): K[c, A, B] = E[d_A(t) d_B(t) (P_c(t+τ)-μ_c)].
    (Single time shift; for two time shifts use third_order_two_lags.)
    """
    T, N = P.shape
    if tau >= T or tau < 0:
        return np.full((N, N, N), np.nan)
    mu = P.mean(axis=0)
    d = P[:-tau] - mu if tau > 0 else P - mu
    y = (P[tau:] - mu) if tau > 0 else (P - mu)  # (T-τ, N) for t+τ
    n = len(d)
    K = np.zeros((N, N, N))
    for t in range(n):
        for c in range(N):
            K[c] += y[t, c] * np.outer(d[t], d[t])
    return K / n


def third_order_two_lags(P: np.ndarray, tau1: int, tau2: int) -> np.ndarray:
    """
    Two-time-shift third-order CUMULANT (connected three-point correlation):
      κ_{cAB}(τ₁, τ₂) = E[ABC] - E[AB]E[C] - E[AC]E[B] - E[BC]E[A] + 2 E[A]E[B]E[C],
    with A = P_A(t)-μ_A, B = P_B(t-τ₁)-μ_B, C = P_c(t+τ₂)-μ_c.
    So we subtract the reducible 2pt×1pt terms; for centered vars this is just E[ABC],
    but the explicit subtraction is correct when means are not exactly zero (e.g. small samples).
    Returns (N, N, N): K[c,A,B].
    """
    T, N = P.shape
    if tau1 < 0 or tau2 < 0 or tau1 + tau2 >= T:
        return np.full((N, N, N), np.nan)
    mu = P.mean(axis=0)
    d_at_t = P[tau1 : T - tau2] - mu   # (n, N)  A
    d_past = P[0 : T - tau1 - tau2] - mu  # (n, N) B at t-τ₁
    y = P[tau1 + tau2 : T] - mu        # (n, N) C at t+τ₂
    n = len(d_at_t)
    # Raw third moment E[ABC]
    K_raw = np.zeros((N, N, N))
    for t in range(n):
        for c in range(N):
            K_raw[c] += y[t, c] * np.outer(d_at_t[t], d_past[t])
    K_raw /= n
    # 2pt × 1pt terms: E[AB], E[AC], E[BC], E[A], E[B], E[C]
    mA = d_at_t.mean(axis=0)   # (N,)
    mB = d_past.mean(axis=0)   # (N,)
    mC = y.mean(axis=0)        # (N,)
    cov_AB = (d_at_t.T @ d_past) / n   # (N, N)  E[A_a B_b]
    cov_AC = (d_at_t.T @ y) / n       # (N, N)  E[A_a C_c]
    cov_BC = (d_past.T @ y) / n       # (N, N)  E[B_b C_c]
    # κ[c,A,B] = K_raw - E[AB]E[C] - E[AC]E[B] - E[BC]E[A] + 2 E[A]E[B]E[C]  (all indices c,A,B)
    K = (
        K_raw
        - cov_AB[np.newaxis, :, :] * mC[:, np.newaxis, np.newaxis]
        - cov_AC.T[:, :, np.newaxis] * mB[np.newaxis, np.newaxis, :]
        - cov_BC.T[:, np.newaxis, :] * mA[np.newaxis, :, np.newaxis]
        + 2 * mA[np.newaxis, :, np.newaxis] * mB[np.newaxis, np.newaxis, :] * mC[:, np.newaxis, np.newaxis]
    )
    return K


def slice_averaged_third_order(
    slices: list[pd.DataFrame],
    tau: int,
) -> np.ndarray:
    """Slice-averaged third-order moment K_{cAB}(τ). Shape (N, N, N)."""
    N = slices[0].shape[1]
    K_list = []
    for df in slices:
        P = df.values.astype(float)
        T = P.shape[0]
        if T <= tau:
            continue
        K_list.append(third_order_at_lag(P, tau))
    if not K_list:
        return np.full((N, N, N), np.nan)
    return np.nanmean(K_list, axis=0)


def slice_averaged_third_order_two_lags(
    slices: list[pd.DataFrame],
    tau2: int,
    max_tau1: int,
) -> list[np.ndarray]:
    """
    Slice-averaged K_{cAB}(τ₁, τ₂) for τ₂ = tau2 (prediction horizon) and
    τ₁ = 0, 1, ..., max_tau1. Returns list of (N,N,N) arrays: K_at_tau1[τ₁][c,A,B].
    """
    N = slices[0].shape[1]
    K_by_tau1 = {t1: [] for t1 in range(max_tau1 + 1)}
    for df in slices:
        P = df.values.astype(float)
        T = P.shape[0]
        for tau1 in range(max_tau1 + 1):
            if tau1 + tau2 >= T or T - tau1 - tau2 < 2:
                continue
            K_by_tau1[tau1].append(third_order_two_lags(P, tau1, tau2))
    out = []
    for tau1 in range(max_tau1 + 1):
        if K_by_tau1[tau1]:
            out.append(np.nanmean(K_by_tau1[tau1], axis=0))
        else:
            out.append(np.zeros((N, N, N)))
    return out


def slice_averaged_covariances(
    slices: list[pd.DataFrame],
    max_lag: int,
) -> Tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Compute slice-averaged mean vector and covariance at lags 0..max_lag.

    Returns
    -------
    mu_avg : (N,) mean price per stock (averaged over slices)
    cov_at_lag : list of (N,N) arrays, cov_at_lag[τ] = averaged Cov(τ)
    """
    N = slices[0].shape[1]
    mus = []
    covs = {tau: [] for tau in range(max_lag + 1)}
    for df in slices:
        P = df.values.astype(float)
        T, n = P.shape
        if n != N or T <= max_lag:
            continue
        mu = P.mean(axis=0)
        mus.append(mu)
        for tau in range(max_lag + 1):
            if T - tau < 2:
                continue
            covs[tau].append(covariance_at_lag(P, tau))
    mu_avg = np.nanmean(mus, axis=0)
    cov_avg = []
    for tau in range(max_lag + 1):
        if covs[tau]:
            cov_avg.append(np.nanmean(covs[tau], axis=0))
        else:
            cov_avg.append(np.full((N, N), np.nan))
    return mu_avg, cov_avg


def prices_to_returns(P: np.ndarray) -> np.ndarray:
    """Simple return: (P[t+1]-P[t])/P[t]. Shape (T-1, N)."""
    return (P[1:] - P[:-1]) / np.where(P[:-1] != 0, P[:-1], np.nan)


def slice_averaged_covariances_returns(
    slices: list[pd.DataFrame],
    max_lag: int,
    show_progress: bool = True,
) -> Tuple[np.ndarray, list[np.ndarray]]:
    """
    Slice-averaged mean return and covariance of returns at lags 0..max_lag.
    Uses running sums to avoid storing all matrices in memory.
    Returns mu_ret (N,), list of (N,N) cov_ret_at_lag.
    """
    N = slices[0].shape[1]
    mu_sum = np.zeros(N)
    mu_count = 0
    cov_sum = [np.zeros((N, N)) for _ in range(max_lag + 1)]
    cov_count = [0] * (max_lag + 1)
    it = slices
    if show_progress and len(slices) > 100:
        try:
            from tqdm import tqdm
            it = tqdm(slices, desc=f"Covariances (lags 0..{max_lag})", unit="slice")
        except ImportError:
            pass
    for df in it:
        P = df.values.astype(float)
        R = prices_to_returns(P)
        T, n = R.shape
        if n != N or T <= max_lag:
            continue
        mu = np.nanmean(R, axis=0)
        mu_sum += np.nan_to_num(mu, nan=0.0)
        mu_count += 1
        R_clean = np.nan_to_num(R, nan=0.0)
        for tau in range(max_lag + 1):
            if T - tau < 2:
                continue
            cov_sum[tau] += covariance_at_lag(R_clean, tau)
            cov_count[tau] += 1
    mu_avg = mu_sum / max(mu_count, 1)
    cov_avg = [cov_sum[tau] / max(cov_count[tau], 1) if cov_count[tau] > 0
               else np.full((N, N), np.nan) for tau in range(max_lag + 1)]
    return mu_avg, cov_avg


def predict_next(
    P_t: np.ndarray,
    mu: np.ndarray,
    cov_0: np.ndarray,
    cov_tau: np.ndarray,
    reg: float = 1e-6,
    shrink: float = 1.0,
) -> np.ndarray:
    """
    Linear predictor for P(t+τ) given P(t):
    P_pred(t+τ) = μ + shrink * Cov(τ)' @ inv(Cov(0)) @ (P(t) - μ)
    shrink=0 => always predict μ (average); shrink=1 => full correction.
    """
    N = len(mu)
    cov_0_reg = cov_0 + reg * np.eye(N)
    try:
        L = np.linalg.cholesky(cov_0_reg)
        Linv = np.linalg.solve(L, np.eye(N))
        cov0_inv = Linv.T @ Linv
    except np.linalg.LinAlgError:
        cov0_inv = np.linalg.pinv(cov_0_reg)
    delta = (P_t - mu).reshape(-1, 1)
    pred_delta = (cov_tau.T @ cov0_inv) @ delta
    return (mu + shrink * pred_delta.ravel()).ravel()


def predict_next_diagonal(
    P_t: np.ndarray,
    mu: np.ndarray,
    cov_0: np.ndarray,
    cov_tau: np.ndarray,
    shrink: float = 1.0,
) -> np.ndarray:
    """
    Per-stock predictor using only diagonal (own-stock) covariance; explicitly uses average μ.
    P_pred_i = μ_i + shrink * (Cov_ii(τ)/Cov_ii(0)) * (P_i(t) - μ_i)
    No cross-stock terms, so scale is stable and average is central.
    """
    var_0 = np.diag(cov_0)
    cov_tau_diag = np.diag(cov_tau)
    ratio = np.where(np.abs(var_0) > 1e-12, cov_tau_diag / var_0, 0.0)
    pred_delta = shrink * ratio * (P_t - mu)
    return (mu + pred_delta).ravel()


def compute_quadratic_model(
    slices: list[pd.DataFrame],
    tau2: int,
    max_tau1: int = 2,
    reg: float = 1e-5,
) -> dict:
    """
    Build the proper second-order conditional predictor **in return space**.

    Returns r(t) = (P(t)-P(t-1))/P(t-1). Stacks input:
      x = [r(t), r(t-1), ..., r(t-L)]  with L = max_tau1, dimension D = N*(L+1).
    Output:
      y = r(t+τ₂)  (return τ₂ steps ahead), dimension N.

    Computes (pooled over slices):
      C_xx  (D,D) = Cov(x, x)
      C_yx  (N,D) = Cov(y, x)
      κ_c   (D,D) = E[δy_c · δx δxᵀ]   (third cumulant = connected 3-point function)
      b     (N,D) = C_yx C_xx⁻¹          (linear coefficient — multi-lag VAR)
      H_c   (D,D) = C_xx⁻¹ κ_c C_xx⁻¹  (quadratic coefficient — proper contractions)

    Predictor (in return space):
      r̂_c = μ_y_c + b_c · δx + ½ α δxᵀ H_c δx
    Then convert back: P_pred = P(t) · (1 + r̂).
    """
    N = slices[0].shape[1]
    L = max_tau1
    D = N * (L + 1)

    all_X, all_Y = [], []
    for df in slices:
        P = df.values.astype(float)
        R = prices_to_returns(P)  # (T-1, N)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        T_r = R.shape[0]
        t_start = L
        t_end = T_r - tau2
        if t_end - t_start < 2:
            continue
        for t in range(t_start, t_end):
            x = np.concatenate([R[t - lag] for lag in range(L + 1)])
            all_X.append(x)
            all_Y.append(R[t + tau2])

    X = np.array(all_X)  # (n, D)
    Y = np.array(all_Y)  # (n, N)
    n = len(X)
    print(f"  Quadratic model (return space): D={D} (N={N}, L={L}), {n} training samples")

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    dX = X - mu_x
    dY = Y - mu_y

    C_xx = (dX.T @ dX) / n
    C_yx = (dY.T @ dX) / n

    C_xx_reg = C_xx + reg * np.eye(D)
    try:
        Lc = np.linalg.cholesky(C_xx_reg)
        C_xx_inv = np.linalg.solve(Lc.T, np.linalg.solve(Lc, np.eye(D)))
    except np.linalg.LinAlgError:
        C_xx_inv = np.linalg.pinv(C_xx_reg)

    b = C_yx @ C_xx_inv  # (N, D)

    H = np.zeros((N, D, D))
    for c in range(N):
        kappa_c = (dX * dY[:, c:c+1]).T @ dX / n  # (D, D)
        H[c] = C_xx_inv @ kappa_c @ C_xx_inv

    return {"mu_x": mu_x, "mu_y": mu_y, "b": b, "H": H, "N": N, "D": D, "L": L}


def predict_with_quadratic_model(
    P_history: np.ndarray,
    quad_model: dict,
    alpha_quad: float = 1.0,
) -> np.ndarray:
    """
    Predict P(t+τ₂) using the pre-computed quadratic model (return space).

    P_history shape (L+2, N): [P(t-L-1), ..., P(t-1), P(t)] (most recent last).
    We compute returns from the price history, stack them, predict return, then
    convert back: P_pred = P(t) * (1 + r_pred).
    """
    N = quad_model["N"]
    L = quad_model["L"]
    mu_x = quad_model["mu_x"]
    mu_y = quad_model["mu_y"]
    b = quad_model["b"]
    H = quad_model["H"]

    R_hist = prices_to_returns(P_history)  # (L+1, N)
    R_hist = np.nan_to_num(R_hist, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.concatenate([R_hist[-(1 + lag)] for lag in range(L + 1)])
    dx = x - mu_x

    linear = b @ dx
    quad = np.array([0.5 * dx @ H[c] @ dx for c in range(N)])

    r_pred = mu_y + linear + alpha_quad * quad
    P_t = P_history[-1]
    return P_t * (1.0 + r_pred)


def fit_var_returns(
    mu_ret: np.ndarray,
    cov_ret_list: list[np.ndarray],
    p: int,
    reg: float = 1e-6,
) -> dict:
    """
    Fit a VAR(p) model in return space via Yule-Walker equations.

    r(t) = mu + A1 @ (r(t-1)-mu) + A2 @ (r(t-2)-mu) + ... + Ap @ (r(t-p)-mu)

    Uses Gamma(0)..Gamma(p) from cov_ret_list to build the block-Toeplitz
    system and solve for A1..Ap.

    Returns dict with A_coeffs (list of p NxN matrices), mu_ret, p, B.
    """
    N = len(mu_ret)
    p = min(p, len(cov_ret_list) - 1)
    if p < 1:
        raise ValueError("Need at least Gamma(0) and Gamma(1) for VAR(1)")

    # Block Toeplitz Gamma_xx (pN x pN): block(i,j) = Gamma(j-i)
    Gamma_xx = np.zeros((p * N, p * N))
    for i in range(p):
        for j in range(p):
            lag = j - i
            if abs(lag) < len(cov_ret_list):
                block = cov_ret_list[abs(lag)]
                if lag < 0:
                    block = block.T
                Gamma_xx[i * N:(i + 1) * N, j * N:(j + 1) * N] = block

    # RHS: Gamma_yx (N x pN): block i = Gamma(i+1)
    Gamma_yx = np.zeros((N, p * N))
    for i in range(p):
        lag = i + 1
        if lag < len(cov_ret_list):
            Gamma_yx[:, i * N:(i + 1) * N] = cov_ret_list[lag]

    # Solve B = Gamma_yx @ Gamma_xx^{-1}, B is (N x pN)
    Gamma_reg = Gamma_xx + reg * np.eye(p * N)
    try:
        L = np.linalg.cholesky(Gamma_reg)
        Gamma_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(p * N)))
    except np.linalg.LinAlgError:
        Gamma_inv = np.linalg.pinv(Gamma_reg)

    B = Gamma_yx @ Gamma_inv  # (N, pN)
    A_coeffs = [B[:, i * N:(i + 1) * N] for i in range(p)]

    # Spectral radius check (largest eigenvalue of companion matrix)
    companion = np.zeros((p * N, p * N))
    companion[:N, :] = B
    if p > 1:
        companion[N:, :-N] = np.eye((p - 1) * N)
    eigs = np.abs(np.linalg.eigvals(companion))
    spectral_radius = np.max(eigs) if len(eigs) > 0 else 0.0

    return {
        "A_coeffs": A_coeffs,
        "mu_ret": mu_ret,
        "p": p,
        "B": B,
        "spectral_radius": spectral_radius,
    }


def predict_next_returns_arp(
    r_window: list[np.ndarray],
    mu_ret: np.ndarray,
    var_model: dict,
    P_t: np.ndarray,
) -> np.ndarray:
    """
    VAR(p) prediction: predict r(t+1) from [r(t), r(t-1), ..., r(t-p+1)].

    r_window: list of (N,) return vectors, most recent first. Length >= 1.
    P_t: (N,) current prices.
    Returns P_pred = P_t * (1 + r_pred).
    """
    p = var_model["p"]
    A_coeffs = var_model["A_coeffs"]
    N = len(mu_ret)
    P_t = np.asarray(P_t).ravel()[:N]

    r_pred = mu_ret.copy()
    for i in range(min(p, len(r_window))):
        r_i = np.asarray(r_window[i]).ravel()[:N]
        r_pred = r_pred + A_coeffs[i] @ (r_i - mu_ret)

    return (P_t * (1 + r_pred)).ravel()[:N]


def predict_next_returns(
    r_t: np.ndarray,
    mu_ret: np.ndarray,
    cov_0_ret: np.ndarray,
    cov_tau_ret: np.ndarray,
    P_t: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    """
    Predict return at t+τ from return at t; then P_pred = P(t) * (1 + r_pred).
    Keeps scale per-stock and uses mean return (≈0) in return space.
    """
    N = len(mu_ret)
    r_t = np.asarray(r_t).ravel()[:N]
    P_t = np.asarray(P_t).ravel()[:N]
    mu_ret = np.asarray(mu_ret).ravel()[:N]
    cov_0_reg = np.asarray(cov_0_ret)[:N, :N] + reg * np.eye(N)
    cov_tau_ret = np.asarray(cov_tau_ret)[:N, :N]
    try:
        L = np.linalg.cholesky(cov_0_reg)
        cov0_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N)))
    except np.linalg.LinAlgError:
        cov0_inv = np.linalg.pinv(cov_0_reg)
    delta = (r_t - mu_ret).reshape(N, 1)
    # (N,N) @ (N,1) -> (N,1); ensure 1D length N
    correction = (cov_tau_ret.T @ cov0_inv) @ delta
    r_pred = (mu_ret + correction.reshape(N)).ravel()[:N]
    return (P_t * (1 + r_pred)).ravel()[:N]


def evaluate_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    tickers: list[str],
    prior_prices: np.ndarray | None = None,
) -> dict:
    """
    Compute per-stock and overall RMSE, MAE, R², and correlation.
    If prior_prices is given (same shape as actual), also report metrics in return space:
    return_t = (actual_t - prior_t) / prior_t (simple return from prior to actual).
    """
    results = {"overall": {}, "per_stock": {}, "returns": None}
    # Flatten or per-stock
    a = np.asarray(actual).ravel()
    p = np.asarray(predicted).ravel()
    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if len(a) < 2:
        results["overall"] = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "corr": np.nan}
        return results
    rmse = np.sqrt(np.mean((a - p) ** 2))
    mae = np.mean(np.abs(a - p))
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    corr = np.corrcoef(a, p)[0, 1] if np.std(p) > 0 and np.std(a) > 0 else np.nan
    results["overall"] = {"rmse": rmse, "mae": mae, "r2": r2, "corr": corr}

    # Per stock (if actual/predicted are (T, N))
    if actual.ndim == 2 and predicted.ndim == 2 and actual.shape[1] == len(tickers):
        for j, sym in enumerate(tickers):
            aj, pj = actual[:, j], predicted[:, j]
            mask = np.isfinite(aj) & np.isfinite(pj)
            if mask.sum() < 2:
                results["per_stock"][sym] = {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "corr": np.nan}
                continue
            aj, pj = aj[mask], pj[mask]
            results["per_stock"][sym] = {
                "rmse": np.sqrt(np.mean((aj - pj) ** 2)),
                "mae": np.mean(np.abs(aj - pj)),
                "r2": 1 - (np.sum((aj - pj) ** 2) / (np.sum((aj - np.mean(aj)) ** 2) or 1e-20)),
                "corr": np.corrcoef(aj, pj)[0, 1] if np.std(pj) > 0 and np.std(aj) > 0 else np.nan,
            }

    # Return-space metrics (predictive power for direction/scale of move)
    if prior_prices is not None and prior_prices.shape == actual.shape:
        prior = np.asarray(prior_prices)
        ret_actual = (actual - prior) / (np.where(prior != 0, prior, np.nan))
        ret_pred = (predicted - prior) / (np.where(prior != 0, prior, np.nan))
        mask = np.isfinite(ret_actual) & np.isfinite(ret_pred)
        ra = ret_actual[mask].ravel()
        rp = ret_pred[mask].ravel()
        if len(ra) >= 2 and np.any(ra != 0):
            ss_res = np.sum((ra - rp) ** 2)
            ss_tot = np.sum((ra - np.mean(ra)) ** 2)
            r2_ret = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            corr_ret = np.corrcoef(ra, rp)[0, 1] if np.std(rp) > 0 and np.std(ra) > 0 else np.nan
            results["returns"] = {"r2": r2_ret, "corr": corr_ret, "rmse": np.sqrt(np.mean((ra - rp) ** 2))}
    return results
