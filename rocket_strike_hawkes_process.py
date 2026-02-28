"""
Hawkes Process model for rocket strike prediction.

A Hawkes process is a self-exciting point process — exactly right for rocket
strikes, which cluster in bursts.  Each strike temporarily raises the
probability of another strike, with exponential decay.

Intensity function:
    λ(t) = μ  +  α * Σ_{t_i < t} exp(−β * (t − t_i))

Parameters (all in units of **minutes**):
    μ (mu)    : baseline intensity  [strikes / minute]
    α (alpha) : excitability        [how much each strike raises intensity]
    β (beta)  : decay rate          [1/β = mean self-excitation lifetime in minutes]
    α/β < 1   : stationarity condition (branching ratio < 1)

Public API
----------
fit_hawkes(strike_times_minutes)        → (mu, alpha, beta)
hawkes_predict(past_relative, mu, alpha, beta, horizon=60)
    → (per_minute_probs, cumulative_probs)
hawkes_backtest_intensity(df_datetime_col, df_strike_col, mu, alpha, beta, window_minutes=300)
    → list of per-minute P(strike) for the backtest window
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------

def _loglik(params: np.ndarray, t: np.ndarray, T: float) -> float:
    """Negative log-likelihood of univariate Hawkes with exponential kernel.
    t  : sorted event times in [0, T]
    T  : observation end time
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return 1e15

    n = len(t)
    if n == 0:
        return mu * T  # just penalty, no events

    # Recursive R_i = Σ_{j<i} exp(−β*(t_i−t_j)), vectorised over dt array
    dt = np.diff(t)               # shape (n-1,)
    decay = np.exp(-beta * dt)    # precompute exp terms
    R = np.zeros(n)
    for i in range(1, n):         # still sequential but tight inner work
        R[i] = decay[i - 1] * (1.0 + R[i - 1])

    intensities = mu + alpha * R
    if np.any(intensities <= 0):
        return 1e15

    ll = float(np.sum(np.log(intensities)))
    ll -= mu * T
    ll -= (alpha / beta) * float(np.sum(1.0 - np.exp(-beta * (T - t))))
    return -ll   # return NEGATIVE for scipy minimise


def fit_hawkes(
    strike_times_minutes: List[float],
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """Fit Hawkes process (mu, alpha, beta) via MLE on historical strike times.

    Args:
        strike_times_minutes : list of event times in minutes (any reference point).
        verbose              : print fitted parameters.

    Returns:
        (mu, alpha, beta) — all in units of minutes.
    """
    if len(strike_times_minutes) < 3:
        return 0.005, 0.5, 0.2   # sensible defaults when no data

    t = np.array(sorted(strike_times_minutes), dtype=float)
    t -= t[0]                    # normalise so t[0] == 0
    T = float(t[-1]) + 1.0
    n = len(t)
    mu0 = n / T * 0.2            # rough baseline (20 % of overall rate)

    best_val, best_params = np.inf, np.array([mu0, 0.5, 0.2])

    # Try a small grid of starting points to avoid local minima
    for mu_init in [mu0 * 0.5, mu0 * 2.0]:
        for alpha_init in [0.3, 0.8]:
            for beta_init in [0.05, 0.2]:
                try:
                    res = minimize(
                        _loglik,
                        [mu_init, alpha_init, beta_init],
                        args=(t, T),
                        method="L-BFGS-B",
                        bounds=[(1e-7, 20.0), (1e-7, 20.0), (1e-7, 20.0)],
                        options={"maxiter": 500, "ftol": 1e-10},
                    )
                    if res.fun < best_val:
                        best_val, best_params = res.fun, res.x
                except Exception:
                    pass

    mu, alpha, beta = best_params.tolist()
    branching = alpha / beta
    if verbose:
        print(
            f"  Hawkes fit: μ={mu:.5f}  α={alpha:.4f}  β={beta:.4f}"
            f"  branching={branching:.3f}"
            f"  (mean decay={1/beta:.1f} min)",
            flush=True,
        )
    return float(mu), float(alpha), float(beta)


# ---------------------------------------------------------------------------
# Forward prediction
# ---------------------------------------------------------------------------

def _current_excitation(past_events_relative: List[float], beta: float) -> float:
    """A = Σ_{t_i ≤ 0} exp(−β * (0 − t_i))  — the excitation sum at t=0 (now)."""
    A = 0.0
    for ti in past_events_relative:
        if ti <= 0:
            A += np.exp(beta * ti)   # ti is negative, so this decays
    return A


def hawkes_predict(
    past_events_relative: List[float],   # minutes before now (negative or 0)
    mu: float,
    alpha: float,
    beta: float,
    horizon: int = 60,
) -> Tuple[List[float], List[float]]:
    """Forecast P(strike) for each of the next `horizon` minutes.

    Uses the Poisson-approximation survival function (ignores within-horizon
    self-excitation from future events — valid when branching ratio α/β < 1).

    Returns:
        per_minute : P(at least one strike in minute k)   k=1..horizon
        cumulative : P(at least one strike by minute k)   k=1..horizon
    """
    A = _current_excitation(past_events_relative, beta)

    # Survival function: P(no strike in [0, τ]) = exp(−Λ(0, τ))
    # Λ(0, τ) = μτ + (α*A/β)*(1 − exp(−βτ))
    def survival(tau: float) -> float:
        if tau <= 0:
            return 1.0
        lam_int = mu * tau + (alpha * A / beta) * (1.0 - np.exp(-beta * tau))
        return float(np.exp(-max(0.0, lam_int)))

    per_minute = []
    cumulative = []
    for k in range(1, horizon + 1):
        s_prev = survival(k - 1)
        s_curr = survival(k)
        p_min  = float(np.clip(s_prev - s_curr, 0.0, 1.0))
        per_minute.append(p_min)
        cumulative.append(float(np.clip(1.0 - s_curr, 0.0, 1.0)))

    return per_minute, cumulative


# ---------------------------------------------------------------------------
# Historical backtest: Hawkes intensity over past window
# ---------------------------------------------------------------------------

def hawkes_backtest_intensity(
    times_sorted: List[float],    # all event times in the full timeline (minutes from epoch)
    backtest_times: List[float],  # query minutes (must be >= min of times_sorted)
    mu: float,
    alpha: float,
    beta: float,
) -> List[float]:
    """Return per-minute P(strike) for each minute in backtest_times.

    For each query minute t, uses only strikes that occurred BEFORE t
    (causally correct evaluation).
    """
    all_events = np.array(sorted(times_sorted), dtype=float)
    probs = []
    for t in backtest_times:
        past = all_events[all_events < t]
        A = float(np.sum(np.exp(-beta * (t - past)))) if len(past) else 0.0
        lam = mu + alpha * A
        p = float(np.clip(1.0 - np.exp(-lam), 0.0, 1.0))  # P(>=1 in this 1-min window)
        probs.append(p)
    return probs
