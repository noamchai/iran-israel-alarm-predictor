"""
NN predictors inspired by out-of-time covariance structure:
- nn: naive P(t) -> P(t+τ) (no covariance).
- nn_anchored: linear covariance predictor + NN learns the residual (never worse than Cov).
- nn_pairs: NN on pairwise (out-of-time style) features d_i*d_j from P(t)-μ.
- nn_nchoose2: one small MLP per pair (i,j), 2 inputs -> 2 outputs; combine by averaging per stock.
- factor_transformer: PCA factor reduction + Transformer attention over time.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import List, Tuple, Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    MLPRegressor = None
    StandardScaler = None


def _linear_pred(P_t: np.ndarray, mu: np.ndarray, cov_0: np.ndarray, cov_tau: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """P_linear = μ + Cov(τ)' inv(Cov(0)) (P(t)-μ)."""
    N = len(mu)
    P_t = np.asarray(P_t).ravel()[:N]
    mu = np.asarray(mu).ravel()[:N]
    cov_0 = np.asarray(cov_0)[:N, :N]
    cov_tau = np.asarray(cov_tau)[:N, :N]
    cov_0_reg = cov_0 + reg * np.eye(N)
    try:
        L = np.linalg.cholesky(cov_0_reg)
        cov0_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N)))
    except np.linalg.LinAlgError:
        cov0_inv = np.linalg.pinv(cov_0_reg)
    delta = (P_t - mu).reshape(N, 1)
    pred_delta = (cov_tau.T @ cov0_inv) @ delta
    out = mu + pred_delta.reshape(N)
    return out.ravel()[:N]


def _pairwise_features(d: np.ndarray) -> np.ndarray:
    """d is (N,). Return [d; upper triangle of d d'] so N + N(N+1)/2."""
    N = len(d)
    d = np.asarray(d).ravel()
    pairs = np.outer(d, d)
    upper = pairs[np.triu_indices(N)]
    return np.concatenate([d, upper])


def build_sequences(
    slices: List[pd.DataFrame],
    tau: int,
    predict_returns: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    From a list of slices (each T x N), build (X, y) for predicting τ bars ahead.
    X[i] = P(t), y[i] = P(t+τ) [or return (P(t+τ)-P(t))/P(t) if predict_returns].
    """
    X_list, y_list = [], []
    for df in slices:
        P = df.values.astype(float)
        T, N = P.shape
        if T <= tau:
            continue
        for t in range(0, T - tau):
            x = P[t]
            if predict_returns:
                y = (P[t + tau] - P[t]) / np.where(P[t] != 0, P[t], np.nan)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                y = P[t + tau]
            X_list.append(x)
            y_list.append(y)
    if not X_list:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.array(X_list), np.array(y_list)


def build_sequences_anchored(
    slices: List[pd.DataFrame],
    tau: int,
    mu: np.ndarray,
    cov_0: np.ndarray,
    cov_tau: np.ndarray,
    reg: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """X = [r(t), r_linear_pred], y = r_actual(t+τ) - r_linear (residual in return space)."""
    X_list, y_list = [], []
    for df in slices:
        P = df.values.astype(float)
        T, N = P.shape
        if T <= tau + 1:
            continue
        for t in range(1, T - tau):
            r_t = (P[t] - P[t-1]) / np.where(P[t-1] != 0, P[t-1], 1.0)
            r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)
            p_linear = _linear_pred(P[t], mu, cov_0, cov_tau, reg=reg)
            r_linear = (p_linear - P[t]) / np.where(P[t] != 0, P[t], 1.0)
            r_linear = np.nan_to_num(r_linear, nan=0.0, posinf=0.0, neginf=0.0)
            r_actual = (P[t+tau] - P[t]) / np.where(P[t] != 0, P[t], 1.0)
            r_actual = np.nan_to_num(r_actual, nan=0.0, posinf=0.0, neginf=0.0)
            x = np.concatenate([r_t, r_linear])
            y_residual = r_actual - r_linear
            X_list.append(x)
            y_list.append(y_residual)
    if not X_list:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.array(X_list), np.array(y_list)


def build_sequences_pairs(
    slices: List[pd.DataFrame],
    tau: int,
    mu: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """X = [r(t); upper triangle of r(t) r(t)'] (return-space pairwise features), y = r(t+τ)."""
    X_list, y_list = [], []
    for df in slices:
        P = df.values.astype(float)
        T, N = P.shape
        if T <= tau + 1:
            continue
        for t in range(1, T - tau):
            r_t = (P[t] - P[t-1]) / np.where(P[t-1] != 0, P[t-1], 1.0)
            r_t = np.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)
            r_fut = (P[t+tau] - P[t+tau-1]) / np.where(P[t+tau-1] != 0, P[t+tau-1], 1.0)
            r_fut = np.nan_to_num(r_fut, nan=0.0, posinf=0.0, neginf=0.0)
            x = _pairwise_features(r_t)
            X_list.append(x)
            y_list.append(r_fut)
    if not X_list:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.array(X_list), np.array(y_list)


def train_nn_anchored(
    train_slices: List[pd.DataFrame],
    tau: int,
    mu: np.ndarray,
    cov_0: np.ndarray,
    cov_tau: np.ndarray,
    reg: float = 1e-6,
    hidden_layer_sizes: tuple = (128, 64),
    max_iter: int = 400,
    random_state: int = 0,
) -> Tuple[Any, Any]:
    """Train NN to predict residual over linear covariance predictor. P_pred = P_linear + NN(...)."""
    if MLPRegressor is None:
        raise ImportError("sklearn required. pip install scikit-learn")
    X, y = build_sequences_anchored(train_slices, tau, mu, cov_0, cov_tau, reg=reg)
    if X.size == 0:
        raise ValueError("No training sequences for nn_anchored")
    scaler_X = StandardScaler()
    X_sc = scaler_X.fit_transform(X)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_sc, y)
    return model, scaler_X


def predict_nn_anchored(
    model,
    scaler_X,
    P_t: np.ndarray,
    P_prev: np.ndarray,
    mu: np.ndarray,
    cov_0: np.ndarray,
    cov_tau: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    """P_pred = P(t) * (1 + r_linear + NN_residual), all in return space."""
    P_t = np.asarray(P_t).ravel()
    P_prev = np.asarray(P_prev).ravel()
    r_t = np.where(P_prev != 0, (P_t - P_prev) / P_prev, 0.0)
    p_linear = _linear_pred(P_t, mu, cov_0, cov_tau, reg=reg)
    r_linear = np.where(P_t != 0, (p_linear - P_t) / P_t, 0.0)
    x = np.concatenate([r_t, r_linear]).reshape(1, -1)
    x_sc = scaler_X.transform(x)
    r_residual = model.predict(x_sc).ravel()
    r_pred = r_linear + r_residual
    return (P_t * (1.0 + r_pred)).ravel()


def train_nn_pairs(
    train_slices: List[pd.DataFrame],
    tau: int,
    mu: np.ndarray,
    hidden_layer_sizes: tuple = (256, 128, 64),
    max_iter: int = 400,
    random_state: int = 0,
) -> Tuple[Any, Any, Any]:
    """Train NN on pairwise features [d; upper(d d')]. Output = P(t+τ)."""
    if MLPRegressor is None:
        raise ImportError("sklearn required. pip install scikit-learn")
    X, y = build_sequences_pairs(train_slices, tau, mu)
    if X.size == 0:
        raise ValueError("No training sequences for nn_pairs")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_sc, y_sc)
    return model, scaler_X, scaler_y


def predict_nn_pairs(
    model,
    scaler_X,
    scaler_y,
    P_t: np.ndarray,
    P_prev: np.ndarray,
) -> np.ndarray:
    """Predict from pairwise features of r(t). Returns price: P(t) * (1 + r_pred)."""
    P_t = np.asarray(P_t).ravel()
    P_prev = np.asarray(P_prev).ravel()
    r_t = np.where(P_prev != 0, (P_t - P_prev) / P_prev, 0.0)
    x = _pairwise_features(r_t).reshape(1, -1)
    x_sc = scaler_X.transform(x)
    y_sc = model.predict(x_sc)
    r_pred = scaler_y.inverse_transform(y_sc).ravel()
    return (P_t * (1.0 + r_pred)).ravel()


def train_nn_predictor(
    train_slices: List[pd.DataFrame],
    tau: int,
    predict_returns: bool = False,
    hidden_layer_sizes: tuple = (1024,512,256,128, 64),
    max_iter: int = 500,
    random_state: int = 0,
) -> tuple:
    """
    Train a single MLP: input N (current prices or returns), output N (future prices or returns).
    Returns (fitted_model, scaler_X, scaler_y) for price prediction; scaler_y is None if predict_returns.
    """
    if MLPRegressor is None:
        raise ImportError("sklearn is required for --predictor nn. Install with: pip install scikit-learn")
    X, y = build_sequences(train_slices, tau, predict_returns=predict_returns)
    if X.size == 0:
        raise ValueError("No training sequences: train slices too short for tau")
    N = X.shape[1]
    scaler_X = StandardScaler()
    X_sc = scaler_X.fit_transform(X)
    scaler_y = None if predict_returns else StandardScaler()
    if scaler_y is not None:
        y_sc = scaler_y.fit_transform(y)
    else:
        y_sc = y
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_sc, y_sc)
    return model, scaler_X, scaler_y


def predict_nn(
    model,
    scaler_X,
    scaler_y,
    P_t: np.ndarray,
    predict_returns: bool,
) -> np.ndarray:
    """Single-step prediction: P(t) -> P(t+τ) or return at t+τ."""
    x = np.asarray(P_t).ravel().reshape(1, -1)
    x_sc = scaler_X.transform(x)
    y_sc = model.predict(x_sc)
    if scaler_y is not None:
        y = scaler_y.inverse_transform(y_sc)
    else:
        y = y_sc
    if predict_returns:
        # y is predicted return; convert to price: P_pred = P(t) * (1 + r_pred)
        P_t = np.asarray(P_t).ravel()
        return (P_t * (1 + y.ravel())).ravel()
    return y.ravel()


def _build_sequences_pair(
    slices: List[pd.DataFrame],
    tau: int,
    i: int,
    j: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """For pair (i,j): X = (r_i(t), r_j(t)), y = (r_i(t+τ), r_j(t+τ)) in return space."""
    X_list, y_list = [], []
    for df in slices:
        P = df.values.astype(float)
        T, N = P.shape
        if T <= tau + 1 or i >= N or j >= N:
            continue
        for t in range(1, T - tau):
            ri = (P[t, i] - P[t-1, i]) / P[t-1, i] if P[t-1, i] != 0 else 0.0
            rj = (P[t, j] - P[t-1, j]) / P[t-1, j] if P[t-1, j] != 0 else 0.0
            ri_fut = (P[t+tau, i] - P[t+tau-1, i]) / P[t+tau-1, i] if P[t+tau-1, i] != 0 else 0.0
            rj_fut = (P[t+tau, j] - P[t+tau-1, j]) / P[t+tau-1, j] if P[t+tau-1, j] != 0 else 0.0
            X_list.append([ri, rj])
            y_list.append([ri_fut, rj_fut])
    if not X_list:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.array(X_list), np.array(y_list)


def train_nn_nchoose2(
    train_slices: List[pd.DataFrame],
    tau: int,
    hidden_layer_sizes: tuple = (16, 8),
    max_iter: int = 1000,
    random_state: int = 0,
) -> List[Tuple[int, int, Any, Any, Any]]:
    """
    Train one small MLP per pair (i,j) with i < j. Each: input (P_i(t), P_j(t)), output (P_i(t+τ), P_j(t+τ)).
    Returns list of (i, j, model, scaler_X, scaler_y) for i < j.
    """
    if MLPRegressor is None:
        raise ImportError("sklearn required. pip install scikit-learn")
    N = train_slices[0].shape[1]
    pair_models: List[Tuple[int, int, Any, Any, Any]] = []
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    import warnings
    for (i, j) in tqdm(pairs, desc="Training pair models", unit="pair"):
        X, y = _build_sequences_pair(train_slices, tau, i, j)
        if len(X) < 10:
            continue
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_sc = scaler_X.fit_transform(X)
        y_sc = scaler_y.fit_transform(y)
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sc, y_sc)
        pair_models.append((i, j, model, scaler_X, scaler_y))
    return pair_models


def predict_nn_nchoose2(
    pair_models: List[Tuple[int, int, Any, Any, Any]],
    P_t: np.ndarray,
    P_prev: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    For each stock k, average the predicted return from all pairs (i,j) that contain k.
    Input: current returns r_i(t) = (P(t)-P(t-1))/P(t-1).
    Output: predicted returns, converted back to price: P_pred = P(t) * (1 + r_pred).
    """
    P_t = np.asarray(P_t).ravel()[:N]
    P_prev = np.asarray(P_prev).ravel()[:N]
    r_t = np.where(P_prev != 0, (P_t - P_prev) / P_prev, 0.0)
    ret_sum = np.zeros(N)
    ret_count = np.zeros(N)
    for (i, j, model, scaler_X, scaler_y) in pair_models:
        x = np.array([[r_t[i], r_t[j]]])
        x_sc = scaler_X.transform(x)
        y_sc = model.predict(x_sc)
        y = scaler_y.inverse_transform(y_sc).ravel()
        ret_sum[i] += y[0]
        ret_count[i] += 1
        ret_sum[j] += y[1]
        ret_count[j] += 1
    ret_count = np.where(ret_count > 0, ret_count, 1)
    r_pred = ret_sum / ret_count
    return (P_t * (1.0 + r_pred)).ravel()


# ── Multi-step trajectory predictor ──────────────────────────────────


def build_sequences_multistep(
    slices: List[pd.DataFrame],
    W: int,
    H: int,
    max_samples: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training data for multi-step trajectory prediction.

    Input X: flattened window of W past returns for all N stocks -> (W*N,)
    Output Y: cumulative returns for next H steps for all N stocks -> (H*N,)
        cum_ret(t+k) = (P(t+k) - P(t)) / P(t)

    Returns (X, Y) arrays.
    """
    from correlation_and_predict import prices_to_returns

    N = slices[0].shape[1]
    T0 = slices[0].shape[0]
    samples_per_slice = max(1, T0 - W - H)
    total_est = len(slices) * samples_per_slice

    rng = np.random.RandomState(42)
    if total_est > max_samples:
        keep_ratio = max_samples / total_est
        n_keep = max(1, int(len(slices) * keep_ratio))
        slice_idx = np.sort(rng.choice(len(slices), n_keep, replace=False))
        use_slices = [slices[i] for i in slice_idx]
        print(f"  Subsampling {n_keep}/{len(slices)} slices (~{n_keep * samples_per_slice} samples) to stay under {max_samples}")
    else:
        use_slices = slices

    all_X, all_Y = [], []
    for df in tqdm(use_slices, desc="  NN sequences", unit="slice"):
        P = df.values.astype(float)
        T = P.shape[0]
        if T < W + H + 1:
            continue
        R = prices_to_returns(P)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        for t in range(W, T - H):
            x = R[t - W:t][::-1].ravel()
            P_t = P[t]
            safe = np.where(P_t != 0, P_t, 1.0)
            cum_ret = np.array([(P[t + k] - P_t) / safe for k in range(1, H + 1)])
            y = cum_ret.ravel()
            all_X.append(x)
            all_Y.append(y)

    X = np.array(all_X)
    Y = np.array(all_Y)
    return X, Y


def train_nn_multistep(
    train_slices: List[pd.DataFrame],
    W: int = 30,
    H: int = 10,
    hidden_layer_sizes: tuple = (512, 256, 128),
    max_iter: int = 50,
    max_samples: int = 500_000,
    random_state: int = 0,
) -> Tuple[Any, Any, Any, int, int]:
    """
    Train MLP to predict next H cumulative returns from past W returns.

    Returns (model, scaler_X, scaler_Y, W, H).
    """
    if MLPRegressor is None:
        raise ImportError("scikit-learn required: pip install scikit-learn")

    print(f"  Building multi-step sequences (W={W}, H={H})...")
    X, Y = build_sequences_multistep(train_slices, W, H, max_samples=max_samples)
    print(f"  Training data: {X.shape[0]} samples, input dim={X.shape[1]}, output dim={Y.shape[1]}")

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_sc = scaler_X.fit_transform(X)
    Y_sc = scaler_Y.fit_transform(Y)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=True,
    )
    print(f"  Training MLP {X.shape[1]} -> {hidden_layer_sizes} -> {Y.shape[1]}...")
    model.fit(X_sc, Y_sc)
    best = model.best_loss_ if model.best_loss_ is not None else model.loss_
    print(f"  NN trained: {model.n_iter_} iterations, final loss={best:.6f}")

    return model, scaler_X, scaler_Y, W, H


def predict_nn_multistep(
    model,
    scaler_X,
    scaler_Y,
    R_window: np.ndarray,
    P_t: np.ndarray,
    H: int,
) -> np.ndarray:
    """
    Predict next H prices from a window of past returns.

    R_window: (W, N) array of returns, most recent first (row 0 = r(t), row 1 = r(t-1), ...).
    P_t: (N,) current prices.
    H: prediction horizon.

    Returns (H, N) array of predicted prices [P(t+1), ..., P(t+H)].
    """
    N = len(P_t)
    x = R_window.ravel().reshape(1, -1)
    x_sc = scaler_X.transform(x)
    y_sc = model.predict(x_sc)
    cum_ret = scaler_Y.inverse_transform(y_sc).ravel().reshape(H, N)
    P_t = np.asarray(P_t).ravel()[:N]
    return P_t[np.newaxis, :] * (1.0 + cum_ret)


# ── PCA Factor Transformer ──────────────────────────────────────────

try:
    import torch
    import torch.nn as nn_torch
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True

    def _get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        return torch.device("cpu")
except ImportError:
    _HAS_TORCH = False
    _get_device = None


class FactorTransformerModel(nn_torch.Module if _HAS_TORCH else object):
    """Small Transformer encoder that maps (W, K) factor sequence → (H, K) future factors."""

    def __init__(self, K: int, H: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.H = H
        self.d_model = d_model
        self.input_proj = nn_torch.Linear(K, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout, max_len=512)
        encoder_layer = nn_torch.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn_torch.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn_torch.Linear(d_model, H * K)

    def forward(self, x):
        # x: (batch, W, K)
        x = self.input_proj(x)       # (batch, W, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)          # (batch, W, d_model)
        x = x[:, -1, :]              # take last timestep (batch, d_model)
        out = self.head(x)            # (batch, H*K)
        return out.view(-1, self.H, self.K)


class _PositionalEncoding(nn_torch.Module if _HAS_TORCH else object):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn_torch.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def _compute_pca(train_slices: List[pd.DataFrame], K: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PCA basis from training returns. Returns (V, mu) where V is (N, K)."""
    from correlation_and_predict import prices_to_returns
    all_r = []
    for df in train_slices[::max(1, len(train_slices) // 2000)]:
        P = df.values.astype(float)
        R = prices_to_returns(P)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        all_r.append(R)
    R_all = np.vstack(all_r)
    mu = R_all.mean(axis=0)
    R_centered = R_all - mu
    cov = (R_centered.T @ R_centered) / max(len(R_centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:K]
    V = eigvecs[:, idx]  # (N, K)
    explained = eigvals[idx].sum() / max(eigvals.sum(), 1e-12)
    print(f"  PCA: top {K} factors explain {explained*100:.1f}% of variance")
    return V, mu


def _build_factor_sequences(
    train_slices: List[pd.DataFrame],
    V: np.ndarray, mu_r: np.ndarray,
    W: int, H: int, max_samples: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) in factor space. X: (n, W, K), Y: (n, H, K) cumulative factor returns."""
    from correlation_and_predict import prices_to_returns
    K = V.shape[1]
    T0 = train_slices[0].shape[0]
    samples_per_slice = max(1, T0 - W - H)
    total_est = len(train_slices) * samples_per_slice

    rng = np.random.RandomState(42)
    if total_est > max_samples:
        keep_ratio = max_samples / total_est
        n_keep = max(1, int(len(train_slices) * keep_ratio))
        slice_idx = np.sort(rng.choice(len(train_slices), n_keep, replace=False))
        use_slices = [train_slices[i] for i in slice_idx]
        print(f"  Subsampling {n_keep}/{len(train_slices)} slices for factor sequences")
    else:
        use_slices = train_slices

    all_X, all_Y = [], []
    for df in tqdm(use_slices, desc="  Factor seqs", unit="slice"):
        P = df.values.astype(float)
        T = P.shape[0]
        if T < W + H + 1:
            continue
        R = prices_to_returns(P)
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        F = (R - mu_r) @ V  # (T, K)

        for t in range(W, T - H):
            x = F[t - W:t]  # (W, K) chronological order
            # Cumulative factor sum for horizon
            f_t = F[t]
            y = np.cumsum(F[t + 1:t + H + 1], axis=0) - np.cumsum(np.zeros_like(F[t:t+1]), axis=0)
            # Actually: cumulative factor return from t
            y = np.array([F[t + 1:t + k + 1].sum(axis=0) for k in range(1, H + 1)])
            all_X.append(x)
            all_Y.append(y)

    return np.array(all_X), np.array(all_Y)


def _build_price_ratio_sequences(
    train_slices: List[pd.DataFrame],
    W: int,
    H: int,
    max_samples: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X, Y) in price-ratio space. X: (n, W, N) = P(t-W:t)/P(t), Y: (n, H, N) = P(t+1:t+H+1)/P(t)."""
    N = train_slices[0].shape[1]
    T0 = train_slices[0].shape[0]
    samples_per_slice = max(1, T0 - W - H)
    total_est = len(train_slices) * samples_per_slice

    rng = np.random.RandomState(42)
    if total_est > max_samples:
        keep_ratio = max_samples / total_est
        n_keep = max(1, int(len(train_slices) * keep_ratio))
        slice_idx = np.sort(rng.choice(len(train_slices), n_keep, replace=False))
        use_slices = [train_slices[i] for i in slice_idx]
        print(f"  Subsampling {n_keep}/{len(train_slices)} slices for price-ratio sequences")
    else:
        use_slices = train_slices

    all_X, all_Y = [], []
    for df in tqdm(use_slices, desc="  Price ratio seqs", unit="slice"):
        P = df.values.astype(float)
        T = P.shape[0]
        if T < W + H + 1:
            continue
        P_safe = np.where(P != 0, P, np.nan)
        for t in range(W, T - H):
            p_t = P_safe[t]
            denom = np.where(np.isfinite(p_t) & (p_t != 0), p_t, 1.0)
            x = P_safe[t - W:t] / denom  # (W, N)
            y = P_safe[t + 1:t + H + 1] / denom  # (H, N)
            x = np.nan_to_num(x, nan=1.0, posinf=1.0, neginf=1.0)
            y = np.nan_to_num(y, nan=1.0, posinf=1.0, neginf=1.0)
            all_X.append(x)
            all_Y.append(y)

    return np.array(all_X), np.array(all_Y)


def train_factor_transformer(
    train_slices: List[pd.DataFrame],
    W: int = 30,
    H: int = 10,
    K: int = 10,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 1e-3,
    max_samples: int = 500_000,
    use_price_space: bool = False,
) -> dict:
    """
    Train Transformer: either PCA factor space (use_price_space=False) or price-ratio space (use_price_space=True).
    Returns dict with model and all needed for prediction.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")

    N = train_slices[0].shape[1]
    if use_price_space:
        # Use fewer samples and smaller model to avoid OOM (price-space has K=N=50)
        price_max_samples = min(max_samples, 150_000)
        d_model = 64
        n_layers = 2
        n_heads = 4
        print(f"  Building price-ratio sequences (W={W}, H={H}) in price space...")
        X, Y = _build_price_ratio_sequences(train_slices, W, H, max_samples=price_max_samples)
        K = N  # model operates on N stocks directly
        V, mu_r = None, None
        print(f"  Training data: {X.shape[0]} samples, input ({W},{N}), output ({H},{N})")
    else:
        print(f"  Computing PCA (K={K} factors)...")
        V, mu_r = _compute_pca(train_slices, K)
        print(f"  Building factor sequences (W={W}, H={H})...")
        X, Y = _build_factor_sequences(train_slices, V, mu_r, W, H, max_samples=max_samples)
        print(f"  Training data: {X.shape[0]} samples, input ({W},{K}), output ({H},{K})")

    n_samples = len(X)
    n_val = max(1, int(n_samples * 0.1))
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    if StandardScaler is not None:
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        scaler_X.fit(X_train.reshape(len(X_train), -1))
        scaler_Y.fit(Y_train.reshape(len(Y_train), -1))
        X = scaler_X.transform(X.reshape(n_samples, -1)).reshape(n_samples, W, K)
        Y = scaler_Y.transform(Y.reshape(n_samples, -1)).reshape(n_samples, H, K)
        print(f"  Standardized inputs and targets (per feature)")
    else:
        scaler_X = scaler_Y = None

    device = _get_device()
    print(f"  Using device: {device}")

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    X_train_t, Y_train_t = X_t[train_idx], Y_t[train_idx]
    X_val = X_t[val_idx].to(device)
    Y_val = Y_t[val_idx].to(device)

    train_ds = TensorDataset(X_train_t, Y_train_t)
    effective_batch = batch_size if not use_price_space else min(batch_size, 512)
    train_dl = DataLoader(train_ds, batch_size=effective_batch, shuffle=True)

    model = FactorTransformerModel(
        K=K, H=H, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.2
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn_torch.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Transformer: {n_params:,} parameters, training {epochs} epochs...")

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()
            ss_res = ((val_pred - Y_val) ** 2).sum().item()
            ss_tot = ((Y_val - Y_val.mean(dim=0)) ** 2).sum().item()
            val_r2 = 1 - ss_res / max(ss_tot, 1e-12)

        avg_train = train_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:2d}/{epochs}: train_loss={avg_train:.6f}, val_loss={val_loss:.6f}, val_R²={val_r2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  Factor Transformer trained, best val_loss={best_val_loss:.6f}")

    return {
        "model": model, "V": V, "mu_r": mu_r,
        "W": W, "H": H, "K": K, "device": device,
        "scaler_X": scaler_X, "scaler_Y": scaler_Y,
        "use_price_space": use_price_space,
    }


def save_factor_transformer(ft_dict: dict, path_dir: Any) -> None:
    """Save FT model and extras to path_dir (Path or str)."""
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(ft_dict["model"].state_dict(), path_dir / "ft_state.pt")
    extra = {
        "V": ft_dict["V"],
        "mu_r": ft_dict["mu_r"],
        "scaler_X": ft_dict["scaler_X"],
        "scaler_Y": ft_dict["scaler_Y"],
        "W": ft_dict["W"],
        "H": ft_dict["H"],
        "K": ft_dict["K"],
        "use_price_space": ft_dict["use_price_space"],
    }
    if ft_dict["V"] is not None:
        import pickle
        with open(path_dir / "ft_extra.pkl", "wb") as f:
            pickle.dump(extra, f)
    else:
        # price space: V is None, still save extra
        import pickle
        with open(path_dir / "ft_extra.pkl", "wb") as f:
            pickle.dump(extra, f)


def load_factor_transformer(path_dir: Any) -> dict:
    """Load FT model and extras from path_dir. Returns ft_dict ready for predict_factor_transformer."""
    path_dir = Path(path_dir)
    import pickle
    import torch
    with open(path_dir / "ft_extra.pkl", "rb") as f:
        extra = pickle.load(f)
    K = extra["K"]
    H = extra["H"]
    d_model = 128
    n_heads = 8
    n_layers = 4
    if extra.get("use_price_space"):
        d_model, n_heads, n_layers = 64, 4, 2
    model = FactorTransformerModel(K=K, H=H, d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=0.2)
    state = torch.load(path_dir / "ft_state.pt", map_location="cpu")
    model.load_state_dict(state)
    device = _get_device() if callable(_get_device) else torch.device("cpu")
    model = model.to(device)
    model.eval()
    return {
        "model": model,
        "V": extra["V"],
        "mu_r": extra["mu_r"],
        "W": extra["W"],
        "H": extra["H"],
        "K": extra["K"],
        "device": device,
        "scaler_X": extra["scaler_X"],
        "scaler_Y": extra["scaler_Y"],
        "use_price_space": extra["use_price_space"],
    }


def predict_factor_transformer(
    ft_dict: dict,
    R_window: np.ndarray = None,
    P_t: np.ndarray = None,
    P_window: np.ndarray = None,
) -> np.ndarray:
    """
    Predict next H prices using the Transformer.

    Factor space (use_price_space=False): pass R_window (W, N), P_t (N,).
    Price space (use_price_space=True): pass P_window (W+1, N) and P_t (N,). P_window[-1] must be P_t.

    Returns (H, N) predicted prices.
    """
    model = ft_dict["model"]
    H = ft_dict["H"]
    K = ft_dict["K"]
    W = ft_dict["W"]
    use_price_space = ft_dict.get("use_price_space", False)

    if use_price_space:
        if P_window is None or P_t is None:
            raise ValueError("use_price_space=True requires P_window and P_t")
        N = len(P_t)
        P_window = np.nan_to_num(P_window, nan=1.0, posinf=1.0, neginf=1.0)
        P_t = np.asarray(P_t).ravel()[:N]
        denom = np.where(P_t != 0, P_t, 1.0)
        # input: P(t-W:t) / P(t) -> (W, N); P_window is (W+1, N) with last row = P(t)
        x_ratios = P_window[:-1] / denom  # (W, N)
        scaler_X = ft_dict.get("scaler_X")
        scaler_Y = ft_dict.get("scaler_Y")
        if scaler_X is not None:
            x_flat = scaler_X.transform(x_ratios.reshape(1, -1)).reshape(1, W, N)
        else:
            x_flat = x_ratios.reshape(1, W, N)
        dev = ft_dict.get("device", torch.device("cpu"))
        x = torch.tensor(x_flat, dtype=torch.float32, device=dev)
        with torch.no_grad():
            ratio_pred = model(x).squeeze(0).cpu().numpy()  # (H, N)
        if scaler_Y is not None:
            ratio_pred = scaler_Y.inverse_transform(ratio_pred.reshape(1, -1)).reshape(H, N)
        ratio_pred = np.nan_to_num(ratio_pred, nan=1.0, posinf=1.0, neginf=1.0)
        return P_t[np.newaxis, :] * ratio_pred
    else:
        if R_window is None or P_t is None:
            raise ValueError("Factor space requires R_window and P_t")
        V = ft_dict["V"]
        mu_r = ft_dict["mu_r"]
        N = len(P_t)
        R_window = np.nan_to_num(R_window, nan=0.0, posinf=0.0, neginf=0.0)
        F_window = (R_window - mu_r) @ V  # (W, K)
        scaler_X = ft_dict.get("scaler_X")
        scaler_Y = ft_dict.get("scaler_Y")
        if scaler_X is not None:
            F_flat = scaler_X.transform(F_window.reshape(1, -1)).reshape(1, W, K)
        else:
            F_flat = F_window.reshape(1, W, K)
        dev = ft_dict.get("device", torch.device("cpu"))
        x = torch.tensor(F_flat, dtype=torch.float32, device=dev)
        with torch.no_grad():
            f_pred = model(x).squeeze(0).cpu().numpy()  # (H, K)
        if scaler_Y is not None:
            f_pred = scaler_Y.inverse_transform(f_pred.reshape(1, -1)).reshape(H, K)
        cum_stock_ret = f_pred @ V.T  # (H, N)
        P_t = np.asarray(P_t).ravel()[:N]
        return P_t[np.newaxis, :] * (1.0 + cum_stock_ret)


# ── Per-stock 1h-window predictor (50 models, returns space) ───────────

def _prices_to_returns(P: np.ndarray) -> np.ndarray:
    """(T, N) prices -> (T, N) returns; R[0]=0, R[t] = (P[t]-P[t-1])/P[t-1]."""
    R = np.zeros_like(P)
    R[1:] = np.where(P[:-1] != 0, (P[1:] - P[:-1]) / P[:-1], 0.0)
    return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)


def _build_per_stock_1h_sequences(
    prices: np.ndarray,
    W: int = 60,
    max_samples_per_stock: int = 100_000,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    In returns space: X_i = [flatten(60 min of other 49 stocks' returns), current return i], Y_i = next minute return i.
    prices: (T, N) array. Returns (list of X arrays length N, list of Y arrays length N).
    """
    T, N = prices.shape
    P = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
    R = _prices_to_returns(P)
    others_idx = [np.array([j for j in range(N) if j != i]) for i in range(N)]

    X_list = [[] for _ in range(N)]
    Y_list = [[] for _ in range(N)]

    for t in range(W, T - 1):
        for i in range(N):
            window_others = R[t - W + 1:t + 1, others_idx[i]]  # (W, N-1)
            current_ret_i = R[t, i]
            x = np.concatenate([window_others.ravel(), [current_ret_i]])
            y = R[t + 1, i]
            X_list[i].append(x)
            Y_list[i].append(y)

    rng = np.random.RandomState(42)
    X_out, Y_out = [], []
    for i in range(N):
        Xi = np.array(X_list[i])
        Yi = np.array(Y_list[i])
        if len(Xi) > max_samples_per_stock:
            idx = rng.choice(len(Xi), max_samples_per_stock, replace=False)
            Xi, Yi = Xi[idx], Yi[idx]
        X_out.append(Xi)
        Y_out.append(Yi)
    return X_out, Y_out


def train_per_stock_1h(
    train_slices: List[pd.DataFrame],
    W: int = 60,
    max_samples_per_stock: int = 100_000,
    hidden_layer_sizes: tuple = (256, 128),
    max_iter: int = 100,
) -> List[Tuple[Any, Any, Any]]:
    """
    Train one NN per stock in returns space. Each NN: input = (49 × W) + 1 (returns), output = 1 (next minute return).
    Returns list of (model, scaler_X, scaler_y) for each of N stocks.
    """
    if MLPRegressor is None or StandardScaler is None:
        raise ImportError("scikit-learn required")

    blocks = [s.values.astype(float) for s in train_slices]
    blocks = [b for b in blocks if b.shape[0] >= W + 2]
    if not blocks:
        raise ValueError("No slices with enough rows for W+2")
    prices = np.vstack(blocks)
    T, N = prices.shape
    print(f"  Per-stock 1h (returns): building sequences (W={W}, N={N}), total bars={T}")

    X_list, Y_list = _build_per_stock_1h_sequences(prices, W=W, max_samples_per_stock=max_samples_per_stock)

    models = []
    for i in tqdm(range(N), desc="  Training 50 stock models", unit="stock"):
        Xi, Yi = X_list[i], Y_list[i]
        if len(Xi) < 10:
            models.append((None, None, None))
            continue
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        Xi_sc = scaler_X.fit_transform(Xi)
        Yi_sc = scaler_y.fit_transform(Yi.reshape(-1, 1)).ravel()
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        model.fit(Xi_sc, Yi_sc)
        models.append((model, scaler_X, scaler_y))
    print(f"  Per-stock 1h: trained {sum(1 for m in models if m[0] is not None)} models")
    return models


def predict_per_stock_1h_one(
    models: List[Tuple[Any, Any, Any]],
    R_window: np.ndarray,
    target_stock_idx: int,
    current_return_target: float,
) -> float:
    """
    Returns space: R_window (W, N), current_return_target = return of target stock at current time.
    Returns predicted next-minute return for target stock.
    """
    N = R_window.shape[1]
    others_idx = np.array([j for j in range(N) if j != target_stock_idx])
    window_others = R_window[:, others_idx]  # (W, N-1)
    x = np.concatenate([window_others.ravel(), [current_return_target]])
    model, scaler_X, scaler_y = models[target_stock_idx]
    if model is None:
        return current_return_target
    x_sc = scaler_X.transform(x.reshape(1, -1))
    y_sc = model.predict(x_sc)
    pred_ret = scaler_y.inverse_transform(y_sc.reshape(1, -1))[0, 0]
    return float(pred_ret)


def predict_per_stock_1h_propagate(
    models: List[Tuple[Any, Any, Any]],
    P_test: np.ndarray,
    start_t: int,
    target_stock_idx: int,
    H: int = 60,
    W: int = 60,
) -> np.ndarray:
    """
    Propagate using only predicted values after the first step. True data only at start_t.
    At each step we predict all N stocks; that row becomes the next window row (no more true data).
    P_test: (T, N) full test prices. Returns (H,) predicted prices for target at start_t+1..start_t+H.
    """
    if start_t < W:
        raise ValueError("start_t must be >= W so we have a full initial window from true data")
    N = P_test.shape[1]
    # Buffer of last W+1 price rows; from it we get W returns. Initially all true.
    buffer = np.asarray(P_test[start_t - W : start_t + 1], dtype=float)  # (W+1, N)
    pred_prices = np.zeros((H, N))
    for s in range(H):
        # Returns from consecutive buffer rows: R[i] = (buffer[i+1]-buffer[i])/buffer[i]
        P_prev = np.where(buffer[:-1] != 0, buffer[:-1], np.nan)
        R_window = np.where(P_prev != 0, (buffer[1:] - buffer[:-1]) / P_prev, 0.0)  # (W, N)
        R_window = np.nan_to_num(R_window, nan=0.0, posinf=0.0, neginf=0.0)
        current_prices = buffer[-1]  # (N,)
        next_row = np.zeros(N)
        for i in range(N):
            current_ret = float(R_window[-1, i])
            pred_ret = predict_per_stock_1h_one(models, R_window, i, current_ret)
            next_row[i] = current_prices[i] * (1.0 + pred_ret)
        pred_prices[s] = next_row
        # Roll buffer: drop first row, append predicted row
        buffer = np.vstack([buffer[1:], next_row])
    return pred_prices[:, target_stock_idx]
