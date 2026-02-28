# Out-of-time correlation and stock prediction experiment

## Goal

1. **Data**: Load the highest resolution stock prices for many tickers over one year (daily with free data; intraday if you have a longer history source).
2. **Correlation**: Compute the out-of-time correlation  
   **C_AB(τ) = E[P_A(t) P_B(t+τ)]**  
   for pairs (A,B), by splitting the year into equal-sized slices, estimating C_AB(τ) in each slice, then **averaging over slices** to get a stable covariance/correlation structure.
3. **Prediction**: Use the averaged covariance at lag 0 and lag τ plus the averaged mean price to predict **P(t+τ)** from **P(t)** via the optimal linear predictor:  
   **P_pred(t+τ) = μ + Cov(τ)' inv(Cov(0)) (P(t) − μ)**.
4. **Evaluation**: Compare predictions to real data on a held-out slice and report **RMSE, MAE, R², correlation** in both **price level** and **return** space.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_experiment.py [options]
```

Options:

- `--tickers AAPL MSFT ...` — tickers to use (default: ~50 names across sectors).
- `--years 1.0` — years of history.
- `--interval 1d` — bar interval (`1d` for 1 year; `1h` only has ~60 days on Yahoo).
- `--n_slices 12` — number of equal slices in the year.
- `--lag 1` — prediction horizon τ (e.g. 1 = next bar).
- `--reg 1e-5` — ridge regularization when inverting Cov(0).
- `--predictor full|diagonal|returns|nn|nn_anchored|nn_pairs` — **returns** = return-space Cov (default, recommended); full/diagonal = price Cov; nn variants usually worse.
- `--window-length W` — use overlapping windows of W bars (last W bars = test). Often better with 1h data.
- `--step S` — step between overlapping windows (default W/2). Smaller = more overlap.
- `--shrink 1.0` — shrink correction toward mean (0 = always predict μ).

**Recommended run** (1h data, return predictor, many overlapping slices):

```bash
python run_experiment.py --window-length 100 --interval 1h --predictor returns --step 1
```

### Higher resolution and shorter slices

If daily + 12 slices doesn’t predict returns well, you can try:

- **Shorter slices (daily):** more slices → shorter windows, less smoothing.  
  `--interval 1d --n_slices 26` or `--n_slices 52` (e.g. ~2 weeks or ~1 week per slice).
- **Higher resolution (intraday):** 1h bars over the last 60 days (Yahoo limit), with more slices so each slice is short (e.g. ~4 days per slice).  
  `--interval 1h --n_slices 15 --lag 1`  
  (Period is auto-capped to 60 days when using `1h`/`30m`/`15m`.)

Example — hourly data, 15 slices (~4 days per slice):

```bash
python run_experiment.py --interval 1h --n_slices 15 --lag 1
```

Example — daily data, shorter slices (~2 weeks per slice):

```bash
python run_experiment.py --interval 1d --n_slices 26 --lag 1
```

## Interpretation

- **Price-level R²** is often high because prices are persistent; it does not mean the correlation model adds much.
- **Return-space R² and correlation** show how well the slice-averaged correlation structure predicts the *next move*; these are the meaningful measures of predictive success.
- Negative per-stock R² means the linear predictor does worse than predicting the mean for that stock.

## Files

- `data_loader.py` — fetch adjusted close prices (yfinance).
- `correlation_and_predict.py` — slice splitting, C_AB(τ) and Cov_AB(τ), linear predictor, evaluation (level + returns).
- `run_experiment.py` — full pipeline and CLI.
