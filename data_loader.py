"""
Load high-resolution stock price data for many tickers.
Supports two backends:
  - yfinance (default): free, but intraday limited to ~60 days.
  - alpaca: free with API key, minute-level data for years of history.

To use Alpaca: set env vars ALPACA_API_KEY and ALPACA_SECRET_KEY,
or pass source="alpaca" (will read keys from env).
Get free keys at https://app.alpaca.markets/signup
"""
from __future__ import annotations

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


def _yf_download_chunk(tickers, start_str, end_str, interval):
    """Single yfinance download call."""
    import yfinance as yf
    return yf.download(
        tickers,
        start=start_str,
        end=end_str,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )


def _fetch_yfinance(
    tickers: list[str],
    start: datetime,
    end: datetime,
    interval: str,
) -> pd.DataFrame:
    # Yahoo 1m data: max 7 days per request; 2m/5m/15m/30m: max 60 days
    max_days_per_chunk = 7 if interval == "1m" else None
    if max_days_per_chunk and (end - start).days > max_days_per_chunk:
        print(f"  [yfinance] {interval} data: fetching in {max_days_per_chunk}-day chunks...")
        chunks = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=max_days_per_chunk), end)
            chunk = _yf_download_chunk(
                tickers, chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), interval
            )
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)
            chunk_start = chunk_end
        if not chunks:
            raise ValueError(f"No data returned from yfinance for {interval}")
        data = pd.concat(chunks)
        data = data[~data.index.duplicated(keep="first")].sort_index()
    else:
        data = _yf_download_chunk(
            tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval
        )

    if len(tickers) == 1:
        if "Adj Close" in data.columns:
            out = data[["Adj Close"]].copy()
        else:
            out = data[["Close"]].copy()
        out.columns = [tickers[0]]
        return out.dropna(how="all")

    if isinstance(data.columns, pd.MultiIndex):
        parts = []
        for t in tickers:
            try:
                sub = data[t] if t in data.columns.get_level_values(0) else None
                if sub is not None:
                    c = sub["Adj Close"] if "Adj Close" in sub.columns else sub["Close"]
                    parts.append(c.rename(t))
            except (KeyError, TypeError):
                continue
        adj = pd.concat(parts, axis=1) if parts else pd.DataFrame()
        if adj.size == 0:
            try:
                adj = data.xs("Adj Close", axis=1, level=1)
            except (KeyError, ValueError):
                adj = data.xs("Close", axis=1, level=1)
    else:
        adj = data[["Adj Close"]].copy() if "Adj Close" in data.columns else data[["Close"]].copy()
        adj.columns = tickers[: adj.shape[1]]
    return adj.dropna(how="all").ffill().bfill()


def _interval_to_alpaca(interval: str):
    """Convert our interval strings to Alpaca TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    mapping = {
        "1m": TimeFrame(1, TimeFrameUnit.Minute),
        "5m": TimeFrame(5, TimeFrameUnit.Minute),
        "15m": TimeFrame(15, TimeFrameUnit.Minute),
        "30m": TimeFrame(30, TimeFrameUnit.Minute),
        "1h": TimeFrame(1, TimeFrameUnit.Hour),
        "60m": TimeFrame(1, TimeFrameUnit.Hour),
        "1d": TimeFrame(1, TimeFrameUnit.Day),
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported Alpaca interval '{interval}'. Use one of: {list(mapping.keys())}")
    return mapping[interval]


def _fetch_alpaca(
    tickers: list[str],
    start: datetime,
    end: datetime,
    interval: str,
) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.enums import DataFeed

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca requires API keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.\n"
            "Get free keys at https://app.alpaca.markets/signup"
        )

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x

    client = StockHistoricalDataClient(api_key, secret_key)
    timeframe = _interval_to_alpaca(interval)

    parts = []
    for t in tqdm(tickers, desc="Fetching stocks", unit="ticker"):
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[t],
                timeframe=timeframe,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(request)
            df = bars.df
            if df is not None and len(df) > 0:
                sub = df.loc[t]["close"].rename(t) if t in df.index.get_level_values(0) else None
                if sub is not None and len(sub) > 0:
                    parts.append(sub)
        except Exception as e:
            print(f"  Warning: {t} failed: {e}")
            continue
    if not parts:
        raise ValueError(f"No data returned from Alpaca for {tickers}")
    result = pd.concat(parts, axis=1)
    return result.dropna(how="all").ffill().bfill()


def fetch_stock_prices(
    tickers: list[str],
    years: float = 1.0,
    interval: str = "1d",
    end_date: Optional[str] = None,
    source: str = "auto",
) -> pd.DataFrame:
    """
    Fetch close prices for multiple tickers.

    Parameters
    ----------
    tickers : list of ticker symbols
    years : number of years of history
    interval : '1d', '1h', '1m', '5m', '15m', '30m'
    end_date : optional end date YYYY-MM-DD; default is today
    source : 'yfinance', 'alpaca', or 'auto' (uses alpaca if keys are set, else yfinance)

    Returns
    -------
    DataFrame with DatetimeIndex and one column per ticker (close price).
    """
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    lookback_days = years * 365

    if source == "auto":
        has_alpaca_keys = bool(os.environ.get("ALPACA_API_KEY")) and bool(os.environ.get("ALPACA_SECRET_KEY"))
        source = "alpaca" if has_alpaca_keys else "yfinance"

    if source == "alpaca":
        start = end - timedelta(days=lookback_days)
        print(f"  [Alpaca] Fetching {interval} data from {start.date()} to {end.date()}")
        return _fetch_alpaca(tickers, start, end, interval)
    else:
        if interval == "1m" and lookback_days > 7:
            lookback_days = 7
            print(f"  [yfinance] 1m data capped to 7 days (Yahoo's real limit)")
        elif interval in ("2m", "5m", "15m", "30m", "60m", "1h", "90m") and lookback_days > 60:
            lookback_days = 60
            print(f"  [yfinance] Intraday capped to 60 days")
        start = end - timedelta(days=lookback_days)
        return _fetch_yfinance(tickers, start, end, interval)


if __name__ == "__main__":
    # Quick test
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    df = fetch_stock_prices(tickers, years=1.0, interval="1d")
    print(df.shape)
    print(df.tail(3))
