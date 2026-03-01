"""
Hazard-style neural network to predict probability of next rocket strike on Israel.

Data sources (in order of use when available):
- Kaggle: "Rocket alerts in Israel made by Tzeva Adom" (sab30226) – real alert history.
- Pikud Haoref (Oref) API: live alerts for "today" validation.
  See: https://lobehub.com/mcp/leonmelamud-pikud-a-oref-mcp
- Fallback: curated iran_rocket_strikes_israel.csv (April 2024 / June 2025).

Predicts P(strike in next day or next minute) and compares with today's actual alerts from Oref.
Resolution: per minute when Kaggle datetime data is available; otherwise per day.
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import requests
except ImportError:
    requests = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )
except ImportError:
    MLPClassifier = None
    StandardScaler = None

# -----------------------------------------------------------------------------
# Paths & URLs
# -----------------------------------------------------------------------------

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
DATA_PATH = CACHE_DIR / "iran_rocket_strikes_israel.csv"
# Kaggle Tzeva Adom: place downloaded CSV here (e.g. from https://www.kaggle.com/datasets/sab30226/rocket-alerts-in-israel-made-by-tzeva-adom)
KAGGLE_ALERT_NAMES = [
    "rocket_alerts_tzeva_adom.csv",
    "rocket-alerts-in-israel-made-by-tzeva-adom.csv",
    "alerts.csv",
    "red_alert.csv",
]
OREF_ALERTS_URL = "https://www.oref.org.il/WarningMessages/alert/alerts.json"
OREF_HISTORY_URL = "https://www.oref.org.il/WarningMessages/History/AlertsHistory.json"
OREF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Referer": "https://www.oref.org.il/",
    "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest",
}
# GitHub: historical Pikud Haoref alerts (no geo-restriction). https://github.com/dleshem/israel-alerts-data
GITHUB_ALERTS_URL = "https://raw.githubusercontent.com/dleshem/israel-alerts-data/main/israel-alerts.csv"
GITHUB_ALERTS_CACHE = CACHE_DIR / "israel_alerts_github.csv"
# Cap minute-level rows so feature building and training finish in reasonable time
MAX_MINUTE_ROWS = 500_000


def fetch_israel_alerts_github(use_cache: bool = True, max_age_hours: float = 24) -> Optional[pd.DataFrame]:
    """
    Load Israeli rocket alerts from GitHub (dleshem/israel-alerts-data). No geo-restriction.
    Returns DataFrame with columns [datetime, date, strike, strike_count] for daily,
    or raw rows with '_dt' for minute use. Caches to data_cache/israel_alerts_github.csv.
    """
    if use_cache and GITHUB_ALERTS_CACHE.exists():
        try:
            age = datetime.now().timestamp() - GITHUB_ALERTS_CACHE.stat().st_mtime
            if age <= max_age_hours * 3600:
                df = pd.read_csv(GITHUB_ALERTS_CACHE, nrows=500000)
                if _parse_github_alerts(df) is not None:
                    return _parse_github_alerts(df)
        except Exception:
            pass
    if requests is None:
        if GITHUB_ALERTS_CACHE.exists():
            df = pd.read_csv(GITHUB_ALERTS_CACHE, nrows=500000)
            return _parse_github_alerts(df)
        return None
    try:
        r = requests.get(GITHUB_ALERTS_URL, timeout=30, stream=True)
        r.raise_for_status()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        total = int(r.headers.get("Content-Length", 0)) or None
        chunks = []
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc="Download alerts", leave=False)
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                pbar.update(len(chunk))
        pbar.close()
        GITHUB_ALERTS_CACHE.write_bytes(b"".join(chunks))
        df = pd.read_csv(GITHUB_ALERTS_CACHE, nrows=500000)
        return _parse_github_alerts(df)
    except Exception:
        if GITHUB_ALERTS_CACHE.exists():
            try:
                df = pd.read_csv(GITHUB_ALERTS_CACHE, nrows=500000)
                return _parse_github_alerts(df)
            except Exception:
                pass
        return None


def _parse_github_alerts(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Parse dleshem CSV: expect alertDate (ISO) or date+time. Return daily [date, strike, strike_count]."""
    if df is None or df.empty:
        return None
    # dleshem/israel-alerts-data: data,date,time,alertDate,category,category_desc,matrix_id,rid
    date_col = None
    for c in df.columns:
        c_lower = str(c).strip().lower()
        if c_lower == "alertdate":  # prefer ISO column
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            c_lower = str(c).strip().lower()
            if c_lower in ("alert_date", "datetime", "date", "time"):
                date_col = c
                break
    if date_col is None:
        return None
    df = df.copy()
    if str(date_col).lower() == "alertdate":
        df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["_dt"] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["_dt"])
    if df.empty:
        return None
    df["_date"] = df["_dt"].dt.normalize()
    daily = df.groupby("_date", as_index=False).size()
    daily.columns = ["date", "strike_count"]
    daily["strike"] = (daily["strike_count"] > 0).astype(int)
    return daily[["date", "strike", "strike_count"]]


def _read_csv_tail(path: Path, max_rows: int, encoding: str = "utf-8") -> Optional[pd.DataFrame]:
    """Read only the last max_rows data rows from a CSV (keeps header). Use for low-memory recent-only load."""
    if not path.exists():
        return None
    try:
        with open(path, encoding=encoding) as f:
            f.readline()  # header
            n = sum(1 for _ in f)
        skip = max(0, n - max_rows)
        return pd.read_csv(path, encoding=encoding, skiprows=range(1, skip + 1), nrows=max_rows)
    except Exception:
        return None


TZEVAADOM_HISTORY_URL = "https://api.tzevaadom.co.il/alerts-history"


def _fetch_oref_history_strike_minutes() -> set:
    """Fetch AlertsHistory.json from Oref and return set of minute-floored timestamps
    for rocket alerts (matrix_id=1). Returns empty set on failure (geo-restricted / network error).
    This endpoint is updated within seconds of an alert — far more current than the GitHub CSV.
    Only works from Israeli IPs; use _fetch_tzevaadom_strike_minutes() as global fallback."""
    if requests is None:
        return set()
    try:
        r = requests.get(OREF_HISTORY_URL, headers=OREF_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    minutes = set()
    for alert in data:
        if not isinstance(alert, dict):
            continue
        # Filter to rockets only (matrix_id=1); if field missing, include the alert
        mid = alert.get("matrix_id") or alert.get("category")
        if mid is not None:
            try:
                if int(mid) != 1:
                    continue
            except (TypeError, ValueError):
                pass
        for key in ("alertDate", "alert_date", "alertdate", "datetime", "date"):
            val = alert.get(key)
            if val:
                try:
                    minutes.add(pd.to_datetime(val).floor("min"))
                    break
                except Exception:
                    pass
    return minutes


_ISRAEL_TZ_OFFSET = pd.Timedelta(hours=2)  # Israel = UTC+2 (winter); timeline datetimes are Israeli local (tz-naive)


def _now_israeli_time() -> pd.Timestamp:
    """Return current time as Israeli local (tz-naive), matching GitHub CSV timezone.
    Uses zoneinfo (Python 3.9+) for DST-aware conversion; falls back to UTC+2 offset.
    IMPORTANT: pd.Timestamp.now() returns server local time, which is UTC on Render (US servers).
    The GitHub CSV uses Israeli local time, so we must always convert explicitly."""
    try:
        from zoneinfo import ZoneInfo
        return pd.Timestamp.now(tz=ZoneInfo("Asia/Jerusalem")).tz_localize(None)
    except Exception:
        return pd.Timestamp.utcnow() + _ISRAEL_TZ_OFFSET


def _fetch_tzevaadom_strike_minutes() -> set:
    """Fetch recent alerts from api.tzevaadom.co.il/alerts-history (globally accessible, no geo-restriction).
    Returns set of minute-floored ISRAELI-TIME (tz-naive) timestamps for non-drill alerts.
    Response: list of {id, alerts: [{time (unix/UTC), cities, threat, isDrill}]}
    Converts UTC → Israeli local time to match the GitHub CSV timeline timezone."""
    if requests is None:
        return set()
    try:
        r = requests.get(TZEVAADOM_HISTORY_URL, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return set()
    if not isinstance(data, list):
        return set()
    minutes = set()
    for group in data:
        if not isinstance(group, dict):
            continue
        for alert in group.get("alerts", []):
            if not isinstance(alert, dict):
                continue
            if alert.get("isDrill", False):
                continue
            ts = alert.get("time")
            if ts is not None:
                try:
                    # Convert UTC unix → Israeli local time (tz-naive, matching GitHub CSV)
                    il_time = (pd.Timestamp(int(ts), unit="s") + _ISRAEL_TZ_OFFSET).floor("min")
                    minutes.add(il_time)
                except Exception:
                    pass
    return minutes


def _supplement_with_oref_history(timeline: pd.DataFrame) -> pd.DataFrame:
    """Fill the gap between the last GitHub CSV row and now using real-time alert sources.
    Only adds minutes AFTER the last GitHub CSV timestamp — never overwrites historical data.
    Sources (tried in order):
      1. Oref AlertsHistory.json — seconds-fresh, Israeli IPs only
      2. tzevaadom.co.il alerts-history — seconds-fresh, globally accessible"""
    last_ts = timeline["datetime"].max()
    now_floor = _now_israeli_time().floor("min")
    if pd.isna(last_ts) or last_ts >= now_floor:
        return timeline

    # Try Oref first (Israeli IPs); fall back to tzevaadom (global)
    gap_minutes = _fetch_oref_history_strike_minutes()
    source = "Oref"
    if not gap_minutes:
        gap_minutes = _fetch_tzevaadom_strike_minutes()
        source = "tzevaadom"
    if not gap_minutes:
        return timeline

    # Keep only minutes strictly inside the gap (after last GitHub row, up to now)
    strike_in_gap = {m for m in gap_minutes if last_ts < m <= now_floor}

    # Build a full minute range for the gap, marking strikes
    gap_range = pd.date_range(start=last_ts + pd.Timedelta(minutes=1), end=now_floor, freq="min")
    if len(gap_range) == 0:
        return timeline
    gap_df = pd.DataFrame({"datetime": gap_range})
    gap_df["strike"] = gap_df["datetime"].isin(strike_in_gap).astype(int)
    gap_df["strike_count"] = gap_df["strike"]
    if "prepare_alert" in timeline.columns:
        gap_df["prepare_alert"] = 0

    n_strikes = int(gap_df["strike"].sum())
    if n_strikes > 0:
        print(f"  {source} history supplement: {n_strikes} strike minute(s) in {len(gap_range)}-min gap after GitHub CSV.", flush=True)

    return pd.concat([timeline, gap_df], ignore_index=True).sort_values("datetime").reset_index(drop=True)


def _fetch_github_alerts_minute(
    force_refresh: bool = False,
    max_cache_age_minutes: Optional[float] = 5,
    max_rows: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Fetch GitHub alerts and return minute-level timeline [datetime, strike, strike_count].
    If max_rows is set, only the last max_rows CSV rows are read (low memory, recent data only)."""
    if requests is None and not GITHUB_ALERTS_CACHE.exists():
        return None
    try:
        should_fetch = force_refresh and max_rows is None  # don't re-fetch when we only want tail
        if not should_fetch and GITHUB_ALERTS_CACHE.exists() and max_cache_age_minutes is not None and max_rows is None:
            age_min = (datetime.now().timestamp() - GITHUB_ALERTS_CACHE.stat().st_mtime) / 60
            if age_min > max_cache_age_minutes:
                should_fetch = True
        if not GITHUB_ALERTS_CACHE.exists() and requests is not None:
            # Stream to file to avoid OOM when we only need tail
            print("     Fetching GitHub alerts (streaming)...", flush=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            r = requests.get(GITHUB_ALERTS_URL, timeout=60, stream=True)
            r.raise_for_status()
            with open(GITHUB_ALERTS_CACHE, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        elif max_rows is not None and GITHUB_ALERTS_CACHE.exists() and requests is not None:
            # When using recent-only, refresh cache if older than 1 min so server gets latest data
            age_min = (datetime.now().timestamp() - GITHUB_ALERTS_CACHE.stat().st_mtime) / 60
            if age_min > 1:
                print("     Fetching latest alerts from GitHub (cache was {:.0f} min old)...".format(age_min), flush=True)
                r = requests.get(GITHUB_ALERTS_URL, timeout=60, stream=True)
                r.raise_for_status()
                with open(GITHUB_ALERTS_CACHE, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            else:
                print("     Using cached alerts ({:.0f} min old).".format(age_min), flush=True)
        if should_fetch and requests is not None and (max_rows is None or not GITHUB_ALERTS_CACHE.exists()):
            print("     Re-fetching GitHub alerts (cache stale or --refresh)...", flush=True)
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                r = requests.get(GITHUB_ALERTS_URL, timeout=120, stream=True)
                r.raise_for_status()
                with open(GITHUB_ALERTS_CACHE, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            except Exception as _e:
                print(f"     Re-fetch failed ({_e}); using cached copy.", flush=True)
        if GITHUB_ALERTS_CACHE.exists():
            if max_rows is not None:
                df = _read_csv_tail(GITHUB_ALERTS_CACHE, max_rows)
            else:
                df = pd.read_csv(GITHUB_ALERTS_CACHE, nrows=500000)
        else:
            return None
        if df is None or df.empty:
            return None
    except Exception:
        return None
    result = _minute_timeline_from_parsed_df(df)
    if result is not None:
        result = _supplement_with_oref_history(result)
    return result


def _minute_timeline_from_parsed_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Build minute-level timeline from a parsed alerts DataFrame with _dt or date+time."""
    if df.empty:
        return None

    # --- Extract prepare-warning minutes BEFORE filtering to strikes ---
    # "בדקות הקרובות" = "In the coming minutes alerts are expected" — issued by Home Front
    # Command when they detect an incoming attack on radar; precedes actual rockets by 2–5 min.
    prepare_minutes = None
    if "matrix_id" in df.columns and "category_desc" in df.columns:
        prep_df = df[(df["matrix_id"] == 10) &
                     (df["category_desc"].str.contains("בדקות הקרובות", na=False))]
        if not prep_df.empty:
            for c in prep_df.columns:
                if str(c).strip().lower() in ("alertdate", "alert_date", "datetime"):
                    prep_df = prep_df.copy()
                    prep_df["_dt"] = pd.to_datetime(prep_df[c], errors="coerce")
                    break
            else:
                if "date" in prep_df.columns and "time" in prep_df.columns:
                    prep_df = prep_df.copy()
                    prep_df["_dt"] = pd.to_datetime(prep_df["date"] + " " + prep_df["time"],
                                                     format="%d.%m.%Y %H:%M:%S", errors="coerce")
            if "_dt" in prep_df.columns:
                prep_df = prep_df.dropna(subset=["_dt"])
                if not prep_df.empty:
                    prep_df["_minute"] = prep_df["_dt"].dt.floor("min")
                    prepare_minutes = set(prep_df["_minute"].unique())

    # Filter to actual attack alerts only:
    #   matrix_id 1 = rocket/missile fire (ירי רקטות וטילים)
    #   matrix_id 6 = hostile aircraft/drones (חדירת כלי טיס עוין)
    if "matrix_id" in df.columns:
        df = df[df["matrix_id"].isin([1])].copy()
        if df.empty:
            return None

    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("alertdate", "alert_date", "datetime"):
            date_col = c
            break
    if date_col is None and "date" in df.columns and "time" in df.columns:
        df["_dt"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
    else:
        if date_col is None:
            date_col = "date" if "date" in df.columns else df.columns[0]
        df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_dt"])
    if df.empty:
        return None
    # Ensure chronological order (tail might be read from end of file; sort so start/end are min/max date)
    df = df.sort_values("_dt").reset_index(drop=True)
    df["_minute"] = df["_dt"].dt.floor("min")
    minute_counts = df.groupby("_minute", as_index=False).size()
    minute_counts.columns = ["datetime", "strike_count"]
    minute_counts["strike"] = 1
    start = minute_counts["datetime"].min()
    end = minute_counts["datetime"].max()
    full_range = pd.date_range(start=start, end=end, freq="min")
    timeline = pd.DataFrame({"datetime": full_range})
    timeline = timeline.merge(minute_counts, on="datetime", how="left")
    timeline["strike"] = timeline["strike"].fillna(0).astype(int)
    timeline["strike_count"] = timeline["strike_count"].fillna(0).astype(int)
    # Add prepare_alert column: 1 if a "coming minutes" warning was issued in this exact minute
    if prepare_minutes:
        timeline["prepare_alert"] = timeline["datetime"].isin(prepare_minutes).astype(int)
    else:
        timeline["prepare_alert"] = 0
    return timeline.sort_values("datetime").reset_index(drop=True)


def load_kaggle_tzeva_adom() -> Optional[pd.DataFrame]:
    """
    Load Kaggle "Rocket alerts in Israel made by Tzeva Adom" from data_cache.
    Expects CSV with at least one date/datetime column; aggregates to daily strike count.
    Returns DataFrame with columns [date, strike, strike_count] or None if not found.
    """
    for name in KAGGLE_ALERT_NAMES:
        path = CACHE_DIR / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, nrows=100000)
        except Exception:
            continue
        # Infer date column (case-insensitive)
        date_col = None
        for c in df.columns:
            c_lower = c.strip().lower()
            if c_lower in ("date", "datetime", "time", "alert_date", "תאריך"):
                date_col = c
                break
        if date_col is None and len(df.columns) >= 1:
            # Try first column as date
            date_col = df.columns[0]
        if date_col is None:
            continue
        df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_dt"])
        if df.empty:
            continue
        df["_date"] = df["_dt"].dt.normalize()
        daily = df.groupby("_date", as_index=False).size()
        daily.columns = ["date", "strike_count"]
        daily["strike"] = (daily["strike_count"] > 0).astype(int)
        return daily[["date", "strike", "strike_count"]]
    return None


def load_kaggle_tzeva_adom_minute() -> Optional[pd.DataFrame]:
    """
    Load Kaggle Tzeva Adom with per-minute resolution.
    Returns DataFrame with columns [datetime, strike, strike_count] (one row per minute),
    or None if no Kaggle CSV or no parseable datetime.
    """
    for name in KAGGLE_ALERT_NAMES:
        path = CACHE_DIR / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, nrows=500000)
        except Exception:
            continue
        date_col = None
        for c in df.columns:
            c_lower = c.strip().lower()
            if c_lower in ("date", "datetime", "time", "alert_date", "תאריך"):
                date_col = c
                break
        if date_col is None and len(df.columns) >= 1:
            date_col = df.columns[0]
        if date_col is None:
            continue
        df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_dt"])
        if df.empty:
            continue
        df["_minute"] = df["_dt"].dt.floor("min")
        minute_counts = df.groupby("_minute", as_index=False).size()
        minute_counts.columns = ["datetime", "strike_count"]
        minute_counts["strike"] = 1
        start = minute_counts["datetime"].min()
        end = minute_counts["datetime"].max()
        full_range = pd.date_range(start=start, end=end, freq="min")
        timeline = pd.DataFrame({"datetime": full_range})
        timeline = timeline.merge(minute_counts, on="datetime", how="left")
        timeline["strike"] = timeline["strike"].fillna(0).astype(int)
        timeline["strike_count"] = timeline["strike_count"].fillna(0).astype(int)
        return timeline.sort_values("datetime").reset_index(drop=True)
    return None


def load_strike_events() -> pd.DataFrame:
    """Load curated CSV of strike events (date, strike 0/1, strike_count, notes)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Strike data not found at {DATA_PATH}. "
            "Ensure data_cache/iran_rocket_strikes_israel.csv exists."
        )
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fetch_oref_alerts_today() -> tuple[list, bool]:
    """
    Fetch current/recent alerts from Pikud Haoref (Oref) API.
    Returns (list of alert dicts, success).
    Data source: https://www.oref.org.il/WarningMessages/alert/alerts.json
    """
    if requests is None:
        return [], False
    try:
        r = requests.get(OREF_ALERTS_URL, headers=OREF_HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return [], False
    # API returns {"data": ["Area1","Area2"], "id": "...", "title": "..."} or similar
    if isinstance(data, list):
        return data, True
    if isinstance(data, dict):
        raw = data.get("data")
        if isinstance(raw, list):
            # Each item can be string (area name) or dict; one entry per alert event
            return raw if raw else [], True
        return [], True
    return [], True


def load_oref_alerts_from_file(path: Optional[Path] = None) -> list:
    """Load today's alerts from a JSON file (e.g. from Pikud Haoref MCP get_alert_history)."""
    if path is None:
        path = CACHE_DIR / "oref_alerts_today.json"
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data.get("data") or []
    return []


def count_today_alerts(alerts: list) -> int:
    """
    Count alerts that occurred today. If alert items are area-name strings (Oref format),
    each non-empty response is one alert event; if dicts, use date field.
    """
    today = pd.Timestamp(datetime.now()).normalize()
    if not alerts:
        return 0
    # Oref API often returns {"data": ["Area1", "Area2"]} = one event, multiple areas
    first = alerts[0]
    if isinstance(first, str):
        return 1 if any(a for a in alerts) else 0
    count = 0
    for a in alerts:
        if not isinstance(a, dict):
            continue
        dt = None
        for key in ("date", "datetime", "time", "datestamp", "תאריך"):
            if key in a and a[key]:
                try:
                    dt = pd.to_datetime(a[key])
                    break
                except Exception:
                    pass
        if dt is not None and pd.Timestamp(dt).normalize() == today:
            count += 1
        else:
            count += 1  # assume one event if we have dicts without date
    return count


def _get_today_strike_minutes() -> set:
    """
    Return set of minute timestamps (pd.Timestamp floored to minute) when a strike/alert occurred today.
    Uses data_cache/oref_alerts_today.json (from MCP get_alert_history) and optionally Oref API (current snapshot).
    So the model can see today's strikes up to this time.
    """
    out: set = set()
    today = pd.Timestamp(datetime.now()).normalize()
    now_floor = pd.Timestamp(datetime.now()).floor("min")

    # 1) File: per-event timestamps (best: full history of today)
    alerts_file = load_oref_alerts_from_file()
    if alerts_file:
        for a in alerts_file:
            if isinstance(a, dict):
                dt = None
                for key in ("date", "datetime", "time", "datestamp", "alertDate", "תאריך"):
                    if key in a and a[key]:
                        try:
                            dt = pd.to_datetime(a[key])
                            break
                        except Exception:
                            pass
                if dt is not None and pd.Timestamp(dt).normalize() == today:
                    out.add(pd.Timestamp(dt).floor("min"))
            # If list of strings (area names), one event at "now" – add below via API or once here
        if not out and alerts_file and isinstance(alerts_file[0], str):
            out.add(now_floor)

    # 2) API: current snapshot – if there's an alert right now, mark this minute
    alerts_api, ok = fetch_oref_alerts_today()
    if ok and alerts_api and any(a for a in alerts_api):
        out.add(now_floor)

    return out


def build_daily_timeline(
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    use_kaggle: bool = True,
) -> pd.DataFrame:
    """
    Build a continuous daily timeline. Prefers Kaggle Tzeva Adom if present;
    otherwise uses curated iran_rocket_strikes_israel.csv.
    """
    end = end or pd.Timestamp(datetime.now()).normalize()
    if isinstance(end, str):
        end = pd.to_datetime(end).normalize()
    end = pd.Timestamp(end).normalize()

    # 1) Try GitHub (dleshem/israel-alerts-data) – no geo-restriction
    github = fetch_israel_alerts_github()
    if github is not None and len(github) > 0:
        github["date"] = pd.to_datetime(github["date"]).dt.normalize()
        start = start or github["date"].min()
        if isinstance(start, str):
            start = pd.to_datetime(start).normalize()
        start = pd.Timestamp(start).normalize()
        days = pd.date_range(start=start, end=end, freq="D")
        timeline = pd.DataFrame({"date": days})
        timeline = timeline.merge(
            github.drop_duplicates("date"),
            on="date",
            how="left",
        )
        timeline["strike"] = timeline["strike"].fillna(0).astype(int)
        timeline["strike_count"] = timeline["strike_count"].fillna(0).astype(int)
        return timeline.sort_values("date").reset_index(drop=True)

    # 2) Try Kaggle Tzeva Adom (local)
    if use_kaggle:
        kaggle = load_kaggle_tzeva_adom()
        if kaggle is not None and len(kaggle) > 0:
            kaggle["date"] = pd.to_datetime(kaggle["date"]).dt.normalize()
            start = start or kaggle["date"].min()
            if isinstance(start, str):
                start = pd.to_datetime(start).normalize()
            start = pd.Timestamp(start).normalize()
            days = pd.date_range(start=start, end=end, freq="D")
            timeline = pd.DataFrame({"date": days})
            timeline = timeline.merge(
                kaggle.drop_duplicates("date"),
                on="date",
                how="left",
            )
            timeline["strike"] = timeline["strike"].fillna(0).astype(int)
            timeline["strike_count"] = timeline["strike_count"].fillna(0).astype(int)
            return timeline.sort_values("date").reset_index(drop=True)

    # 3) Fallback: curated Iran strikes CSV
    events = load_strike_events()
    start = start or events["date"].min().normalize()
    if isinstance(start, str):
        start = pd.to_datetime(start).normalize()
    start = pd.Timestamp(start).normalize()

    days = pd.date_range(start=start, end=end, freq="D")
    timeline = pd.DataFrame({"date": days})
    strike_info = events[["date", "strike", "strike_count"]].copy()
    strike_info["date"] = strike_info["date"].dt.normalize()
    timeline = timeline.merge(
        strike_info.drop_duplicates("date"),
        on="date",
        how="left",
    )
    timeline["strike"] = timeline["strike"].fillna(0).astype(int)
    timeline["strike_count"] = timeline["strike_count"].fillna(0).astype(int)
    return timeline.sort_values("date").reset_index(drop=True)


def build_minute_timeline(
    end: Optional[pd.Timestamp] = None,
    use_kaggle: bool = True,
    force_refresh: bool = False,
    max_rows: Optional[int] = None,
    keep_last_minutes: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Build a continuous per-minute timeline. Tries GitHub (dleshem) first, then Kaggle.
    If max_rows is set, only the last max_rows CSV rows are loaded (low memory, for live-update servers).
    keep_last_minutes: when set with max_rows, keep this many minutes (default 7d+5h). Use larger (e.g. 30*24*60) for pretrained model so features match training distribution.
    Returns None if no minute-level data available.
    """
    import sys
    print("     Loading/parsing minute data...", flush=True)
    minute_df = _fetch_github_alerts_minute(force_refresh=force_refresh, max_rows=max_rows)
    if minute_df is None or len(minute_df) < 1000:
        if max_rows is None:
            minute_df = load_kaggle_tzeva_adom_minute()
        else:
            minute_df = None
    if minute_df is None or len(minute_df) == 0:
        return None
    end = end or _now_israeli_time().floor("min")
    if isinstance(end, str):
        end = pd.to_datetime(end).floor("min")
    last = minute_df["datetime"].max()
    if last < end:
        extra = pd.date_range(start=last + pd.Timedelta(minutes=1), end=end, freq="min")
        extra_df = pd.DataFrame({"datetime": extra, "strike": 0, "strike_count": 0})
        minute_df = pd.concat([minute_df, extra_df], ignore_index=True).sort_values("datetime").reset_index(drop=True)
    # Merge today's strikes (Oref file or API) so the model sees today up to this time
    today_minutes = _get_today_strike_minutes()
    if today_minutes:
        minute_df = minute_df.copy()
        dt_floor = minute_df["datetime"].dt.floor("min")
        mask = dt_floor.isin(today_minutes)
        if mask.any():
            minute_df.loc[mask, "strike"] = 1
            minute_df.loc[mask, "strike_count"] = np.maximum(minute_df.loc[mask, "strike_count"].values, 1)
            print(f"     Merged {mask.sum()} strike minute(s) from today (Oref) into timeline.", flush=True)
    # When using recent-only (max_rows), trim to last N minutes. Larger N = closer to full-data feature distribution (for pretrained model).
    if max_rows is not None:
        max_minutes = keep_last_minutes if keep_last_minutes is not None else (7 * 24 * 60 + 300)
        if len(minute_df) > max_minutes:
            minute_df = minute_df.iloc[-max_minutes:].reset_index(drop=True)
            print(f"     Kept last {max_minutes} minutes for recent-only mode.", flush=True)
    t0, t1 = minute_df["datetime"].min(), minute_df["datetime"].max()
    print("     Timeline: {} to {} (through now).".format(t0.strftime("%Y-%m-%d %H:%M"), t1.strftime("%Y-%m-%d %H:%M")), flush=True)
    return minute_df


def hazard_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hazard-style features for predicting next strike.
    - days_since_last_strike (fill with large value when none yet)
    - strikes_in_last_7d, strikes_in_last_30d
    - day_of_week (0–6), month, day_of_year
    - trend: days since timeline start (normalized)
    """
    out = df.copy()
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear

    # Days since last strike
    last_strike_idx = None
    days_since = []
    for i, row in tqdm(out.iterrows(), total=len(out), desc="Hazard (day)", leave=False):
        if row["strike"] == 1:
            last_strike_idx = i
        if last_strike_idx is None:
            days_since.append(999.0)  # no prior strike in window
        else:
            days_since.append(float(i - last_strike_idx))
    out["days_since_last_strike"] = days_since

    # Rolling counts (past 7 and 30 days, excluding current day)
    out["strikes_in_last_7d"] = (
        out["strike"]
        .rolling(8, min_periods=1)
        .apply(lambda x: max(0, x.iloc[:-1].sum()), raw=False)
        .fillna(0)
        .astype(int)
    )
    out["strikes_in_last_30d"] = (
        out["strike"]
        .rolling(31, min_periods=1)
        .apply(lambda x: max(0, x.iloc[:-1].sum()), raw=False)
        .fillna(0)
        .astype(int)
    )

    # Normalized trend
    start = out["date"].min()
    out["days_since_start"] = (out["date"] - start).dt.days
    out["trend"] = out["days_since_start"] / max(out["days_since_start"].max(), 1)
    return out


def hazard_features_minute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-minute hazard features.
    Expects column 'datetime'. Adds: minutes_since_last_strike, strikes_in_last_60min,
    strikes_in_last_1440min (24h), minute_of_day, hour, day_of_week, trend.
    """
    out = df.copy()
    if "date" in out.columns and "datetime" not in out.columns:
        out["datetime"] = out["date"]
    out["minute_of_day"] = out["datetime"].dt.hour * 60 + out["datetime"].dt.minute
    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek
    # Vectorized: minutes since last strike (index-based)
    strike_idx = np.where(out["strike"].values == 1)[0]
    if len(strike_idx) == 0:
        out["minutes_since_last_strike"] = 999999.0
    else:
        n = len(out)
        i_arr = np.arange(n)
        pos = np.searchsorted(strike_idx, i_arr, side="right") - 1
        out["minutes_since_last_strike"] = np.where(
            pos >= 0, (i_arr - strike_idx[pos]).astype(float), 999999.0
        )
    def _rolling_past_sum(s: pd.Series, w: int) -> np.ndarray:
        # rolling(w).sum() includes current row; subtract it to get past-only sum (causal)
        r = s.rolling(w, min_periods=1).sum() - s
        return r.clip(lower=0).fillna(0).values.astype(int)

    n = len(out)
    W7D = 10080
    print("     Computing rolling features (vectorized)...", flush=True)
    out["strikes_in_last_60min"]   = _rolling_past_sum(out["strike"], 61)
    out["strikes_in_last_1440min"] = _rolling_past_sum(out["strike"], 1441)
    out["strikes_in_last_10080min"] = _rolling_past_sum(out["strike"], W7D + 1)
    # Prepare-alert features: "בדקות הקרובות" (coming minutes) warning from Home Front Command
    # precedes actual rocket barrages by 2–5 min — very strong leading indicator
    if "prepare_alert" in out.columns:
        pa = out["prepare_alert"].fillna(0).values.astype(int)
        pa_idx = np.where(pa == 1)[0]
        if len(pa_idx) > 0:
            pos = np.searchsorted(pa_idx, np.arange(n), side="right") - 1
            out["minutes_since_last_prepare"] = np.where(
                pos >= 0, (np.arange(n) - pa_idx[pos]).astype(float), 999999.0
            )
        else:
            out["minutes_since_last_prepare"] = 999999.0
        out["prepare_alert_in_last_15min"] = _rolling_past_sum(out["prepare_alert"].fillna(0), 16)
    else:
        out["minutes_since_last_prepare"] = 999999.0
        out["prepare_alert_in_last_15min"] = 0
    # Log scale for time since strike (helps model use both short and long gaps)
    out["log_minutes_since_strike"] = np.log1p(np.minimum(out["minutes_since_last_strike"], 1e6))
    start = out["datetime"].min()
    out["minutes_since_start"] = (out["datetime"] - start).dt.total_seconds() / 60
    out["trend"] = out["minutes_since_start"] / max(out["minutes_since_start"].max(), 1)
    return out


def build_sequences_hazard(
    df: pd.DataFrame,
    horizon_days: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each day t, features = hazard features at t, target = any strike in [t+1, t+horizon_days].
    Returns (X, y) where y is binary.
    """
    X_list, y_list = [], []
    n_seq = len(df) - horizon_days
    for i in tqdm(range(n_seq), desc="Sequences (day)", leave=False):
        row = df.iloc[i]
        X_list.append(row[FEATURE_COLS_DAY].values.astype(float))
        future = df.iloc[i + 1 : i + 1 + horizon_days]["strike"]
        y_list.append(1 if future.any() else 0)
    if not X_list:
        return np.empty((0, len(FEATURE_COLS_DAY))), np.empty(0)
    return np.array(X_list), np.array(y_list, dtype=int)


FEATURE_COLS_DAY = [
    "days_since_last_strike",
    "strikes_in_last_7d",
    "strikes_in_last_30d",
    "day_of_week",
    "month",
    "day_of_year",
    "trend",
]

FEATURE_COLS_MINUTE = [
    "minutes_since_last_strike",
    "log_minutes_since_strike",
    "strikes_in_last_60min",
    "strikes_in_last_1440min",
    "strikes_in_last_10080min",
    "minute_of_day",
    "hour",
    "day_of_week",
    "trend",
    # Prepare-alert features (Home Front Command "coming minutes" warning)
    "minutes_since_last_prepare",
    "prepare_alert_in_last_15min",
]


def _minute_features_at_index(df: pd.DataFrame, strike_array: np.ndarray, idx: int) -> np.ndarray:
    """Compute one row of minute hazard features at index idx using strike_array (for predicted-path backtest)."""
    strike = np.asarray(strike_array).ravel()
    n = len(strike)
    if idx < 0 or idx >= n:
        raise IndexError(idx)
    # Minutes since last strike (scan backward)
    last_strike_idx = None
    for i in range(idx - 1, -1, -1):
        if strike[i] == 1:
            last_strike_idx = i
            break
    minutes_since = (idx - last_strike_idx) if last_strike_idx is not None else 999999.0
    log_since = np.log1p(min(minutes_since, 1e6))
    # Past-only rolling counts (excluding current minute)
    s60 = int(strike[max(0, idx - 60) : idx].sum())
    s1440 = int(strike[max(0, idx - 1440) : idx].sum())
    W7D = 10080
    s7d = int(strike[max(0, idx - W7D) : idx].sum())
    # Time from df (trend is time-based, not strike-based)
    row = df.iloc[idx]
    minute_of_day = row["datetime"].hour * 60 + row["datetime"].minute
    hour = row["datetime"].hour
    day_of_week = row["datetime"].dayofweek
    trend = float(row["trend"]) if "trend" in df.columns else 0.0
    # Prepare-alert features
    mins_since_prepare = float(row["minutes_since_last_prepare"]) if "minutes_since_last_prepare" in df.columns else 999999.0
    prep_15 = int(row["prepare_alert_in_last_15min"]) if "prepare_alert_in_last_15min" in df.columns else 0
    return np.array(
        [minutes_since, log_since, s60, s1440, s7d, minute_of_day, hour, day_of_week, trend,
         mins_since_prepare, prep_15],
        dtype=float,
    )


def _next_hour_features(df: pd.DataFrame, last_idx: int, n_minutes: int = 60) -> np.ndarray:
    """Build feature rows for the next n_minutes (1..n_minutes), assuming no strike in between."""
    row = df.iloc[last_idx]
    base = {c: row[c] for c in FEATURE_COLS_MINUTE}
    strike = df["strike"].values
    n = len(df)
    W7D = 10080
    last_60 = strike[max(0, n - 60) : n]
    last_1440 = strike[max(0, n - 1440) : n]
    last_7d = strike[max(0, n - W7D) : n]
    if len(last_60) < 60:
        last_60 = np.concatenate([np.zeros(60 - len(last_60), dtype=int), last_60])
    if len(last_1440) < 1440:
        last_1440 = np.concatenate([np.zeros(1440 - len(last_1440), dtype=int), last_1440])
    if len(last_7d) < W7D:
        last_7d = np.concatenate([np.zeros(W7D - len(last_7d), dtype=int), last_7d])
    total_minutes = max(df["minutes_since_start"].max(), 1)
    dt = df["datetime"].iloc[last_idx]
    # Prepare-alert features: carry forward from last known state (alert ages into the future)
    base_mins_since_prepare = float(row["minutes_since_last_prepare"]) if "minutes_since_last_prepare" in row.index else 999999.0
    base_prep_15 = int(row["prepare_alert_in_last_15min"]) if "prepare_alert_in_last_15min" in row.index else 0
    rows = []
    for k in range(1, n_minutes + 1):
        mins_since = base["minutes_since_last_strike"] + k
        log_since = np.log1p(min(mins_since, 1e6))
        new_60 = np.concatenate([last_60[k:], np.zeros(k, dtype=int)])[-60:]
        new_1440 = np.concatenate([last_1440[k:], np.zeros(k, dtype=int)])[-1440:]
        new_7d = np.concatenate([last_7d[k:], np.zeros(k, dtype=int)])[-W7D:]
        dt_k = dt + pd.Timedelta(minutes=k)
        minute_of_day = dt_k.hour * 60 + dt_k.minute
        hour = dt_k.hour
        day_of_week = dt_k.dayofweek
        trend = (row["minutes_since_start"] + k) / total_minutes
        # Prepare alert ages by k more minutes; count stays same (we don't know future alerts)
        mins_since_prepare = min(base_mins_since_prepare + k, 999999.0)
        rows.append([mins_since, log_since, int(new_60.sum()), int(new_1440.sum()), int(new_7d.sum()),
                     minute_of_day, hour, day_of_week, trend, mins_since_prepare, base_prep_15])
    return np.array(rows, dtype=float)


def build_sequences_hazard_minute(
    df: pd.DataFrame,
    horizon_minutes: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-minute: features at t, target = any strike in next horizon_minutes. Vectorized."""
    n = len(df)
    if n <= horizon_minutes:
        return np.empty((0, len(FEATURE_COLS_MINUTE))), np.empty(0)
    X = df[FEATURE_COLS_MINUTE].values.astype(float)[:n - horizon_minutes]
    strike = df["strike"].values
    if horizon_minutes == 1:
        y = strike[1:].astype(int)
    else:
        # any strike in [t+1, t+horizon]
        y = np.array([
            1 if strike[i + 1 : i + 1 + horizon_minutes].any() else 0
            for i in range(n - horizon_minutes)
        ], dtype=int)
    return X, y


def train_test_split_by_minutes(
    df: pd.DataFrame,
    test_minutes: int = 7 * 24 * 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: last test_minutes for test."""
    n = len(df)
    if n <= test_minutes:
        return df, pd.DataFrame()
    train = df.iloc[: n - test_minutes].copy()
    test = df.iloc[n - test_minutes :].copy()
    return train, test


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def _balanced_oversample(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Downsample negatives to 1:1 ratio. Keeps all positives (rare real events), samples equal negatives.
    Far faster and less memory than oversampling, and forces 50% baseline so model must learn."""
    y = np.asarray(y).ravel()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return X, y
    rng = np.random.default_rng(random_state)
    # Downsample negatives to match positives → 50% positive rate, model can't cheat with all-zeros
    sampled_neg = rng.choice(neg_idx, size=min(n_neg, n_pos), replace=False)
    all_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(all_idx)
    print(f"  Balanced: {n_pos} pos + {len(sampled_neg)} neg (from {n_neg} neg) → {len(all_idx)} total", flush=True)
    return X[all_idx], y[all_idx]


def train_hazard_nn(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: tuple = (256, 128, 64, 32),
    max_iter: int = 1000,
    random_state: int = 42,
    class_weight_balanced: bool = True,
) -> tuple:
    """Train MLP classifier for P(strike in next period). Returns (model, scaler)."""
    if MLPClassifier is None or StandardScaler is None:
        raise ImportError("scikit-learn required: pip install scikit-learn")
    if class_weight_balanced:
        X, y = _balanced_oversample(X, y, random_state=random_state)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.15,
        verbose=True,
    )
    model.fit(X_sc, y)
    return model, scaler


def predict_proba_strike(model, scaler, X: np.ndarray) -> np.ndarray:
    """Predict P(strike in next period) for each row. Returns (n_samples,) proba of class 1."""
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_sc = scaler.transform(X)
    X_sc = np.nan_to_num(X_sc, nan=0.0, posinf=0.0, neginf=0.0)
    return model.predict_proba(X_sc)[:, 1]


# -----------------------------------------------------------------------------
# Train / test split and evaluation
# -----------------------------------------------------------------------------

def train_test_split_by_date(
    df: pd.DataFrame,
    test_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time: last test_days for test, rest for train."""
    n = len(df)
    if n <= test_days:
        train = df
        test = pd.DataFrame()
        return train, test
    train = df.iloc[: n - test_days].copy()
    test = df.iloc[n - test_days :].copy()
    return train, test


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute metrics for binary prediction."""
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    metrics = {"accuracy": acc, "n": len(y_true)}
    if len(np.unique(y_true)) >= 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return metrics


# -----------------------------------------------------------------------------
# Main: run with today's data as test
# -----------------------------------------------------------------------------

def _run_daily(
    timeline: pd.DataFrame,
    test_days: int = 60,
) -> tuple:
    """Daily resolution pipeline. timeline has 'date' column."""
    df = hazard_features(timeline)
    X, y = build_sequences_hazard(df, horizon_days=1)
    print(f"  Samples (X, y): {X.shape[0]}, positive rate {y.mean():.3f}")
    train_df, test_df = train_test_split_by_date(df, test_days=test_days)
    if len(test_df) < 2:
        n = len(df) - 1
        split = int(0.8 * n)
        train_df = df.iloc[: split + 1]
        test_df = df.iloc[split + 1 :]
    X_train, y_train = build_sequences_hazard(train_df, horizon_days=1)
    X_test, y_test = build_sequences_hazard(test_df, horizon_days=1)
    if X_train.size == 0:
        return None
    if X_test.size == 0:
        X_test, y_test = X_train, y_train
    print("\nTraining hazard NN (P(strike in next day))...")
    model, scaler = train_hazard_nn(X_train, y_train)
    p_train = predict_proba_strike(model, scaler, X_train)
    m_train = evaluate(y_train, p_train)
    print("\n--- Train set ---")
    print(f"  Accuracy: {m_train['accuracy']:.4f}")
    if not np.isnan(m_train["roc_auc"]):
        print(f"  ROC-AUC:  {m_train['roc_auc']:.4f}")
    print(f"  Confusion:\n{m_train['confusion_matrix']}")
    p_test = predict_proba_strike(model, scaler, X_test)
    m_test = evaluate(y_test, p_test)
    print("\n--- Test set ---")
    print(f"  Test window: {test_df['date'].min().date()} to {test_df['date'].max().date()}")
    print(f"  Accuracy: {m_test['accuracy']:.4f}")
    print(f"  Confusion:\n{m_test['confusion_matrix']}")
    last_idx = len(df) - 1
    if last_idx >= 0:
        last_X = df.iloc[last_idx][FEATURE_COLS_DAY].values.astype(float).reshape(1, -1)
        prob_today = predict_proba_strike(model, scaler, last_X)[0]
        today_str = df.iloc[last_idx]["date"].strftime("%Y-%m-%d")
        print(f"\n--- Prediction for {today_str} (today) ---")
        print(f"  P(strike in next day) = {prob_today:.4f}")
    return model, scaler, df


# Coarser horizons derived from 1-min model (no extra training)
HAZARD_HORIZONS_DERIVED = [1, 5, 15, 60]


def _plot_last_5h_backtest(
    df: pd.DataFrame,
    model,
    scaler,
    now_str: str,
) -> None:
    """Plot last 5 hours: actual (true) vs prediction; and predicted-path (P>50% → treat as strike, recompute next prob). Saves to data_cache/rocket_strike_last_5h.png."""
    last_idx = len(df) - 1
    n_5h = 5 * 60
    start = max(1, last_idx - n_5h + 1)
    # True-data prediction: features from actual timeline
    X_5h = df.iloc[start - 1 : last_idx][FEATURE_COLS_MINUTE].values.astype(float)
    pred_5h = predict_proba_strike(model, scaler, X_5h)
    actual_5h = df.iloc[start : last_idx + 1]["strike"].values
    times_5h = df.iloc[start : last_idx + 1]["datetime"]
    n_strikes = int(actual_5h.sum())
    mean_pred = float(np.mean(pred_5h))
    t_start = times_5h.iloc[0]
    t_end = times_5h.iloc[-1]
    # Predicted path: when P(strike in next min) > 50%, treat as strike and use that to compute next probability (no true data for that minute)
    strike_hybrid = df["strike"].values.copy()
    pred_path_probs = np.zeros(len(pred_5h))
    pred_path_strikes = np.zeros(len(pred_5h), dtype=int)
    for j in range(len(pred_5h)):
        if j == 0:
            feat_idx = start - 1
            feat_row = df.iloc[feat_idx][FEATURE_COLS_MINUTE].values.astype(float)
        else:
            strike_hybrid[start : start + j] = pred_path_strikes[:j]
            feat_idx = start + j - 1
            feat_row = _minute_features_at_index(df, strike_hybrid, feat_idx)
        p_j = predict_proba_strike(model, scaler, feat_row.reshape(1, -1))[0]
        pred_path_probs[j] = p_j
        pred_path_strikes[j] = 1 if p_j > 0.5 else 0
    n_pred_strikes = int(pred_path_strikes.sum())
    mean_pred_path = float(np.mean(pred_path_probs))
    print(f"\n--- Last 5 hours (backtest: {t_start} → {t_end}) ---")
    print(f"  Actual strikes: {n_strikes} minute(s)")
    print(f"  Mean predicted P (true data): {mean_pred:.4f}")
    print(f"  Predicted-path (P>50% → strike): {n_pred_strikes} predicted strike(s), mean P: {mean_pred_path:.4f}")
    if plt is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), height_ratios=[0.9, 1], sharex=True)
    ax1, ax2 = axes
    x = np.arange(len(times_5h))
    ax1.fill_between(x, 0, actual_5h, step="post", alpha=0.6, color="C2", label="Actual strike (true)")
    ax1.fill_between(x, 0, pred_path_strikes, step="post", alpha=0.5, color="C1", label="Model strike (P>50%, predicted path)")
    ax1.set_ylabel("Strike (0/1)")
    ax1.set_ylim(-0.05, 1.3)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("Last 5 hours: true data vs prediction vs predicted-path (P>50% → strike)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(x, pred_5h, color="C0", linewidth=0.9, label="P(strike) using true data")
    ax2.plot(x, pred_path_probs, color="C1", linewidth=0.8, linestyle="--", label="P(strike) using predicted path (P>50%→strike)")
    ax2.axhline(0.5, color="gray", linewidth=0.6, alpha=0.7)
    ax2.set_ylabel("Predicted P(strike)")
    ax2.set_xlabel("Time (minutes ago → now)")
    ax2.legend(loc="upper right", fontsize=8)
    y_max = max(pred_5h.max(), pred_path_probs.max()) * 1.1
    ax2.set_ylim(0, max(y_max, 0.06))
    ax2.grid(True, alpha=0.3)
    step = max(1, len(x) // 6)
    tick_idx = list(range(0, len(x), step))
    if len(x) - 1 not in tick_idx:
        tick_idx.append(len(x) - 1)
    ax2.set_xticks(tick_idx)
    ax2.set_xticklabels([pd.Timestamp(times_5h.iloc[i]).strftime("%H:%M") for i in tick_idx], rotation=15)
    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "data_cache" / "rocket_strike_last_5h.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Graph saved: {out_path}")


def _derive_horizon_probs_from_current(p_1min: float, horizons: list) -> dict:
    """From current 1-min probability p, derive P(strike in next H min) = 1 - (1-p)^H (constant hazard).
    Longer horizons then correctly show higher probability (more time = more chance of at least one strike)."""
    out = {}
    for h in horizons:
        if h <= 0:
            continue
        p = max(0.0, min(1.0, float(p_1min)))
        out[h] = float(1.0 - (1.0 - p) ** h)
    return out


def _run_minute(
    timeline: pd.DataFrame,
    test_minutes: int = 7 * 24 * 60,
    horizon_minutes: int = 1,
) -> tuple:
    """Single 1-min model at finest resolution; 5/15/60 min hazards derived from it (no extra training)."""
    print("  Building minute-level hazard features (deep context: 60m, 24h, 7d windows + log time)...", flush=True)
    df = hazard_features_minute(timeline)
    train_df, test_df = train_test_split_by_minutes(df, test_minutes=test_minutes)
    if len(test_df) < 2:
        n = len(df) - 1
        split = int(0.8 * n)
        train_df = df.iloc[: split + 1]
        test_df = df.iloc[split + 1 :]

    # Single model at finest resolution (1 min only)
    print("  Building sequences (1-min horizon only)...", flush=True)
    X_train, y_train = build_sequences_hazard_minute(train_df, horizon_minutes=1)
    X_test, y_test = build_sequences_hazard_minute(test_df, horizon_minutes=1)
    if X_train.size == 0 or y_train.sum() == 0:
        return None
    print(f"  Samples: {len(y_train):,} train, positive rate {y_train.mean():.3f}", flush=True)
    print("  Training single 1-min hazard model (sklearn will print iteration progress below)...", flush=True)
    model, scaler = train_hazard_nn(X_train, y_train, class_weight_balanced=True)
    p_train = predict_proba_strike(model, scaler, X_train)
    m_train = evaluate(y_train, p_train)
    p_test = predict_proba_strike(model, scaler, X_test)
    m_test = evaluate(y_test, p_test)
    print("\n--- Train set (1-min hazard) ---")
    print(f"  Accuracy: {m_train['accuracy']:.4f}, ROC-AUC: {m_train.get('roc_auc', np.nan):.4f}")
    print(f"  Confusion:\n{m_train['confusion_matrix']}")
    print("\n--- Test set ---")
    print(f"  Accuracy: {m_test['accuracy']:.4f}, Confusion:\n{m_test['confusion_matrix']}")

    last_idx = len(df) - 1
    if last_idx < 0:
        return model, scaler, df
    now_str = df.iloc[last_idx]["datetime"].strftime("%Y-%m-%d %H:%M")
    # Current-state 1-min prob (uses last row: reflects "strikes today" via 60m/24h/7d features)
    last_X = df.iloc[last_idx][FEATURE_COLS_MINUTE].values.astype(float).reshape(1, -1)
    p_1min_current = predict_proba_strike(model, scaler, last_X)[0]
    # Derive 5/15/60 from constant hazard: P(strike in next H min) = 1 - (1-p)^H so probability increases with horizon
    probs_per_horizon = _derive_horizon_probs_from_current(p_1min_current, HAZARD_HORIZONS_DERIVED)
    # Next 60 minutes curve (simulated forward assuming no strike; hazard decays over the hour)
    next_60_X = _next_hour_features(df, last_idx, n_minutes=60)
    next_60_probs = predict_proba_strike(model, scaler, next_60_X)
    print(f"\n--- Hazards from 1-min model (from {now_str}) ---")
    for h in sorted(probs_per_horizon.keys()):
        print(f"  P(strike in next {h:2d} min) = {probs_per_horizon[h]:.4f}")
    print(f"  Next hour (1-min curve): min = {next_60_probs.min():.4f}, max = {next_60_probs.max():.4f}")
    # Last 5 hours: true data vs prediction (backtest)
    _plot_last_5h_backtest(df, model, scaler, now_str)
    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[1.2, 0.8])
        ax1, ax2 = axes
        minutes_ahead = np.arange(1, 61)
        ax1.fill_between(minutes_ahead, 0, next_60_probs, alpha=0.5, color="C0")
        ax1.plot(minutes_ahead, next_60_probs, "o-", color="C0", markersize=2, label="P(strike in this minute)")
        ax1.set_ylabel("P(strike in next minute)")
        ax1.set_title(f"Strike probability – next 60 minutes (from {now_str})")
        ax1.legend(loc="upper right")
        ax1.set_ylim(0, max(next_60_probs.max() * 1.1, 0.01))
        ax1.grid(True, alpha=0.3)
        ax2.bar([str(h) for h in sorted(probs_per_horizon)], [probs_per_horizon[h] for h in sorted(probs_per_horizon)], color="C1", alpha=0.8)
        ax2.set_xlabel("Horizon (minutes)")
        ax2.set_ylabel("P(strike in next N min)")
        ax2.set_title("Hazards derived from 1-min model (no extra training)")
        ax2.set_ylim(0, max(max(probs_per_horizon.values()) * 1.2, 0.01))
        fig.tight_layout()
        out_path = Path(__file__).resolve().parent / "data_cache" / "rocket_strike_next_hour.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Graph saved: {out_path}")
    return model, scaler, df


def main():
    import sys
    print("Rocket strike hazard NN – starting.", flush=True)
    p = argparse.ArgumentParser(description="Rocket strike hazard NN (day or minute resolution)")
    p.add_argument("--resolution", choices=("minute", "day"), default="minute", help="Time resolution (default: minute)")
    p.add_argument("--test-days", type=int, default=60, help="Test set size in days (day resolution)")
    p.add_argument("--test-minutes", type=int, default=7 * 24 * 60, help="Test set size in minutes (minute resolution)")
    p.add_argument("--horizon-minutes", type=int, default=1, help="Predict strike in next N minutes (minute resolution)")
    p.add_argument("--refresh", action="store_true", help="Re-fetch GitHub alert data so the last hour has fresh data (avoids stale cache)")
    args = p.parse_args()

    resolution = args.resolution
    print("Loading rocket strike / alert data...", flush=True)

    if resolution == "minute":
        print("  Building minute-level timeline (may take a minute)...", flush=True)
        minute_timeline = build_minute_timeline(force_refresh=args.refresh)
        if minute_timeline is not None and len(minute_timeline) > 1000:
            n_orig = len(minute_timeline)
            if n_orig > MAX_MINUTE_ROWS:
                minute_timeline = minute_timeline.iloc[-MAX_MINUTE_ROWS:].reset_index(drop=True)
                print(f"  Using last {MAX_MINUTE_ROWS:,} minutes (full timeline had {n_orig:,}) so training can finish.")
            print("  Training on: historical Israeli rocket alerts (Pikud Haoref / Tzeva Adom) from GitHub – 2014 onward (all alerts, not only the 12-day war).")
            print("  Using per-minute resolution (GitHub/Kaggle).")
            print(f"  Timeline: {minute_timeline['datetime'].min()} to {minute_timeline['datetime'].max()} ({len(minute_timeline)} minutes)")
            print(f"  Total strike minutes: {minute_timeline['strike'].sum()}")
            result = _run_minute(minute_timeline, test_minutes=args.test_minutes, horizon_minutes=args.horizon_minutes)
            if result is not None:
                _print_oref_today()
                return result
        print("  Minute resolution requires Kaggle CSV with datetime. Falling back to daily.")

    # Daily resolution
    if fetch_israel_alerts_github() is not None:
        print("  Using GitHub (dleshem/israel-alerts-data) – no geo-restriction.")
    elif load_kaggle_tzeva_adom() is not None:
        print("  Using Kaggle Tzeva Adom dataset (daily).")
    else:
        print("  Using curated iran_rocket_strikes_israel.csv.")
    timeline = build_daily_timeline()
    print(f"  Timeline: {timeline['date'].min().date()} to {timeline['date'].max().date()} ({len(timeline)} days)")
    print(f"  Total strike days: {timeline['strike'].sum()}")
    result = _run_daily(timeline, test_days=args.test_days)
    if result is not None:
        _print_oref_today()
    return result


def _print_oref_today():
    """Print today's actual alerts from Oref API or file."""
    print("\n--- Today's actual alerts (Pikud Haoref) ---")
    alerts_api, ok = fetch_oref_alerts_today()
    if ok and alerts_api:
        n = count_today_alerts(alerts_api)
        areas = alerts_api[:5] if isinstance(alerts_api[0], str) else [str(a) for a in alerts_api[:5]]
        print(f"  Oref API: {n} alert event(s) today. Areas: {areas}{'...' if len(alerts_api) > 5 else ''}")
    else:
        alerts_file = load_oref_alerts_from_file()
        if alerts_file:
            n = count_today_alerts(alerts_file)
            print(f"  From data_cache/oref_alerts_today.json: {n} alert(s) today.")
        elif not ok and requests is not None:
            print("  Oref API unreachable (often geo-restriction: API may only allow Israeli IPs).")
            print("  To compare with today's alerts: use Pikud Haoref MCP get_alert_history and save")
            print("  the JSON to: data_cache/oref_alerts_today.json (refresh that file to see today's alerts by minute).")
        else:
            print("  No alerts reported today (Oref API returned empty).")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)  # show output immediately
    main()
