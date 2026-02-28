# Rocket strike prediction – data sources

## Preferred source (no geo-restriction)

**[dleshem/israel-alerts-data](https://github.com/dleshem/israel-alerts-data)** – Historical Pikud Haoref (Israeli Home Front Command) alerts on GitHub. The script **fetches this automatically** and caches it as `data_cache/israel_alerts_github.csv`. No Israeli IP or API key needed; it works globally.

## Predict today’s strikes (per day or per minute)

The script `rocket_strike_hazard_nn.py` trains a hazard NN and predicts strike probability, then compares with **today’s actual alerts** when available.

- **Per minute** (default): `python rocket_strike_hazard_nn.py --resolution minute`  
  Requires Kaggle CSV with a **datetime** column. Outputs **P(strike in next minute)** (or `--horizon-minutes N`).
- **Per day**: `python rocket_strike_hazard_nn.py --resolution day`  
  Uses Kaggle daily aggregates or curated CSV. Outputs **P(strike in next day)**.

## 1. GitHub (auto-fetched) – primary training data
- **Source:** https://github.com/dleshem/israel-alerts-data (CSV: `israel-alerts.csv`)
- The script downloads it on first run and caches to `data_cache/israel_alerts_github.csv` (refresh after 24h by default).
- Provides **daily** and **minute** resolution (alert timestamps). No geo-restriction.

## 2. Kaggle – optional local file

**Dataset:** [Rocket alerts in Israel made by "Tzeva Adom"](https://www.kaggle.com/datasets/sab30226/rocket-alerts-in-israel-made-by-tzeva-adom)

1. Download the dataset from Kaggle (CSV).
2. Put the CSV in this folder (`data_cache/`) with one of these names:
   - `rocket_alerts_tzeva_adom.csv`
   - `rocket-alerts-in-israel-made-by-tzeva-adom.csv`
   - `alerts.csv`
   - `red_alert.csv`
3. The CSV must have at least one **date** or **datetime** column (e.g. `date`, `time`, `תאריך`).
   - For **per-minute resolution**: the column must parse as full datetime (date + time). The script builds a minute-level timeline and trains **P(strike in next minute)**.
   - For **per-day resolution**: the script aggregates by day and trains **P(strike in next day)**.

If neither GitHub nor Kaggle is available, the script falls back to `iran_rocket_strikes_israel.csv` (daily only).

## 3. Pikud Haoref (Oref) – today’s actual alerts (validation)

**Live API:** `https://www.oref.org.il/WarningMessages/alert/alerts.json`  
**MCP (optional):** [Pikud Haoref Real-Time Alert System](https://lobehub.com/mcp/leonmelamud-pikud-a-oref-mcp)

- The script tries to fetch **today’s alerts** from the Oref API. **The API often only works from Israeli IPs** (geo-restriction), so the live fetch may fail. In that case you can still compare with real data:
  1. Configure the [Pikud Haoref MCP](https://lobehub.com/mcp/leonmelamud-pikud-a-oref-mcp) in Cursor (or run the middleware service).
  2. Use the MCP tool **get_alert_history** (e.g. for today or last 24h).
  3. Save the JSON response as **`data_cache/oref_alerts_today.json`**. Accepted formats:
     - A JSON array of alerts, e.g. `["תל אביב", "רמת גן"]` or `[{"date": "2026-02-28", ...}, ...]`
     - Or an object with a `"data"` key: `{"data": ["תל אביב", ...]}`

The script will load that file if present and report “today’s actual alerts” next to the predicted probability.
