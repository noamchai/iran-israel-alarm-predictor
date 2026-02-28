# Deploy IRAN vs ISRAEL round 2 alarm predictor to the internet

**Recommended for Render free tier (512 MB):** Train locally, upload the **model** (not the data). The server loads the model and **fetches recent data from the web every 5 minutes** to update probabilities and graphs. No training on the server, and no need to keep the full dataset.

---

## Quick upload (Render, 512 MB)

1. **Export the model locally:** `python export_live_state.py`  
   This creates `data_cache/model.joblib` (and optionally `data_cache/live_state.json`).

2. **Push to GitHub and include the model** (do not ignore `data_cache/model.joblib`):
   ```bash
   git add data_cache/model.joblib rocket_strike_app.py rocket_strike_hazard_nn.py requirements.txt
   # If you have .gitignore, ensure data_cache/model.joblib is tracked (e.g. git add -f data_cache/model.joblib)
   git commit -m "Rocket strike predictor with pretrained model"
   git push origin main
   ```

3. **On Render:** New Web Service → connect repo →
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python rocket_strike_app.py`
   - **Environment:** Add variable **REQUIRE_PRETRAINED** = **1** (so the server never trains and avoids out-of-memory; it will only run if `model.joblib` is present).
   - Do **not** set `USE_STATIC_STATE` (so the app loads the model and refreshes data every 5 min).
   - Create Web Service.

4. **URL:** Use the URL Render gives you. Free tier may sleep after ~15 min idle; first load after sleep can take 1–2 min.

If you see **"No open ports"** or **"Out of memory"**: the app is trying to train because it didn’t find `model.joblib`. Fix by (1) committing and pushing `data_cache/model.joblib`, and (2) setting **REQUIRE_PRETRAINED=1** in Render so the server never attempts training.

---

## Upload model, update live (recommended)

1. **On your machine** (full data, enough RAM):
   ```bash
   python export_live_state.py
   ```
   This trains on full data and writes:
   - `data_cache/model.joblib` – trained model + scaler (upload this)
   - `data_cache/live_state.json` – optional snapshot

2. **Commit and push** (include the model file):
   ```bash
   git add data_cache/model.joblib export_live_state.py rocket_strike_app.py rocket_strike_hazard_nn.py
   git commit -m "Add pretrained model; server loads it and updates from live data"
   git push origin main
   ```

3. **On Render**: Deploy **without** setting `USE_STATIC_STATE`. The app will:
   - Load the pretrained model from `data_cache/model.joblib`
   - Fetch only the **last ~120k rows** of alert data (streaming if needed) so it fits in 512 MB
   - Update probabilities and graphs every **5 minutes** from fresh data

4. **No re-deploy needed to refresh**: The server keeps fetching recent data and updating the dashboard. To refresh the **model** itself (e.g. retrain on newer history), run `export_live_state.py` again and push the new `model.joblib`.

---

## Static graphs only (no live updates)

1. Run `python export_live_state.py` and commit `data_cache/live_state.json`.
2. On Render set **USE_STATIC_STATE** = **1**. The app will only load the JSON and serve it (no model, no data fetch). To refresh, re-export and push the JSON.

---

## Full app (train on server)

The app is a Flask server that can also train on startup and listen on `PORT`. Use one of the options below if you have enough RAM (e.g. paid instance).

---

## Option 1: Render (free tier, recommended)

1. **Push your code to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Rocket strike alarm predictor"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Sign up**: Go to [render.com](https://render.com) and sign up (GitHub login is easiest).

3. **New Web Service**:
   - Dashboard → **New +** → **Web Service**
   - Connect your GitHub repo and select the repo that contains `rocket_strike_app.py`
   - **Settings**:
     - **Name**: e.g. `iran-israel-alarm-predictor`
     - **Runtime**: Python 3
     - **Build command**: `pip install -r requirements.txt`
     - **Start command**: `python rocket_strike_app.py`
   - Render sets `PORT` automatically; the app already uses it.
   - **Create Web Service**

4. **First deploy**: The first run can take several minutes (downloading data, training). If it times out, increase "Start command" timeout in the service settings or use a paid plan for more resources.

5. **Your URL**: After deploy you get a URL like `https://iran-israel-alarm-predictor.onrender.com`.

**Note**: Free tier sleeps after ~15 min of no traffic; the first request after sleep may take 1–2 minutes while the app wakes and retrains.

---

## Option 2: Railway

1. Go to [railway.app](https://railway.app), sign up with GitHub.
2. **New Project** → **Deploy from GitHub repo** → select your repo.
3. Railway will detect Python. Set **Start Command** to: `python rocket_strike_app.py`
4. Add a **Variable**: `PORT` is usually set by Railway automatically; if not, add `PORT=5050`.
5. Deploy. Use the generated public URL.

---

## Option 3: Your own server (VPS)

On a Linux server (DigitalOcean, Linode, AWS EC2, etc.):

```bash
# Install Python 3.11, pip, and dependencies
sudo apt update && sudo apt install -y python3.11 python3-pip
cd /path/to/FinanceExp
pip install -r requirements.txt

# Run in background (port 80 needs sudo or use 5050 and reverse proxy)
export PORT=5050
nohup python3 rocket_strike_app.py > app.log 2>&1 &
```

To keep it running after logout, use **systemd** or **tmux/screen**. For HTTPS and a domain, put **Nginx** or **Caddy** in front and use Let’s Encrypt.

---

## Checklist before deploy

- [ ] Code is in a Git repo (GitHub/GitLab).
- [ ] `requirements.txt` includes: `flask`, `numpy`, `pandas`, `scikit-learn`, `requests`.
- [ ] App uses `PORT` from the environment (this app does: `port = int(os.environ.get("PORT", "5050"))`).
- [ ] No secrets or API keys hardcoded (this app doesn’t need any for basic run).

## Data on the cloud

The app fetches data from GitHub (dleshem/israel-alerts-data) and optionally Oref. No local files are required for a basic run; the first request may be slow while the model trains.
