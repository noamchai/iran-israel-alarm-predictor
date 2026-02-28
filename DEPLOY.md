# Deploy IRAN vs ISRAEL round 2 alarm predictor to the internet

The app is a Flask server that trains on startup and listens on `PORT` (set by the host). Use one of the options below.

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
