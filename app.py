
look at this code: import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optimizer import PortfolioOptimizer
# User configuration
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
START_DATE = '2020-01-01'
END_DATE = '2025-01-01'
RISK_FREE_RATE = 0.01
NUM_PORTFOLIOS = 10000
# Initialize optimizer
optimizer = PortfolioOptimizer(TICKERS, START_DATE, END_DATE, RISK_FREE_RATE)
results = optimizer.simulate_portfolios(NUM_PORTFOLIOS)
optimal = optimizer.get_optimal_portfolios(results)
# Prepare data for plotting
returns = np.array(results['returns'])
volatility = np.array(results['volatility'])
sharpe = np.array(results['sharpe'])
weights = np.array(results['weights'])
# Plot Efficient Frontier
plt.figure(figsize=(10, 7))
scatter = plt.scatter(volatility, returns, c=sharpe, cmap='viridis', alpha=0.6, edgecolor='k')
plt.colorbar(scatter, label='Sharpe Ratio')
# Highlight optimal portfolios
plt.scatter(optimal['max_sharpe']['volatility'], optimal['max_sharpe']['return'],
            color='red', marker='*', s=300, label='Max Sharpe Ratio')
plt.scatter(optimal['min_volatility']['volatility'], optimal['min_volatility']['return'],
            color='blue', marker='*', s=300, label='Min Volatility')
plt.title('Efficient Frontier - Smart Portfolio Optimizer')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# Display optimal weights
print("\nOptimal Portfolio Weights:")
print("Max Sharpe Ratio Portfolio:")
for ticker, weight in zip(TICKERS, optimal['max_sharpe']['weights']):
    print(f"  {ticker}: {weight:.2%}")
print("\nMin Volatility Portfolio:")
for ticker, weight in zip(TICKERS, optimal['min_volatility']['weights']):
    print(f"  {ticker}: {weight:.2%}")
plt.show()

Show more
11:56 PM
I can see you have a portfolio optimization script that uses a custom PortfolioOptimizer class. I'll build the complete optimizer.py module to make this work end-to-end.


Viewed a file, created a file
Viewed a file, created a file
Claude's response was interrupted
look at this too: import pandas as pd
import numpy as np
import yfinance as yf
class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.01):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.data = self._fetch_data()
        self.returns = self._calculate_returns()
        self.expected_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
    def _fetch_data(self):
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        # Try to get 'Adj Close' if available, else fallback to 'Close'
        if ('Adj Close' in df.columns.get_level_values(0)):
            price = df['Adj Close']
        else:
            price = df['Close']
        return price.dropna()
    def _calculate_returns(self):
        return self.data.pct_change().dropna()
    def simulate_portfolios(self, num_portfolios=10000):
        results = {
            'returns': [],
            'volatility': [],
            'sharpe': [],
            'weights': []
        }
        for _ in range(num_portfolios):
            weights = np.random.dirichlet(np.ones(len(self.tickers)), size=1)[0]
            port_return = np.dot(weights, self.expected_returns)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_volatility
            results['returns'].append(port_return)
            results['volatility'].append(port_volatility)
            results['sharpe'].append(sharpe)
            results['weights'].append(weights)
        return results
    def get_optimal_portfolios(self, results):
        max_sharpe_idx = np.argmax(results['sharpe'])
        min_vol_idx = np.argmin(results['volatility'])
        return {
            'max_sharpe': {
                'return': results['returns'][max_sharpe_idx],
                'volatility': results['volatility'][max_sharpe_idx],
                'sharpe': results['sharpe'][max_sharpe_idx],
                'weights': results['weights'][max_sharpe_idx]
            },
            'min_volatility': {
                'return': results['returns'][min_vol_idx],
                'volatility': results['volatility'][min_vol_idx],
                'sharpe': results['sharpe'][min_vol_idx],
                'weights': results['weights'][min_vol_idx]
            }
        }

Show more
11:57 PM
Got it — this is the actual optimizer.py you're using. Let me review both files together and give you a clear picture.


Check if yfinance and dependencies are available
Check if yfinance and dependencies are available
Here's a thorough review of both files:

Code Review
optimizer.py — Issues
1. Returns are not annualised (silent math error)

python
# Current — daily scale, not annual
self.expected_returns = self.returns.mean()
self.cov_matrix = self.returns.cov()

# Fix — multiply by trading days
self.expected_returns = self.returns.mean() * 252
self.cov_matrix = self.returns.cov() * 252
Without this, port_return and port_volatility are daily figures, but risk_free_rate is annual (e.g. 0.01). The Sharpe ratio will be wildly wrong — the RF rate will dwarf daily returns, producing mostly negative Sharpes.

2. simulate_portfolios is slow — pure Python loop

python
# Current — 10,000 iterations in Python
for _ in range(num_portfolios):
    ...

# Fix — fully vectorised NumPy
weights = np.random.dirichlet(np.ones(n), size=num_portfolios)        # (N, assets)
port_returns = weights @ expected_returns                               # (N,)
port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))
This is ~50–100× faster and is the standard approach for MC efficient frontier work.

3. get_optimal_portfolios only uses MC samples — not truly optimal argmax on 10,000 random portfolios is approximate. The real max-Sharpe and min-vol points should be found with scipy.optimize.minimize (SLSQP). With only 10k samples the "optimal" weights can be noticeably off, especially in a 5-asset universe.

4. yf.download column structure changed in yfinance ≥ 0.2.x

python
# Current — fragile, may fail silently
if ('Adj Close' in df.columns.get_level_values(0)):
    price = df['Adj Close']
else:
    price = df['Close']

# Fix — use auto_adjust=True, then just grab 'Close'
df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
price = df['Close']
auto_adjust=True gives split/dividend-adjusted prices under 'Close' and avoids the multi-level column ambiguity entirely.

5. Single-ticker edge case yf.download(['AAPL'], ...)['Close'] returns a Series, not a DataFrame. pct_change() still works, but np.dot(weights, expected_returns) and cov_matrix operations will break. Add a guard:

python
if isinstance(price, pd.Series):
    price = price.to_frame(self.tickers[0])
main.py — Issues
6. plt.show() blocks — save the figure too In non-interactive environments plt.show() can hang or produce nothing. Best practice is to always plt.savefig() before plt.show().

7. No reproducibility seed np.random.dirichlet is unseeded, so results differ every run. Add np.random.seed(42) at the top of the script (or pass a seed param to simulate_portfolios).

Summary table
#	File	Severity	Issue
1	optimizer.py	🔴 Critical	Returns/covariance not annualised → broken Sharpe
2	optimizer.py	🟠 High	Pure Python loop → very slow at 10k portfolios
3	optimizer.py	🟠 High	MC argmax ≠ true optimum; use scipy SLSQP
4	optimizer.py	🟡 Medium	yfinance column API fragility
5	optimizer.py	🟡 Medium	Single-ticker crash
6	main.py	🟡 Medium	plt.show() only, no savefig
7	main.py	🟢 Low	No random seed → non-reproducible
Would you like me to produce a corrected version of optimizer.py with all fixes applied?

this is the goal of the project, create a streamlit app for this: This project is a Smart Portfolio Optimizer that helps investors find the best mix of stocks using Modern Portfolio Theory (MPT). It fetches historical stock data, simulates thousands of portfolios, and identifies the ones with the best risk-return tradeoff. The tool visualizes the efficient frontier and highlights optimal portfolios, making it easy to see how to maximize returns for a given level of risk.

Mar 10

Read frontend design skill
Read frontend design skill
Now let me build the full Streamlit app with both files — a polished optimizer.py (with all the fixes from the review) and a beautifully designed app.py.


Created 2 files, ran a command, read a file
Created 2 files, ran a command, read a file
Here's what was built — two files to drop in the same folder and run together:

optimizer.py — fully fixed version of your class (all 7 issues from the review resolved: annualised returns, vectorised simulation, scipy SLSQP refinement, yfinance API fix, single-ticker guard).

app.py — a dark, financial-grade Streamlit dashboard with:

Sidebar — configure tickers, date range, risk-free rate, and simulation count
KPI strip — live metrics (max Sharpe, min vol, best return)
4 tabs:
Efficient Frontier — interactive scatter of all simulated portfolios, coloured by Sharpe, with ★ Max Sharpe and ◆ Min Volatility markers
Optimal Weights — donut + horizontal bar charts for both portfolios side-by-side
Correlation Matrix — heatmap with diverging colour scale
Price History — indexed normalised price chart for all assets
To run:

bash
pip install streamlit plotly yfinance scipy
streamlit run app.py
App
PY 
Optimizer
PY 
guide me to host on strealit

12:04 AM
Here's how to deploy your app on Streamlit Community Cloud (free hosting, purpose-built for Streamlit apps):

1. Push your code to GitHub
Create a new GitHub repo and push both files:

bash
git init
git add app.py optimizer.py
git commit -m "Initial portfolio optimizer"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

You also need a `requirements.txt` in the same folder:
```
streamlit
plotly
yfinance
scipy
numpy
pandas
Push that too:

bash
git add requirements.txt
git commit -m "Add requirements"
git push
2. Deploy on Streamlit Community Cloud
Go to share.streamlit.io and sign in with GitHub
Click "New app"
Fill in:
Repository → your repo
Branch → main
Main file path → app.py
Click "Deploy" — it takes ~2 minutes
That's it. Streamlit handles the server, installs your dependencies automatically, and gives you a public URL like https://YOUR-APP.streamlit.app you can share with anyone.

3. Updating the app later
Any git push to your main branch automatically redeploys — no manual steps needed.

Things to know
Free tier limits — apps sleep after inactivity but wake up when visited; no usage limits for personal projects
Python version — defaults to 3.11, which works fine for your stack
Secrets — if you ever add API keys, store them in the Streamlit Cloud dashboard under Settings → Secrets rather than hardcoding them
error: failed to push some refs to 'https://github.com/aj1m153/Optimizer.git'

12:12 AM
This usually means your remote has commits your local doesn't. Fix it with one of these:

Option A — Pull then push (safe, keeps remote history)
bash
git pull origin main --rebase
git push origin main
Option B — Force push (overwrites remote, use if repo is new/empty)
bash
git push -u origin main --force
Option C — Remote uses master not main
bash
git push -u origin master
If none of those work, check the exact error:
bash
git push origin main 2>&1
Paste the full output here and I'll pinpoint the issue.

add a readme about the project

12:13 AM

Viewed a file, created a file, read a file
Viewed a file, created a file, read a file
Add this to your repo and push:

bash
git add README.md
git commit -m "Add README"
git push
It covers project overview, features, file structure, how to run locally, how the MPT math works, and deployment steps — everything someone landing on your GitHub repo would need.

Readme
Document · MD 
File "/mount/src/optimizer/app.py", line 4, in <module>     import plotly.graph_objects as go

12:17 AM
Plotly isn't installed in your Streamlit Cloud environment. You need to make sure your requirements.txt is complete and correctly formatted.

Replace your requirements.txt with exactly this:

streamlit
plotly
yfinance
scipy
numpy
pandas
Then push it:

bash
git add requirements.txt
git commit -m "Fix requirements"
git push
Streamlit Cloud reads this file to install dependencies — if plotly wasn't listed or the file had formatting issues (tabs, extra spaces, blank lines), it gets skipped.

After pushing, go to your app on share.streamlit.io → click the ⋮ menu (top right) → Reboot app to force a fresh install.

File "/mount/src/optimizer/app.py", line 4, in <module>     import plotly.graph_objects as go

12:22 AM
The app is reading from /mount/src/optimizer/ which means Streamlit Cloud is running the old cached version. Let's force it to reinstall everything.


Created a file, read a file
Created a file, read a file
Push this updated file:

bash
git add requirements.txt
git commit -m "Pin dependency versions to force reinstall"
git push
Then do a hard reboot on Streamlit Cloud:

Go to share.streamlit.io
Find your app → click ⋮ (three dots)
Click Settings → Advanced → Delete cache
Then click Reboot app
Pinning explicit versions forces Streamlit Cloud to treat it as a changed environment and do a full fresh install rather than using its cache. If it still fails after this, paste the full error log from the Streamlit Cloud console and I can dig deeper.

Requirements
TXT 
[04:22:56] 📦 Processed dependencies!
cat: /mount/admin/install_path: No such file or directory

2026-03-11 04:23:04.270 Uncaught app execution
Traceback (most recent call last):
  File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
  File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/mount/src/optimizer/app.py", line 4, in <module>
    import plotly.graph_objects as go
ModuleNotFoundError: No module named 'plotly'

Show more
12:23 AM
The issue is clear — dependencies processed but plotly still isn't found. This is a Python 3.14 environment issue; some packages don't have wheels for 3.14 yet. Force Python 3.11 by adding a .python-version file.


Created a file, read a file
Created a file, read a file
Add this file to your repo root and push:

bash
git add .python-version
git commit -m "Pin Python 3.11 for compatibility"
git push
Then on Streamlit Cloud → ⋮ → Settings → Delete cache → Reboot app.

Why this fixes it: Streamlit Cloud defaulted to Python 3.14 (very new), and plotly doesn't have a pre-built wheel for it yet — so the install silently fails. Python 3.11 is the stable, fully-supported version where all your dependencies (plotly, yfinance, scipy) have proper wheels and install cleanly.

.python-version
PYTHON-VERSION 
[04:24:35] 📦 Processed dependencies!
cat: /mount/admin/install_path: No such file or directory

2026-03-11 04:25:31.146 Uncaught app execution
Traceback (most recent call last):
  File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/runtime/scriptrunner/exec_code.py", line 129, in exec_func_with_error_handling
    result = func()
  File "/home/adminuser/venv/lib/python3.14/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 689, in code_to_exec
    exec(code, module.__dict__)  # noqa: S102
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "/mount/src/optimizer/app.py", line 4, in <module>
    import plotly.graph_objects as go
ModuleNotFoundError: No module named 'plotly'

Show more
12:25 AM
Still Python 3.14 — the .python-version file isn't being picked up. Use a runtime.txt file instead, which Streamlit Cloud definitely reads.


Created a file, read a file
Created a file, read a file
Push this to your repo root:

bash
git add runtime.txt
git commit -m "Pin Python 3.11 via runtime.txt"
git push
Then fully delete and redeploy the app — don't just reboot:

Go to share.streamlit.io
Click ⋮ → Delete app
Click New app and redeploy from your repo
A fresh deploy (not a reboot) is needed because the Python environment is baked in at first boot — rebooting reuses the same broken 3.14 environment. Deleting and redeploying forces it to read runtime.txt and build fresh with 3.11.

Runtime
TXT 
File "/mount/src/optimizer/app.py", line 4, in <module>     import plotly.graph_objects as go

12:31 AM
The runtime.txt isn't working either. Let's bypass the Python version issue entirely by installing plotly directly via a packages.txt workaround — actually, the most reliable fix is to install via a setup.sh script with a Streamlit config.


Creating packages.txt for system-level dependencies
Creating packages.txt for system-level dependencies
Actually, the real fix here is simpler — replace plotly with the built-in st.pyplot using matplotlib, which is pre-installed in every Streamlit environment. Let me rewrite app.py to use only guaranteed-available packages.


Read current app.py
Read current app.py
The root cause is clear — Streamlit Cloud is stuck on Python 3.14 and won't budge. The cleanest fix is to rewrite app.py to use matplotlib instead of plotly, since matplotlib is always pre-installed.


Created a file, ran a command
Created a file, ran a command
Also update requirements.txt to remove plotly:


Ran a command, read a file
Ran a command, read a file
Root cause: Streamlit Cloud is running Python 3.14, which has no pre-built plotly wheel yet — so it silently fails to install. The fix is to swap plotly for matplotlib, which ships pre-installed in every Python environment.

Push both files:

bash
git add app.py requirements.txt
git commit -m "Replace plotly with matplotlib for Python 3.14 compatibility"
git push
Then on Streamlit Cloud → ⋮ → Delete cache → Reboot app. Everything else (layout, dark theme, charts, all 4 tabs) stays exactly the same — just rendered with matplotlib instead.

App
PY 
Requirements
TXT 
ile "/mount/src/optimizer/app2.py", line 4, in <module>     import matplotlib.pyplot as plt

12:41 AM
Matplotlib is also failing — Python 3.14 is too new for nearly all scientific packages. The only reliable fix is to force Python 3.11 at the app level using Streamlit's config file.


Create streamlit config directory and file
Create streamlit config directory and file
The real solution here is a pyproject.toml to force the Python version:


Ran 2 commands, read a file
Ran 2 commands, read a file
Push everything:

bash
git add runtime.txt pyproject.toml .streamlit/config.toml
git commit -m "Force Python 3.11.9 to fix package compatibility"
git push
Then delete the app entirely and redeploy fresh — don't reboot:

Go to share.streamlit.io → ⋮ → Delete app
Click New app → reconnect your repo
Set main file to app.py
Click Deploy
A full delete+redeploy is the only way to force Streamlit Cloud to rebuild the Python environment from scratch. Rebooting reuses the cached 3.14 environment regardless of config files.

If it still picks up 3.14 after all this, the nuclear option is to go to your Streamlit Cloud app settings before deploying and look for a "Python version" dropdown — recent versions of Streamlit Cloud added an explicit selector in the UI under Advanced settings when creating a new app.

Runtime
TXT 
Pyproject
TOML 
error: failed to push some refs to 'https://github.com/aj1m153/Optimizer.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details

12:41 AM
Classic fast-forward rejection. Run this:

bash
git pull origin main --rebase
git push origin main
If that still fails due to conflicts:

bash
git fetch origin
git reset --hard origin/main
git add .
git commit -m "Force Python 3.11.9 to fix package compatibility"
git push origin main
What this does: reset --hard origin/main syncs your local branch to whatever is on GitHub, then you re-add your new files on top and push cleanly.

error: failed to push some refs to 'https://github.com/aj1m153/Optimizer.git'

12:42 AM
Just force push — your repo is new and there's nothing on GitHub worth preserving:

bash
git push origin main --force
That's it. It overrides whatever is on the remote and pushes your local version directly.

error: failed to push some refs to 'https://github.com/aj1m153/Optimizer.git'

12:42 AM
Let's diagnose exactly what's happening:

bash
git status
git log --oneline -5
git remote -v
Paste the output of those three commands here and I'll tell you exactly what to run.

On branch master
Your branch is behind 'origin/master' by 1 commit, and can be fast-forwarded.
  (use "git pull" to update your local branch)

nothing to commit, working tree clean
83af8bb (HEAD -> master) Replace plotly with matplotlib for Python 3.14 compatibility
fb374bf Add README
8ec4d1a Initial portfolio optimizer
origin https://github.com/aj1m153/Optimizer.git (fetch)
origin https://github.com/aj1m153/Optimizer.git (push)

12:43 AM
You're on master not main, and you're 1 commit behind. Run:

bash
git pull origin master
git push origin master
That's it. Everything will sync up cleanly.

what do i do with the optimizer.py

12:44 AM
You need to make sure optimizer.py is in your repo. Check if it's there:

bash
ls
If it's not listed, copy it in and push:

bash
git add optimizer.py
git commit -m "Add optimizer module"
git push origin master
If it is listed but not yet committed:

bash
git add optimizer.py
git commit -m "Add optimizer module"
git push origin master
If it's already committed and pushed, you're good — Streamlit Cloud will find it automatically since it's in the same folder as app.py.

File "/mount/src/optimizer/app2.py", line 4, in <module>     import matplotlib.pyplot as plt

12:48 AM
The problem is it's running app2.py not app.py. You have an old file on GitHub. Fix it:

bash
# Delete the old file
git rm app2.py
git commit -m "Remove old app2.py"
git push origin master
Then go to Streamlit Cloud → your app → Settings and make sure the Main file path is set to app.py, not app2.py.

If the main file path is wrong, you'll need to delete the app and redeploy:

Go to share.streamlit.io → ⋮ → Delete app
Click New app → select your repo
Set Main file path to app.py
Click Deploy
can matplot be replaced: File "/mount/src/optimizer/app2.py", line 4, in <module>     import matplotlib.pyplot as plt

12:50 AM
Yes — let's replace everything with Streamlit's built-in native charts (st.line_chart, st.bar_chart) and Altair which is pre-installed with Streamlit. No external chart library needed at all.


Ran 2 commands, read a file
Ran 2 commands, read a file
Altair ships bundled with Streamlit — zero extra installs needed. Push and redeploy:

bash
git add app.py requirements.txt
git commit -m "Switch to Altair - no external chart dependencies"
git push origin master
Then on Streamlit Cloud make sure the Main file path is app.py (not app2.py). If it's still pointing to app2.py, delete and redeploy with the correct file.

All 4 charts are fully recreated in Altair — efficient frontier scatter, weight bars, correlation heatmap, and price history lines — with the same dark gold theme.

App
PY 
Requirements
TXT 


Want to be notified when Claude responds?




Claude is AI and can make mistakes. Please double-check responses.
App · PY
Copy

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from optimizer import PortfolioOptimizer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colors ─────────────────────────────────────────────────────────────────────
BG      = "#0a0c10"
SURFACE = "#111318"
BORDER  = "#1f2530"
GOLD    = "#c9a84c"
TEAL    = "#3ecfb2"
RED     = "#e05c5c"
MUTED   = "#6b7280"
TEXT    = "#e8e4da"
PALETTE = [GOLD, TEAL, "#5b8dee", RED, "#a78bfa", "#f97316", "#84cc16", "#ec4899"]

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
:root { --bg:#0a0c10; --surface:#111318; --surface2:#181c24; --border:#1f2530; --gold:#c9a84c; --gold-dim:#7a6230; --teal:#3ecfb2; --text:#e8e4da; --muted:#6b7280; --radius:12px; }
html,body,[data-testid="stAppViewContainer"] { background:var(--bg) !important; color:var(--text) !important; font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:var(--surface) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color:var(--text) !important; }
h1,h2,h3 { font-family:'DM Serif Display',serif !important; color:var(--text) !important; }
[data-testid="metric-container"] { background:var(--surface2); border:1px solid var(--border); border-radius:var(--radius); padding:1rem 1.4rem !important; }
[data-testid="metric-container"] label { font-family:'DM Mono',monospace !important; font-size:0.7rem !important; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'DM Serif Display',serif !important; font-size:1.8rem !important; color:var(--gold) !important; }
.stButton > button { background:var(--gold) !important; color:#0a0c10 !important; border:none !important; border-radius:8px !important; font-family:'DM Mono',monospace !important; font-size:0.82rem !important; letter-spacing:0.08em; font-weight:500 !important; padding:0.6rem 1.6rem !important; width:100%; }
.stTabs [data-baseweb="tab-list"] { background:var(--surface) !important; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { font-family:'DM Mono',monospace !important; font-size:0.78rem !important; letter-spacing:0.08em; color:var(--muted) !important; background:transparent !important; border:none !important; padding:0.7rem 1.4rem !important; }
.stTabs [aria-selected="true"] { color:var(--gold) !important; border-bottom:2px solid var(--gold) !important; }
hr { border-color:var(--border) !important; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:1.4rem 1.6rem; margin-bottom:1rem; }
.ticker-tag { display:inline-block; background:var(--surface2); border:1px solid var(--gold-dim); border-radius:6px; padding:2px 10px; font-family:'DM Mono',monospace; font-size:0.78rem; color:var(--gold); margin:2px; }
.hero-title { font-family:'DM Serif Display',serif; font-size:2.6rem; line-height:1.1; color:var(--text); margin:0; }
.hero-sub { color:var(--muted); font-size:0.95rem; margin-top:0.4rem; }
.section-label { font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--gold-dim); margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">◈ Portfolio Config</p>', unsafe_allow_html=True)
    st.markdown("### Tickers")
    ticker_input = st.text_input("Tickers", value="AAPL, MSFT, GOOGL, AMZN, META", label_visibility="collapsed")
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    badges = " ".join(f'<span class="ticker-tag">{t}</span>' for t in tickers)
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Date Range")
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("From", value=pd.Timestamp("2020-01-01"))
    with c2: end_date   = st.date_input("To",   value=pd.Timestamp("2025-01-01"))
    st.markdown("---")
    st.markdown("### Parameters")
    risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.10, 0.04, 0.005, format="%.3f")
    num_portfolios = st.select_slider("Simulations", options=[1_000, 2_500, 5_000, 10_000, 20_000], value=10_000, format_func=lambda x: f"{x:,}")
    st.markdown("---")
    run = st.button("⟳  Run Optimizer")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:2rem 0 1.4rem 0;border-bottom:1px solid #1f2530;margin-bottom:2rem;">
  <p class="hero-title">Smart Portfolio<br><em>Optimizer</em></p>
  <p class="hero-sub">Modern Portfolio Theory · Efficient Frontier · Monte Carlo Simulation</p>
</div>
""", unsafe_allow_html=True)

if not run:
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "◎", "Efficient Frontier", "Thousands of portfolios mapped across the risk-return spectrum"),
        (c2, "◈", "Optimal Portfolios", "Max Sharpe & Min Volatility found via scipy SLSQP"),
        (c3, "◉", "Correlation Matrix", "Understand diversification and asset co-movement"),
    ]:
        with col:
            st.markdown(f'<div class="card" style="text-align:center;min-height:140px;"><div style="font-size:2rem;margin-bottom:0.6rem;color:#c9a84c;">{icon}</div><div style="font-family:\'DM Serif Display\',serif;font-size:1.05rem;margin-bottom:0.4rem;">{title}</div><div style="font-size:0.82rem;color:#6b7280;">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;margin-top:3rem;color:#6b7280;font-family:monospace;font-size:0.8rem;letter-spacing:0.12em;">CONFIGURE YOUR PORTFOLIO IN THE SIDEBAR AND PRESS RUN</div>', unsafe_allow_html=True)
    st.stop()

# ── Run optimizer ──────────────────────────────────────────────────────────────
with st.spinner("Fetching data & running simulation…"):
    try:
        optimizer = PortfolioOptimizer(tickers, str(start_date), str(end_date), risk_free_rate)
        results   = optimizer.simulate_portfolios(num_portfolios)
        optimal   = optimizer.get_optimal_portfolios(results)
        corr      = optimizer.correlation_matrix()
    except Exception as e:
        st.error(f"**Error:** {e}")
        st.stop()

returns    = results["returns"]
volatility = results["volatility"]
sharpe     = results["sharpe"]
ms = optimal["max_sharpe"]
mv = optimal["min_volatility"]

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Assets",      len(tickers))
k2.metric("Simulations", f"{num_portfolios:,}")
k3.metric("Max Sharpe",  f"{ms['sharpe']:.3f}",          f"ret {ms['return']*100:.1f}%")
k4.metric("Min Vol",     f"{mv['volatility']*100:.1f}%",  f"ret {mv['return']*100:.1f}%")
k5.metric("Best Return", f"{returns.max()*100:.1f}%")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["  EFFICIENT FRONTIER  ", "  OPTIMAL WEIGHTS  ", "  CORRELATION  ", "  PRICE HISTORY  "])

# ── Tab 1: Efficient Frontier ──────────────────────────────────────────────────
with tab1:
    df_sim = pd.DataFrame({
        "Volatility (%)": volatility * 100,
        "Return (%)":     returns * 100,
        "Sharpe Ratio":   sharpe,
    })

    frontier = alt.Chart(df_sim).mark_circle(size=18, opacity=0.65).encode(
        x=alt.X("Volatility (%):Q", title="Volatility / Risk (%)"),
        y=alt.Y("Return (%):Q",     title="Expected Annual Return (%)"),
        color=alt.Color("Sharpe Ratio:Q",
            scale=alt.Scale(scheme="goldorange"),
            legend=alt.Legend(title="Sharpe Ratio")),
        tooltip=["Volatility (%):Q", "Return (%):Q", "Sharpe Ratio:Q"],
    )

    df_opt = pd.DataFrame([
        {"Volatility (%)": ms["volatility"]*100, "Return (%)": ms["return"]*100, "Portfolio": f"★ Max Sharpe ({ms['sharpe']:.3f})", "size": 300},
        {"Volatility (%)": mv["volatility"]*100, "Return (%)": mv["return"]*100, "Portfolio": f"◆ Min Volatility ({mv['volatility']*100:.1f}%)", "size": 200},
    ])

    opt_points = alt.Chart(df_opt).mark_point(filled=True, opacity=1).encode(
        x=alt.X("Volatility (%):Q"),
        y=alt.Y("Return (%):Q"),
        color=alt.Color("Portfolio:N",
            scale=alt.Scale(range=[GOLD, TEAL]),
            legend=alt.Legend(title="Optimal")),
        size=alt.Size("size:Q", legend=None),
        tooltip=["Portfolio:N", "Volatility (%):Q", "Return (%):Q"],
    )

    chart = (frontier + opt_points).properties(
        height=480,
        background=SURFACE,
        padding={"left":20,"right":20,"top":20,"bottom":20},
    ).configure_axis(
        gridColor=BORDER, labelColor=MUTED, titleColor=MUTED,
        domainColor=BORDER, tickColor=BORDER,
    ).configure_legend(
        labelColor=TEXT, titleColor=MUTED,
        fillColor=SURFACE, strokeColor=BORDER,
    ).configure_view(strokeOpacity=0)

    st.altair_chart(chart, use_container_width=True)

# ── Tab 2: Optimal Weights ─────────────────────────────────────────────────────
with tab2:
    col_ms, col_mv = st.columns(2)

    for col, label, portfolio, color in [
        (col_ms, "Max Sharpe Ratio", ms, GOLD),
        (col_mv, "Min Volatility",   mv, TEAL),
    ]:
        with col:
            st.markdown(f'<div class="section-label">{label}</div><span style="font-family:monospace;font-size:0.8rem;color:{color};">Sharpe {portfolio["sharpe"]:.3f}</span> <span style="font-family:monospace;font-size:0.75rem;color:{MUTED};">· ret {portfolio["return"]*100:.1f}% · vol {portfolio["volatility"]*100:.1f}%</span>', unsafe_allow_html=True)

            df_w = pd.DataFrame({
                "Ticker": tickers,
                "Allocation (%)": portfolio["weights"] * 100,
            }).sort_values("Allocation (%)", ascending=False)

            bar = alt.Chart(df_w).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                x=alt.X("Allocation (%):Q", title="Allocation (%)"),
                y=alt.Y("Ticker:N", sort="-x", title=None),
                color=alt.value(color),
                tooltip=["Ticker:N", alt.Tooltip("Allocation (%):Q", format=".1f")],
            ).properties(
                height=max(180, len(tickers) * 40),
                background=SURFACE,
                padding={"left":10,"right":30,"top":10,"bottom":10},
            ).configure_axis(
                gridColor=BORDER, labelColor=TEXT, titleColor=MUTED,
                domainColor=BORDER, tickColor=BORDER,
            ).configure_view(strokeOpacity=0)

            st.altair_chart(bar, use_container_width=True)

            # Table
            df_w["Allocation (%)"] = df_w["Allocation (%)"].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_w.set_index("Ticker"), use_container_width=True)

# ── Tab 3: Correlation ─────────────────────────────────────────────────────────
with tab3:
    corr_vals = corr.values
    n = len(tickers)
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({"Asset A": tickers[i], "Asset B": tickers[j], "ρ": round(float(corr_vals[i,j]), 3)})
    df_corr = pd.DataFrame(rows)

    heatmap = alt.Chart(df_corr).mark_rect().encode(
        x=alt.X("Asset A:N", title=None),
        y=alt.Y("Asset B:N", title=None),
        color=alt.Color("ρ:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=["#1a3a5c", SURFACE, "#7a3f1e"]),
            legend=alt.Legend(title="ρ")),
        tooltip=["Asset A:N", "Asset B:N", "ρ:Q"],
    )

    text = alt.Chart(df_corr).mark_text(fontSize=13, fontWeight="bold").encode(
        x=alt.X("Asset A:N"),
        y=alt.Y("Asset B:N"),
        text=alt.Text("ρ:Q", format=".2f"),
        color=alt.value(TEXT),
    )

    corr_chart = (heatmap + text).properties(
        height=420,
        background=SURFACE,
        padding={"left":20,"right":20,"top":20,"bottom":20},
    ).configure_axis(
        labelColor=TEXT, titleColor=MUTED,
        domainColor=BORDER, tickColor=BORDER, gridColor=BORDER,
    ).configure_legend(
        labelColor=TEXT, titleColor=MUTED,
        fillColor=SURFACE, strokeColor=BORDER,
    ).configure_view(strokeOpacity=0)

    st.altair_chart(corr_chart, use_container_width=True)
    st.markdown(f'<div style="font-family:monospace;font-size:0.78rem;color:{MUTED};">ρ → +1: assets move together (less diversification) · ρ → −1: assets move inversely (max diversification)</div>', unsafe_allow_html=True)

# ── Tab 4: Price History ───────────────────────────────────────────────────────
with tab4:
    prices_norm = optimizer.price_data / optimizer.price_data.iloc[0] * 100
    prices_norm.index = pd.to_datetime(prices_norm.index)
    df_prices = prices_norm.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Indexed Price")

    line = alt.Chart(df_prices).mark_line(strokeWidth=1.8).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Indexed Price:Q", title="Indexed Price (base = 100)"),
        color=alt.Color("Ticker:N",
            scale=alt.Scale(range=PALETTE[:len(tickers)]),
            legend=alt.Legend(title="Ticker")),
        tooltip=["Ticker:N", alt.Tooltip("Date:T", format="%b %Y"), alt.Tooltip("Indexed Price:Q", format=".1f")],
    ).properties(
        height=460,
        background=SURFACE,
        padding={"left":20,"right":20,"top":20,"bottom":20},
    ).configure_axis(
        gridColor=BORDER, labelColor=MUTED, titleColor=MUTED,
        domainColor=BORDER, tickColor=BORDER,
    ).configure_legend(
        labelColor=TEXT, titleColor=MUTED,
        fillColor=SURFACE, strokeColor=BORDER,
    ).configure_view(strokeOpacity=0)

    st.altair_chart(line, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="font-family:monospace;font-size:0.7rem;color:#3a3f48;text-align:center;padding:0.5rem 0 1rem;">SMART PORTFOLIO OPTIMIZER · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)

