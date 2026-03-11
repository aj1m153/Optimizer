# ◈ Smart Portfolio Optimizer

A Streamlit web app that helps investors find the optimal mix of stocks using **Modern Portfolio Theory (MPT)**. It fetches historical price data, simulates thousands of portfolios, and identifies the best risk-return tradeoffs — all in an interactive dashboard.

---

## Features

- **Efficient Frontier** — Simulates up to 20,000 random portfolios and plots them across the risk-return spectrum, coloured by Sharpe Ratio
- **Optimal Portfolios** — Finds the true Max Sharpe and Min Volatility portfolios using `scipy` SLSQP optimisation (not just the best MC sample)
- **Weight Visualisation** — Donut and bar charts showing allocation breakdowns for each optimal portfolio
- **Correlation Matrix** — Heatmap of asset co-movement to assess diversification
- **Price History** — Indexed normalised price chart for all selected assets
- **Fully configurable** — Set your own tickers, date range, risk-free rate, and simulation count from the sidebar

---

## Project Structure

```
├── app.py            # Streamlit UI and charts
├── optimizer.py      # PortfolioOptimizer class (data fetching, simulation, optimisation)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. Enter comma-separated ticker symbols in the sidebar (e.g. `AAPL, MSFT, GOOGL`)
2. Select your date range and risk-free rate
3. Choose how many portfolios to simulate (1,000–20,000)
4. Press **Run Optimizer**
5. Explore the four result tabs

---

## How It Works

### Data
Historical price data is fetched from Yahoo Finance via `yfinance`. Daily returns are computed and annualised (×252 trading days).

### Monte Carlo Simulation
Random portfolio weights are drawn using a Dirichlet distribution (vectorised with NumPy), ensuring weights sum to 1 and are all non-negative. For each portfolio, annualised return, volatility, and Sharpe Ratio are computed.

### Optimisation
The best Monte Carlo sample is used to seed a `scipy.optimize.minimize` (SLSQP) run, which finds the mathematically true optimal portfolio — not just the best of the random samples.

### Sharpe Ratio
```
Sharpe = (Portfolio Return − Risk-Free Rate) / Portfolio Volatility
```
A higher Sharpe means more return per unit of risk taken.

---

## Requirements

```
streamlit
plotly
yfinance
scipy
numpy
pandas
```

---

## Deployment

Hosted on [Streamlit Community Cloud](https://share.streamlit.io). To deploy your own:

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **main file** to `app.py`
4. Click **Deploy**

---

## Disclaimer

This tool is for **educational purposes only** and does not constitute financial advice. Past performance is not indicative of future results.
