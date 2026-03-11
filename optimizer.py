import numpy as np
import pandas as pd
import urllib.request
import io


class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.01):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(tickers)
        self.price_data = self._fetch_data()
        self.returns = self.price_data.pct_change().dropna()
        self.expected_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252

    def _fetch_stooq(self, ticker, start, end):
        """Fetch daily close prices from stooq.com (no auth needed)."""
        d1 = pd.Timestamp(start).strftime("%Y%m%d")
        d2 = pd.Timestamp(end).strftime("%Y%m%d")
        symbol = ticker.lower() + ".us"
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={d1}&d2={d2}&i=d"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8")
        if "No data" in content or len(content.strip()) < 50:
            raise ValueError(f"No data returned for {ticker}. Check ticker symbol.")
        df = pd.read_csv(io.StringIO(content), parse_dates=["Date"], index_col="Date")
        df = df.sort_index()
        series = df["Close"].rename(ticker).astype(float)
        return series.dropna()

    def _fetch_data(self):
        frames = {}
        for ticker in self.tickers:
            frames[ticker] = self._fetch_stooq(ticker, self.start_date, self.end_date)
        df = pd.DataFrame(frames)
        return df.dropna()

    def _portfolio_performance(self, weights):
        w = np.asarray(weights)
        ret = float(w @ self.expected_returns.values)
        vol = float(np.sqrt(w @ self.cov_matrix.values @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _project_simplex(self, v):
        """Project vector onto probability simplex (weights >= 0, sum = 1)."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def _minimize_projected_gradient(self, objective_fn, seed_weights):
        """Projected gradient descent on the simplex — no scipy needed."""
        w = np.array(seed_weights, dtype=float)
        w = self._project_simplex(w)
        lr = 0.005
        best_w = w.copy()
        best_val = objective_fn(w)
        eps = 1e-6
        for iteration in range(8000):
            grad = np.zeros(self.num_assets)
            f0 = objective_fn(w)
            for i in range(self.num_assets):
                w[i] += eps
                grad[i] = (objective_fn(w) - f0) / eps
                w[i] -= eps
            w = self._project_simplex(w - lr * grad)
            val = objective_fn(w)
            if val < best_val:
                best_val = val
                best_w = w.copy()
        return best_w

    def simulate_portfolios(self, num_portfolios=10000, seed=42):
        rng = np.random.default_rng(seed)
        raw = rng.random((num_portfolios, self.num_assets))
        weights = raw / raw.sum(axis=1, keepdims=True)
        er = self.expected_returns.values
        cov = self.cov_matrix.values
        port_returns = weights @ er
        port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
        port_sharpes = (port_returns - self.risk_free_rate) / np.where(port_vols > 0, port_vols, np.nan)
        return {
            "returns": port_returns,
            "volatility": port_vols,
            "sharpe": port_sharpes,
            "weights": weights,
        }

    def get_optimal_portfolios(self, simulation_results):
        sharpes = simulation_results["sharpe"]
        vols    = simulation_results["volatility"]
        weights = simulation_results["weights"]
        seed_ms = weights[np.nanargmax(sharpes)].copy()
        opt_ms  = self._minimize_projected_gradient(lambda w: -self._portfolio_performance(w)[2], seed_ms)
        r_ms, v_ms, s_ms = self._portfolio_performance(opt_ms)
        seed_mv = weights[np.argmin(vols)].copy()
        opt_mv  = self._minimize_projected_gradient(lambda w: self._portfolio_performance(w)[1], seed_mv)
        r_mv, v_mv, s_mv = self._portfolio_performance(opt_mv)
        return {
            "max_sharpe":     {"return": r_ms, "volatility": v_ms, "sharpe": s_ms, "weights": opt_ms},
            "min_volatility": {"return": r_mv, "volatility": v_mv, "sharpe": s_mv, "weights": opt_mv},
        }

    def correlation_matrix(self):
        return self.returns.corr()
