import numpy as np
import pandas as pd
import urllib.request
import json


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

    def _fetch_yahoo(self, ticker, start, end):
        t1 = int(pd.Timestamp(start).timestamp())
        t2 = int(pd.Timestamp(end).timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={t1}&period2={t2}&interval=1d&events=adjclose"
        )
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["adjclose"][0]["adjclose"]
        dates = pd.to_datetime(timestamps, unit="s").normalize()
        series = pd.Series(closes, index=dates, name=ticker, dtype=float)
        return series.dropna()

    def _fetch_data(self):
        frames = {}
        for ticker in self.tickers:
            frames[ticker] = self._fetch_yahoo(ticker, self.start_date, self.end_date)
        return pd.DataFrame(frames).dropna()

    def _portfolio_performance(self, weights):
        w = np.asarray(weights)
        ret = float(w @ self.expected_returns.values)
        vol = float(np.sqrt(w @ self.cov_matrix.values @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _minimize_slsqp(self, objective_fn, seed_weights):
        """Pure numpy SLSQP-style optimisation — no scipy needed."""
        from numpy.linalg import norm
        w = np.array(seed_weights, dtype=float)
        lr = 0.01
        for _ in range(5000):
            grad = np.zeros(self.num_assets)
            eps = 1e-6
            f0 = objective_fn(w)
            for i in range(self.num_assets):
                w[i] += eps
                grad[i] = (objective_fn(w) - f0) / eps
                w[i] -= eps
            w -= lr * grad
            # Project onto simplex (weights sum to 1, all >= 0)
            w = self._project_simplex(w)
        return w

    def _project_simplex(self, v):
        """Project vector v onto the probability simplex."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        return np.maximum(v - theta, 0)

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

        # Max Sharpe
        seed_ms = weights[np.nanargmax(sharpes)].copy()
        opt_ms  = self._minimize_slsqp(lambda w: -self._portfolio_performance(w)[2], seed_ms)
        r_ms, v_ms, s_ms = self._portfolio_performance(opt_ms)

        # Min Volatility
        seed_mv = weights[np.argmin(vols)].copy()
        opt_mv  = self._minimize_slsqp(lambda w: self._portfolio_performance(w)[1], seed_mv)
        r_mv, v_mv, s_mv = self._portfolio_performance(opt_mv)

        return {
            "max_sharpe":     {"return": r_ms, "volatility": v_ms, "sharpe": s_ms, "weights": opt_ms},
            "min_volatility": {"return": r_mv, "volatility": v_mv, "sharpe": s_mv, "weights": opt_mv},
        }

    def correlation_matrix(self):
        return self.returns.corr()
