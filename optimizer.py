import numpy as np
import pandas as pd
import yfinance as yf


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

    def _fetch_data(self):
        df = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # Handle single vs multi ticker column structure
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        else:
            df = df[["Close"]] if "Close" in df.columns else df
            if len(self.tickers) == 1:
                df = df.rename(columns={"Close": self.tickers[0]})
        if isinstance(df, pd.Series):
            df = df.to_frame(self.tickers[0])
        return df.dropna()

    def _portfolio_performance(self, weights):
        w = np.asarray(weights)
        ret = float(w @ self.expected_returns.values)
        vol = float(np.sqrt(w @ self.cov_matrix.values @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _project_simplex(self, v):
        """Project onto probability simplex: weights >= 0 and sum to 1."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1.0) / (rho + 1.0)
        return np.maximum(v - theta, 0)

    def _optimize(self, objective_fn, seed_weights):
        """Projected gradient descent — no scipy needed."""
        w = self._project_simplex(np.array(seed_weights, dtype=float))
        best_w, best_val = w.copy(), objective_fn(w)
        eps, lr = 1e-6, 0.005
        for _ in range(8000):
            f0 = objective_fn(w)
            grad = np.array([(objective_fn(w + eps * (np.arange(self.num_assets) == i)) - f0) / eps
                             for i in range(self.num_assets)])
            w = self._project_simplex(w - lr * grad)
            val = objective_fn(w)
            if val < best_val:
                best_val, best_w = val, w.copy()
        return best_w

    def simulate_portfolios(self, num_portfolios=10000, seed=42):
        rng = np.random.default_rng(seed)
        raw = rng.random((num_portfolios, self.num_assets))
        weights = raw / raw.sum(axis=1, keepdims=True)
        er, cov = self.expected_returns.values, self.cov_matrix.values
        port_returns = weights @ er
        port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))
        port_sharpes = (port_returns - self.risk_free_rate) / np.where(port_vols > 0, port_vols, np.nan)
        return {"returns": port_returns, "volatility": port_vols, "sharpe": port_sharpes, "weights": weights}

    def get_optimal_portfolios(self, simulation_results):
        sharpes = simulation_results["sharpe"]
        vols    = simulation_results["volatility"]
        weights = simulation_results["weights"]
        seed_ms = weights[np.nanargmax(sharpes)].copy()
        opt_ms  = self._optimize(lambda w: -self._portfolio_performance(w)[2], seed_ms)
        r_ms, v_ms, s_ms = self._portfolio_performance(opt_ms)
        seed_mv = weights[np.argmin(vols)].copy()
        opt_mv  = self._optimize(lambda w: self._portfolio_performance(w)[1], seed_mv)
        r_mv, v_mv, s_mv = self._portfolio_performance(opt_mv)
        return {
            "max_sharpe":     {"return": r_ms, "volatility": v_ms, "sharpe": s_ms, "weights": opt_ms},
            "min_volatility": {"return": r_mv, "volatility": v_mv, "sharpe": s_mv, "weights": opt_mv},
        }

    def correlation_matrix(self):
        return self.returns.corr()
