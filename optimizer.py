import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Monte Carlo Portfolio Optimizer — Modern Portfolio Theory.

    Fixes vs original:
      - Returns & covariance are annualised (×252)
      - Vectorised simulation (no Python loop)
      - scipy SLSQP refinement for truly optimal portfolios
      - auto_adjust=True to handle yfinance ≥0.2 column changes
      - Single-ticker guard
    """

    def __init__(self, tickers: list, start_date: str, end_date: str, risk_free_rate: float = 0.01):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(tickers)

        self.price_data = self._fetch_data()
        self.returns = self._calculate_returns()

        # Annualised stats
        self.expected_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_data(self) -> pd.DataFrame:
        df = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )["Close"]

        # Guard: single ticker → Series to DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame(self.tickers[0])

        return df.dropna()

    def _calculate_returns(self) -> pd.DataFrame:
        return self.price_data.pct_change().dropna()

    # ------------------------------------------------------------------
    # Portfolio maths
    # ------------------------------------------------------------------

    def _portfolio_performance(self, weights: np.ndarray):
        w = np.asarray(weights)
        ret = float(w @ self.expected_returns.values)
        vol = float(np.sqrt(w @ self.cov_matrix.values @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    # ------------------------------------------------------------------
    # Monte Carlo simulation — fully vectorised
    # ------------------------------------------------------------------

    def simulate_portfolios(self, num_portfolios: int = 10_000, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)

        # Shape: (num_portfolios, num_assets)
        raw = rng.random((num_portfolios, self.num_assets))
        weights = raw / raw.sum(axis=1, keepdims=True)

        er = self.expected_returns.values          # (assets,)
        cov = self.cov_matrix.values               # (assets, assets)

        port_returns = weights @ er                                          # (N,)
        port_vols    = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights))  # (N,)
        port_sharpes = (port_returns - self.risk_free_rate) / np.where(port_vols > 0, port_vols, np.nan)

        return {
            "returns":    port_returns,
            "volatility": port_vols,
            "sharpe":     port_sharpes,
            "weights":    weights,
        }

    # ------------------------------------------------------------------
    # Scipy SLSQP refinement
    # ------------------------------------------------------------------

    def _optimize(self, objective_fn, seed_weights: np.ndarray) -> np.ndarray:
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        result = minimize(
            objective_fn,
            seed_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        return result.x if result.success else seed_weights

    def get_optimal_portfolios(self, simulation_results: dict) -> dict:
        sharpes = simulation_results["sharpe"]
        vols    = simulation_results["volatility"]
        weights = simulation_results["weights"]

        # Seed from best MC sample, then refine
        seed_ms = weights[np.nanargmax(sharpes)]
        opt_ms  = self._optimize(lambda w: -self._portfolio_performance(w)[2], seed_ms)
        r_ms, v_ms, s_ms = self._portfolio_performance(opt_ms)

        seed_mv = weights[np.argmin(vols)]
        opt_mv  = self._optimize(lambda w: self._portfolio_performance(w)[1], seed_mv)
        r_mv, v_mv, s_mv = self._portfolio_performance(opt_mv)

        return {
            "max_sharpe": {
                "return":     r_ms,
                "volatility": v_ms,
                "sharpe":     s_ms,
                "weights":    opt_ms,
            },
            "min_volatility": {
                "return":     r_mv,
                "volatility": v_mv,
                "sharpe":     s_mv,
                "weights":    opt_mv,
            },
        }

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def weights_dataframe(self, weights: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"Ticker": self.tickers, "Weight": weights}).set_index("Ticker")

    def correlation_matrix(self) -> pd.DataFrame:
        return self.returns.corr()
