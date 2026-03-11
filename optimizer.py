import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


class PortfolioOptimizer:
    def __init__(self, tickers, start_date, end_date, risk_free_rate=0.01):
        self.tickers = list(tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(self.tickers)
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
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        else:
            if "Close" in df.columns:
                df = df[["Close"]]
            if len(self.tickers) == 1:
                df = df.rename(columns={"Close": self.tickers[0]})
        if isinstance(df, pd.Series):
            df = df.to_frame(self.tickers[0])

        # Drop fully-NaN columns (bad tickers)
        df = df.dropna(axis=1, how="all")
        failed = [t for t in self.tickers if t not in df.columns]
        if failed:
            st.warning(f"No data found for: {', '.join(failed)} — removed.")
            self.tickers = [t for t in self.tickers if t in df.columns]
            self.num_assets = len(self.tickers)

        if df.shape[1] < 2:
            raise ValueError("Need at least 2 tickers with valid data.")

        return df.dropna()

    # ── Core maths ─────────────────────────────────────────────────────────────
    def _perf(self, w):
        w = np.asarray(w)
        ret = float(w @ self.expected_returns.values)
        vol = float(np.sqrt(w @ self.cov_matrix.values @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    # ── Monte Carlo — fully vectorised ─────────────────────────────────────────
    def simulate_portfolios(self, num_portfolios=5000, seed=42):
        rng = np.random.default_rng(seed)
        raw = rng.random((num_portfolios, self.num_assets))
        w = raw / raw.sum(axis=1, keepdims=True)
        er  = self.expected_returns.values
        cov = self.cov_matrix.values
        rets = w @ er
        vols = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
        sharpes = (rets - self.risk_free_rate) / np.where(vols > 0, vols, np.nan)
        return {"returns": rets, "volatility": vols, "sharpe": sharpes, "weights": w}

    # ── Analytical optimisation — no loops at all ──────────────────────────────
    def _max_sharpe_weights(self):
        """Closed-form max-Sharpe via Markowitz: w* ∝ Σ⁻¹(μ - rf)."""
        er  = self.expected_returns.values
        cov = self.cov_matrix.values
        excess = er - self.risk_free_rate
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        raw = cov_inv @ excess
        raw = np.maximum(raw, 0)          # long-only constraint
        s = raw.sum()
        return raw / s if s > 0 else np.ones(self.num_assets) / self.num_assets

    def _min_vol_weights(self):
        """Closed-form min-variance: w* ∝ Σ⁻¹ 1."""
        cov = self.cov_matrix.values
        ones = np.ones(self.num_assets)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        raw = cov_inv @ ones
        raw = np.maximum(raw, 0)
        s = raw.sum()
        return raw / s if s > 0 else np.ones(self.num_assets) / self.num_assets

    def get_optimal_portfolios(self, simulation_results=None):
        opt_ms = self._max_sharpe_weights()
        r_ms, v_ms, s_ms = self._perf(opt_ms)
        opt_mv = self._min_vol_weights()
        r_mv, v_mv, s_mv = self._perf(opt_mv)
        return {
            "max_sharpe":     {"return": r_ms, "volatility": v_ms, "sharpe": s_ms, "weights": opt_ms},
            "min_volatility": {"return": r_mv, "volatility": v_mv, "sharpe": s_mv, "weights": opt_mv},
        }

    def correlation_matrix(self):
        return self.returns.corr()
