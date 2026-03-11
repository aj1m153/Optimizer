import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from optimizer import PortfolioOptimizer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Matplotlib dark style ──────────────────────────────────────────────────────
BG      = "#0a0c10"
SURFACE = "#111318"
BORDER  = "#1f2530"
GOLD    = "#c9a84c"
TEAL    = "#3ecfb2"
RED     = "#e05c5c"
MUTED   = "#6b7280"
TEXT    = "#e8e4da"
PALETTE = [GOLD, TEAL, "#5b8dee", RED, "#a78bfa", "#f97316", "#84cc16", "#ec4899"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURFACE, "axes.edgecolor": BORDER,
    "axes.labelcolor": MUTED, "axes.titlecolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "grid.color": BORDER, "grid.linestyle": "--",
    "grid.alpha": 0.6, "text.color": TEXT, "font.family": "monospace",
    "legend.facecolor": SURFACE, "legend.edgecolor": BORDER,
    "legend.labelcolor": TEXT, "figure.dpi": 130,
})

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

# ── Run ────────────────────────────────────────────────────────────────────────
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
k3.metric("Max Sharpe",  f"{ms['sharpe']:.3f}",         f"ret {ms['return']*100:.1f}%")
k4.metric("Min Vol",     f"{mv['volatility']*100:.1f}%", f"ret {mv['return']*100:.1f}%")
k5.metric("Best Return", f"{returns.max()*100:.1f}%")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["  EFFICIENT FRONTIER  ", "  OPTIMAL WEIGHTS  ", "  CORRELATION  ", "  PRICE HISTORY  "])

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list("sharpe", ["#1a2a3a", "#2a4a5a", GOLD, "#f0d080"])
    sc = ax.scatter(volatility*100, returns*100, c=sharpe, cmap=cmap, s=4, alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
    cb.set_label("Sharpe Ratio", color=MUTED, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
    ax.scatter(ms["volatility"]*100, ms["return"]*100, marker="*", s=420, color=GOLD, zorder=5, edgecolors=BG, linewidths=0.8, label=f"Max Sharpe  ({ms['sharpe']:.3f})")
    ax.scatter(mv["volatility"]*100, mv["return"]*100, marker="D", s=130, color=TEAL, zorder=5, edgecolors=BG, linewidths=0.8, label=f"Min Volatility  ({mv['volatility']*100:.1f}%)")
    ax.set_xlabel("Volatility / Risk  (%)", fontsize=10)
    ax.set_ylabel("Expected Annual Return  (%)", fontsize=10)
    ax.set_title("Efficient Frontier", fontsize=13, color=TEXT, pad=12)
    ax.grid(True); ax.legend(fontsize=9); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    col_ms, col_mv = st.columns(2)
    for col, label, portfolio, color in [(col_ms, "Max Sharpe Ratio", ms, GOLD), (col_mv, "Min Volatility", mv, TEAL)]:
        with col:
            weights = portfolio["weights"]
            df_w = pd.DataFrame({"Ticker": tickers, "Weight": weights}).sort_values("Weight", ascending=False)
            st.markdown(f'<div class="section-label">{label}</div><span style="font-family:monospace;font-size:0.8rem;color:{color};">Sharpe {portfolio["sharpe"]:.3f}</span> <span style="font-family:monospace;font-size:0.75rem;color:{MUTED};">· ret {portfolio["return"]*100:.1f}% · vol {portfolio["volatility"]*100:.1f}%</span>', unsafe_allow_html=True)
            # Donut
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            wedges, texts, autotexts = ax.pie(df_w["Weight"], labels=df_w["Ticker"], colors=PALETTE[:len(tickers)], autopct="%1.1f%%", pctdistance=0.82, startangle=90, wedgeprops=dict(width=0.52, edgecolor=BG, linewidth=2))
            for t in texts:     t.set(color=TEXT, fontsize=9)
            for t in autotexts: t.set(color=BG, fontsize=8, fontweight="bold")
            ax.text(0, 0, label[:3].upper(), ha="center", va="center", color=color, fontsize=11, fontweight="bold")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            # Bar
            fig2, ax2 = plt.subplots(figsize=(4.5, 2.8))
            bars = ax2.barh(df_w["Ticker"], df_w["Weight"]*100, edgecolor=BG, linewidth=0.5)
            for bar, w in zip(bars, df_w["Weight"]):
                bar.set_facecolor(mcolors.to_rgba(color, alpha=0.35 + 0.65*(w/df_w["Weight"].max())))
                ax2.text(w*100+0.3, bar.get_y()+bar.get_height()/2, f"{w*100:.1f}%", va="center", fontsize=8, color=TEXT)
            ax2.set_xlabel("Allocation (%)", fontsize=9); ax2.set_xlim(0, df_w["Weight"].max()*100*1.2)
            ax2.grid(True, axis="x"); ax2.invert_yaxis(); fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cmap_corr = mcolors.LinearSegmentedColormap.from_list("corr", ["#1a3a5c", SURFACE, "#7a3f1e"])
    im = ax.imshow(corr.values, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(tickers))); ax.set_xticklabels(tickers, fontsize=10, color=TEXT)
    ax.set_yticks(range(len(tickers))); ax.set_yticklabels(tickers, fontsize=10, color=TEXT)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=10, color=TEXT, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("ρ", color=MUTED, fontsize=11); cb.ax.yaxis.set_tick_params(color=MUTED, labelcolor=MUTED)
    ax.set_title("Asset Correlation Matrix", fontsize=12, color=TEXT, pad=12)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown(f'<div style="font-family:monospace;font-size:0.78rem;color:{MUTED};margin-top:-0.5rem;">ρ → +1: assets move together (less diversification) · ρ → −1: assets move inversely (max diversification)</div>', unsafe_allow_html=True)

# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    prices_norm = optimizer.price_data / optimizer.price_data.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, col in enumerate(prices_norm.columns):
        ax.plot(prices_norm.index, prices_norm[col], color=PALETTE[i % len(PALETTE)], linewidth=1.6, label=col)
    ax.set_xlabel("Date", fontsize=10); ax.set_ylabel("Indexed Price  (base = 100)", fontsize=10)
    ax.set_title("Normalised Price History", fontsize=13, color=TEXT, pad=12)
    ax.grid(True); ax.legend(fontsize=9, ncol=min(len(tickers), 5)); fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="font-family:monospace;font-size:0.7rem;color:#3a3f48;text-align:center;padding:0.5rem 0 1rem;">SMART PORTFOLIO OPTIMIZER · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)
