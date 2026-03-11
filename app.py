import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from optimizer import PortfolioOptimizer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --surface2:  #181c24;
    --border:    #1f2530;
    --gold:      #c9a84c;
    --gold-dim:  #7a6230;
    --teal:      #3ecfb2;
    --red:       #e05c5c;
    --text:      #e8e4da;
    --muted:     #6b7280;
    --radius:    12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Headers */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.4rem !important;
}
[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.8rem !important;
    color: var(--gold) !important;
}

/* Buttons */
.stButton > button {
    background: var(--gold) !important;
    color: #0a0c10 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em;
    font-weight: 500 !important;
    padding: 0.6rem 1.6rem !important;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSlider > div,
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1.4rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Section card */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Ticker tag */
.ticker-tag {
    display: inline-block;
    background: var(--surface2);
    border: 1px solid var(--gold-dim);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--gold);
    margin: 2px;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    line-height: 1.1;
    color: var(--text);
    margin: 0;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.4rem;
}
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--gold-dim);
    margin-bottom: 0.5rem;
}
.badge-sharpe {
    background: rgba(201,168,76,0.12);
    border: 1px solid var(--gold-dim);
    color: var(--gold);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
}
.badge-vol {
    background: rgba(62,207,178,0.1);
    border: 1px solid rgba(62,207,178,0.35);
    color: var(--teal);
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0c10",
    plot_bgcolor="#111318",
    font=dict(family="DM Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=50, r=30, t=40, b=50),
    xaxis=dict(gridcolor="#1f2530", zerolinecolor="#1f2530", showgrid=True),
    yaxis=dict(gridcolor="#1f2530", zerolinecolor="#1f2530", showgrid=True),
)

GOLD  = "#c9a84c"
TEAL  = "#3ecfb2"
RED   = "#e05c5c"
MUTED = "#6b7280"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">◈ Portfolio Config</p>', unsafe_allow_html=True)
    st.markdown("### Tickers")

    ticker_input = st.text_input(
        "Enter comma-separated tickers",
        value="AAPL, MSFT, GOOGL, AMZN, META",
        label_visibility="collapsed",
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    # Show ticker badges
    badges = " ".join(f'<span class="ticker-tag">{t}</span>' for t in tickers)
    st.markdown(badges, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.Timestamp("2020-01-01"), label_visibility="visible")
    with col2:
        end_date   = st.date_input("To",   value=pd.Timestamp("2025-01-01"), label_visibility="visible")

    st.markdown("---")
    st.markdown("### Parameters")
    risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.10, 0.04, 0.005, format="%.3f")
    num_portfolios = st.select_slider(
        "Simulations",
        options=[1_000, 2_500, 5_000, 10_000, 20_000],
        value=10_000,
        format_func=lambda x: f"{x:,}",
    )

    st.markdown("---")
    run = st.button("⟳  Run Optimizer")

# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1.4rem 0; border-bottom: 1px solid #1f2530; margin-bottom: 2rem;">
  <p class="hero-title">Smart Portfolio<br><em>Optimizer</em></p>
  <p class="hero-sub">Modern Portfolio Theory · Efficient Frontier · Monte Carlo Simulation</p>
</div>
""", unsafe_allow_html=True)

# ── Main logic ─────────────────────────────────────────────────────────────────
if not run:
    # Landing state
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "◎", "Efficient Frontier", "10,000+ simulated portfolios mapped across the risk-return spectrum"),
        (c2, "◈", "Optimal Portfolios", "Max Sharpe & Min Volatility portfolios found via scipy SLSQP"),
        (c3, "◉", "Correlation Matrix", "Understand diversification and asset co-movement at a glance"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; min-height:140px;">
              <div style="font-size:2rem; margin-bottom:0.6rem; color:#c9a84c;">{icon}</div>
              <div style="font-family:'DM Serif Display',serif; font-size:1.05rem; margin-bottom:0.4rem;">{title}</div>
              <div style="font-size:0.82rem; color:#6b7280;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top: 3rem; color: #6b7280; font-family: 'DM Mono', monospace; font-size: 0.8rem; letter-spacing: 0.12em;">
        CONFIGURE YOUR PORTFOLIO IN THE SIDEBAR AND PRESS RUN
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run optimisation ───────────────────────────────────────────────────────────
with st.spinner("Fetching data & running simulation…"):
    try:
        optimizer = PortfolioOptimizer(
            tickers=tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            risk_free_rate=risk_free_rate,
        )
        results  = optimizer.simulate_portfolios(num_portfolios)
        optimal  = optimizer.get_optimal_portfolios(results)
        corr     = optimizer.correlation_matrix()
    except Exception as e:
        st.error(f"**Error:** {e}")
        st.stop()

returns    = results["returns"]
volatility = results["volatility"]
sharpe     = results["sharpe"]

ms = optimal["max_sharpe"]
mv = optimal["min_volatility"]

# ── KPI strip ──────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Assets",          len(tickers))
k2.metric("Simulations",     f"{num_portfolios:,}")
k3.metric("Max Sharpe",      f"{ms['sharpe']:.3f}",      f"ret {ms['return']*100:.1f}%")
k4.metric("Min Vol",         f"{mv['volatility']*100:.1f}%", f"ret {mv['return']*100:.1f}%")
k5.metric("Best Return",     f"{returns.max()*100:.1f}%")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  EFFICIENT FRONTIER  ",
    "  OPTIMAL WEIGHTS  ",
    "  CORRELATION  ",
    "  PRICE HISTORY  ",
])

# ── Tab 1: Efficient Frontier ──────────────────────────────────────────────────
with tab1:
    fig = go.Figure()

    # Scatter — all portfolios
    fig.add_trace(go.Scattergl(
        x=volatility * 100,
        y=returns * 100,
        mode="markers",
        marker=dict(
            color=sharpe,
            colorscale=[
                [0.0,  "#1a2a3a"],
                [0.35, "#2a4a5a"],
                [0.65, "#c9a84c"],
                [1.0,  "#f0d080"],
            ],
            size=3,
            opacity=0.7,
            colorbar=dict(
                title=dict(text="Sharpe Ratio", font=dict(family="DM Mono", size=11, color=MUTED)),
                tickfont=dict(family="DM Mono", size=10, color=MUTED),
                thickness=12,
                len=0.7,
            ),
            line=dict(width=0),
        ),
        name="Portfolios",
        hovertemplate="<b>Volatility:</b> %{x:.2f}%<br><b>Return:</b> %{y:.2f}%<extra></extra>",
    ))

    # Max Sharpe star
    fig.add_trace(go.Scatter(
        x=[ms["volatility"] * 100],
        y=[ms["return"] * 100],
        mode="markers",
        marker=dict(symbol="star", size=22, color=GOLD, line=dict(color="#0a0c10", width=1.5)),
        name=f"Max Sharpe  ({ms['sharpe']:.3f})",
        hovertemplate=(
            f"<b>★ Max Sharpe</b><br>"
            f"Return: {ms['return']*100:.2f}%<br>"
            f"Volatility: {ms['volatility']*100:.2f}%<br>"
            f"Sharpe: {ms['sharpe']:.3f}<extra></extra>"
        ),
    ))

    # Min Vol diamond
    fig.add_trace(go.Scatter(
        x=[mv["volatility"] * 100],
        y=[mv["return"] * 100],
        mode="markers",
        marker=dict(symbol="diamond", size=16, color=TEAL, line=dict(color="#0a0c10", width=1.5)),
        name=f"Min Volatility  ({mv['volatility']*100:.1f}%)",
        hovertemplate=(
            f"<b>◆ Min Volatility</b><br>"
            f"Return: {mv['return']*100:.2f}%<br>"
            f"Volatility: {mv['volatility']*100:.2f}%<br>"
            f"Sharpe: {mv['sharpe']:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        **PLOT_LAYOUT,
        height=520,
        xaxis_title="Volatility / Risk  (%)",
        yaxis_title="Expected Annual Return  (%)",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(17,19,24,0.9)",
            bordercolor="#1f2530",
            borderwidth=1,
            font=dict(family="DM Mono", size=11, color="#e8e4da"),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Optimal Weights ─────────────────────────────────────────────────────
with tab2:
    col_ms, col_mv = st.columns(2)

    for col, label, badge_cls, portfolio, color in [
        (col_ms, "Max Sharpe Ratio", "badge-sharpe", ms, GOLD),
        (col_mv, "Min Volatility",   "badge-vol",    mv, TEAL),
    ]:
        with col:
            st.markdown(f"""
            <div class="section-label">{label}</div>
            <span class="{badge_cls}">Sharpe {portfolio['sharpe']:.3f}</span>&nbsp;
            <span style="font-family:'DM Mono',monospace;font-size:0.75rem;color:{MUTED};">
              ret {portfolio['return']*100:.1f}% · vol {portfolio['volatility']*100:.1f}%
            </span>
            """, unsafe_allow_html=True)

            weights = portfolio["weights"]
            df_w = pd.DataFrame({
                "Ticker": tickers,
                "Weight": weights,
            }).sort_values("Weight", ascending=False)

            # Donut
            fig_d = go.Figure(go.Pie(
                labels=df_w["Ticker"],
                values=df_w["Weight"],
                hole=0.62,
                textinfo="label+percent",
                textfont=dict(family="DM Mono", size=12),
                marker=dict(
                    colors=[
                        "#c9a84c", "#3ecfb2", "#5b8dee", "#e05c5c",
                        "#a78bfa", "#f97316", "#84cc16", "#ec4899",
                    ][:len(tickers)],
                    line=dict(color="#0a0c10", width=2),
                ),
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            ))
            fig_d.update_layout(
                **PLOT_LAYOUT,
                height=280,
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                annotations=[dict(
                    text=f"<b>{label[:3].upper()}</b>",
                    x=0.5, y=0.5, font=dict(family="DM Serif Display", size=14, color=color),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_d, use_container_width=True)

            # Horizontal bar
            fig_b = go.Figure(go.Bar(
                x=df_w["Weight"] * 100,
                y=df_w["Ticker"],
                orientation="h",
                marker=dict(
                    color=df_w["Weight"] * 100,
                    colorscale=[[0, "#1f2530"], [1, color]],
                    line=dict(width=0),
                ),
                text=[f"{w*100:.1f}%" for w in df_w["Weight"]],
                textposition="outside",
                textfont=dict(family="DM Mono", size=11, color="#e8e4da"),
                hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
            ))
            fig_b.update_layout(
                **PLOT_LAYOUT,
                height=200,
                margin=dict(l=10, r=60, t=10, b=10),
                yaxis=dict(gridcolor="transparent", zeroline=False),
                xaxis=dict(title="Allocation (%)", gridcolor="#1f2530"),
                showlegend=False,
            )
            st.plotly_chart(fig_b, use_container_width=True)

# ── Tab 3: Correlation Matrix ──────────────────────────────────────────────────
with tab3:
    fig_c = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0,  "#1a3a5c"],
            [0.5,  "#111318"],
            [1.0,  "#7a3f1e"],
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(family="DM Mono", size=13, color="#e8e4da"),
        colorbar=dict(
            title=dict(text="ρ", font=dict(family="DM Serif Display", size=14, color=MUTED)),
            tickfont=dict(family="DM Mono", size=10, color=MUTED),
            thickness=12,
        ),
        hovertemplate="<b>%{x} vs %{y}</b><br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig_c.update_layout(
        **PLOT_LAYOUT,
        height=460,
        margin=dict(l=60, r=60, t=30, b=60),
        xaxis=dict(showgrid=False, tickfont=dict(family="DM Mono", size=12)),
        yaxis=dict(showgrid=False, tickfont=dict(family="DM Mono", size=12)),
    )
    st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.78rem; color:#6b7280; margin-top:-0.5rem;">
    ρ close to +1 → assets move together (less diversification) ·
    ρ close to −1 → assets move inversely (maximum diversification benefit)
    </div>
    """, unsafe_allow_html=True)

# ── Tab 4: Price History ───────────────────────────────────────────────────────
with tab4:
    prices_norm = optimizer.price_data / optimizer.price_data.iloc[0] * 100

    palette = [
        "#c9a84c", "#3ecfb2", "#5b8dee", "#e05c5c",
        "#a78bfa", "#f97316", "#84cc16", "#ec4899",
    ]

    fig_p = go.Figure()
    for i, col in enumerate(prices_norm.columns):
        fig_p.add_trace(go.Scatter(
            x=prices_norm.index,
            y=prices_norm[col],
            name=col,
            line=dict(color=palette[i % len(palette)], width=1.8),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %Y}}<br>Indexed: %{{y:.1f}}<extra></extra>",
        ))

    fig_p.update_layout(
        **PLOT_LAYOUT,
        height=460,
        xaxis_title="Date",
        yaxis_title="Indexed Price  (base = 100)",
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(17,19,24,0.9)",
            bordercolor="#1f2530",
            borderwidth=1,
            font=dict(family="DM Mono", size=11, color="#e8e4da"),
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_p, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#3a3f48; text-align:center; padding: 0.5rem 0 1rem;">
SMART PORTFOLIO OPTIMIZER · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
