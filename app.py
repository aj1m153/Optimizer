import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from optimizer import PortfolioOptimizer

st.set_page_config(page_title="Smart Portfolio Optimizer", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

BG      = "#0a0c10"
SURFACE = "#111318"
BORDER  = "#1f2530"
GOLD    = "#c9a84c"
TEAL    = "#3ecfb2"
RED     = "#e05c5c"
MUTED   = "#6b7280"
TEXT    = "#e8e4da"
PALETTE = [GOLD, TEAL, "#5b8dee", RED, "#a78bfa", "#f97316", "#84cc16", "#ec4899"]

# ── Stock universe ─────────────────────────────────────────────────────────────
STOCK_UNIVERSE = {
    "Technology": {
        "Large Cap":  ["AAPL","MSFT","GOOGL","NVDA","META","AMZN","TSLA","AVGO","ORCL","ADBE"],
        "Mid Cap":    ["SNOW","DDOG","CRWD","ZS","NET","GTLB","MDB","HUBS","BILL","TTD"],
        "Small Cap":  ["IONQ","BBAI","SOUN","KOPN","AIOT","PDFS","SMTC","MFAC","LPSN","KPLT"],
    },
    "Healthcare": {
        "Large Cap":  ["JNJ","UNH","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","BMY"],
        "Mid Cap":    ["ALGN","HOLX","ICLR","INSP","NVCR","OTRK","PINC","PRVA","RCM","RNST"],
        "Small Cap":  ["ACCD","ADPT","ARDX","ATRC","AXNX","CDNA","CGEN","CLPT","CNMD","CODX"],
    },
    "Finance": {
        "Large Cap":  ["JPM","BAC","WFC","GS","MS","BLK","AXP","SPGI","CB","MMC"],
        "Mid Cap":    ["ALLY","CBSH","CFG","CMA","FHN","GBCI","IBOC","NBTB","PNFP","SFNC"],
        "Small Cap":  ["AROW","BSVN","BYFC","CARV","CFFI","CLBK","COFS","CZWI","DCOM","EBTC"],
    },
    "Energy": {
        "Large Cap":  ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES"],
        "Mid Cap":    ["CIVI","CRGY","GPRE","HESM","MNRL","MTDR","RRC","SM","VTLE","WTI"],
        "Small Cap":  ["BORR","CEIX","CONN","DNOW","ESTE","FLNG","FANG","GTE","HBT","IMPP"],
    },
    "Consumer": {
        "Large Cap":  ["WMT","HD","MCD","NKE","SBUX","TGT","COST","LOW","TJX","CMG"],
        "Mid Cap":    ["BYND","CAKE","CHUY","DENN","EAT","FAT","FRSH","JACK","KRUS","LOCO"],
        "Small Cap":  ["ANDE","ARCH","BIRD","BOOT","CATO","CHRS","CLAR","CULP","DXLG","EXPR"],
    },
    "Real Estate": {
        "Large Cap":  ["AMT","PLD","CCI","EQIX","PSA","SPG","O","WELL","DLR","AVB"],
        "Mid Cap":    ["ALEX","BRT","CHCT","CLDT","COOR","DEA","GMRE","HIW","INN","JBGS"],
        "Small Cap":  ["AOMR","AHH","BHSE","BRT","CLPR","CORR","CTRL","DBRG","ELME","EPRT"],
    },
    "ETFs & Index": {
        "Broad Market": ["SPY","QQQ","IWM","DIA","VTI","VOO","IVV","SCHB","ITOT","VT"],
        "Sector":       ["XLK","XLF","XLE","XLV","XLC","XLI","XLY","XLP","XLU","XLRE"],
        "International":["EFA","EEM","VEA","VWO","IEFA","ACWI","IXUS","VXUS","SCZ","GXC"],
    },
    "Crypto-Related": {
        "Large Cap":  ["COIN","MSTR","MARA","RIOT","HUT","CLSK","BTBT","CIFR","WGMI","BITO"],
        "Mid Cap":    ["IREN","CORZ","SATO","BTCS","BKKT","GREE","DMGI","BTDR","WULF","NXRA"],
        "Small Cap":  ["BTCM","BTOG","CBIT","DIGI","HIVE","HPNN","MGTI","NKLA","PNTM","SLNH"],
    },
}

RISK_PROFILES = {
    "🟢 Conservative": {
        "desc": "Capital preservation. Low volatility, stable dividend payers.",
        "num_portfolios": 5000,
        "rf_rate": 0.045,
        "preferred_sectors": ["Real Estate", "Finance", "ETFs & Index"],
        "preferred_caps": ["Large Cap", "Broad Market", "Sector"],
        "max_weight": 0.40,
    },
    "🟡 Moderate": {
        "desc": "Balanced growth and income. Mix of stability and upside.",
        "num_portfolios": 10000,
        "rf_rate": 0.04,
        "preferred_sectors": ["Technology", "Healthcare", "Finance", "Consumer"],
        "preferred_caps": ["Large Cap", "Mid Cap"],
        "max_weight": 0.35,
    },
    "🔴 Aggressive": {
        "desc": "Maximum growth. High volatility, high potential returns.",
        "num_portfolios": 15000,
        "rf_rate": 0.02,
        "preferred_sectors": ["Technology", "Crypto-Related", "Energy"],
        "preferred_caps": ["Mid Cap", "Small Cap"],
        "max_weight": 0.50,
    },
}

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
.risk-card { border-radius:10px; padding:0.6rem 1rem; margin-bottom:0.5rem; font-family:'DM Mono',monospace; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">◈ Portfolio Config</p>', unsafe_allow_html=True)

    # ── Risk Profile ───────────────────────────────────────────────────────────
    st.markdown("### Risk Profile")
    risk_choice = st.radio("Risk Level", list(RISK_PROFILES.keys()), index=1, label_visibility="collapsed")
    risk = RISK_PROFILES[risk_choice]
    st.markdown(f'<div class="risk-card" style="background:{SURFACE};border:1px solid {BORDER};color:{MUTED};">{risk["desc"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Stock selection mode ───────────────────────────────────────────────────
    st.markdown("### Stock Selection")
    selection_mode = st.radio("Mode", ["Browse by Industry", "Enter Manually"], label_visibility="collapsed")

    tickers = []

    if selection_mode == "Browse by Industry":
        st.markdown("**Industry**")
        industries = st.multiselect(
            "Industries", list(STOCK_UNIVERSE.keys()),
            default=risk["preferred_sectors"][:2],
            label_visibility="collapsed",
        )

        if industries:
            st.markdown("**Cap Size**")
            all_caps = sorted(set(cap for ind in industries for cap in STOCK_UNIVERSE[ind].keys()))
            caps = st.multiselect(
                "Cap sizes", all_caps,
                default=[c for c in risk["preferred_caps"] if c in all_caps][:2],
                label_visibility="collapsed",
            )

            # Build pool and let user pick
            pool = []
            for ind in industries:
                for cap in caps:
                    if cap in STOCK_UNIVERSE[ind]:
                        pool.extend(STOCK_UNIVERSE[ind][cap])
            pool = sorted(set(pool))

            if pool:
                st.markdown("**Select Stocks**")
                default_picks = pool[:6]
                tickers = st.multiselect("Stocks", pool, default=default_picks, label_visibility="collapsed")
    else:
        ticker_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN, META", label_visibility="collapsed")
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if tickers:
        badges = " ".join(f'<span class="ticker-tag">{t}</span>' for t in tickers)
        st.markdown(badges, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Date Range")
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("From", value=pd.Timestamp("2020-01-01"))
    with c2: end_date   = st.date_input("To",   value=pd.Timestamp("2025-01-01"))

    st.markdown("---")
    st.markdown("### Advanced Parameters")
    with st.expander("Customise", expanded=False):
        risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.10, risk["rf_rate"], 0.005, format="%.3f")
        num_portfolios = st.select_slider("Simulations", options=[1_000, 2_500, 5_000, 10_000, 20_000],
                                          value=risk["num_portfolios"], format_func=lambda x: f"{x:,}")
    if "risk_free_rate" not in dir(): risk_free_rate = risk["rf_rate"]
    if "num_portfolios" not in dir(): num_portfolios = risk["num_portfolios"]

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
    # Risk profile cards
    st.markdown('<p class="section-label">Select a risk profile to get started</p>', unsafe_allow_html=True)
    rc1, rc2, rc3 = st.columns(3)
    for col, key, color in [(rc1, "🟢 Conservative", "#3ecfb2"), (rc2, "🟡 Moderate", "#c9a84c"), (rc3, "🔴 Aggressive", "#e05c5c")]:
        p = RISK_PROFILES[key]
        with col:
            selected = "border:2px solid " + color if key == risk_choice else f"border:1px solid {BORDER}"
            st.markdown(f'<div class="card" style="{selected};text-align:center;min-height:130px;"><div style="font-size:1.8rem;margin-bottom:0.4rem;">{key.split()[0]}</div><div style="font-family:\'DM Serif Display\',serif;font-size:1rem;margin-bottom:0.4rem;color:{color};">{key.split(" ",1)[1]}</div><div style="font-size:0.78rem;color:{MUTED};">{p["desc"]}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "◎", "Efficient Frontier", "Thousands of portfolios mapped across the risk-return spectrum"),
        (c2, "◈", "Optimal Portfolios", "Max Sharpe & Min Volatility found via numpy optimisation"),
        (c3, "◉", "Correlation Matrix", "Understand diversification and asset co-movement"),
    ]:
        with col:
            st.markdown(f'<div class="card" style="text-align:center;min-height:120px;"><div style="font-size:1.8rem;margin-bottom:0.5rem;color:{GOLD};">{icon}</div><div style="font-family:\'DM Serif Display\',serif;font-size:1rem;margin-bottom:0.3rem;">{title}</div><div style="font-size:0.8rem;color:{MUTED};">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;margin-top:2rem;color:{MUTED};font-family:monospace;font-size:0.8rem;letter-spacing:0.12em;">SELECT STOCKS & RISK PROFILE IN THE SIDEBAR → PRESS RUN</div>', unsafe_allow_html=True)
    st.stop()

if len(tickers) < 2:
    st.error("Please select at least 2 stocks to build a portfolio.")
    st.stop()

# ── Run ────────────────────────────────────────────────────────────────────────
with st.spinner("Fetching data & running simulation…"):
    try:
        optimizer = PortfolioOptimizer(tickers, str(start_date), str(end_date), risk_free_rate)
        results   = optimizer.simulate_portfolios(num_portfolios)
        optimal   = optimizer.get_optimal_portfolios(results)
        corr      = optimizer.correlation_matrix()
    except Exception as e:
        st.error(f"**Error fetching data:** {e}")
        st.stop()

returns    = results["returns"]
volatility = results["volatility"]
sharpe     = results["sharpe"]
ms = optimal["max_sharpe"]
mv = optimal["min_volatility"]

# ── Risk profile banner ────────────────────────────────────────────────────────
risk_color = {"🟢 Conservative": TEAL, "🟡 Moderate": GOLD, "🔴 Aggressive": RED}[risk_choice]
st.markdown(f'<div style="background:{SURFACE};border:1px solid {risk_color};border-radius:10px;padding:0.7rem 1.2rem;margin-bottom:1.2rem;font-family:monospace;font-size:0.82rem;"><span style="color:{risk_color};font-weight:bold;">{risk_choice}</span> &nbsp;·&nbsp; <span style="color:{MUTED};">{risk["desc"]}</span> &nbsp;·&nbsp; <span style="color:{MUTED};">RF Rate: {risk_free_rate*100:.1f}% · {num_portfolios:,} simulations</span></div>', unsafe_allow_html=True)

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
        "Sharpe Ratio":   np.nan_to_num(sharpe, nan=0.0),
    })
    frontier = alt.Chart(df_sim).mark_circle(size=18, opacity=0.65).encode(
        x=alt.X("Volatility (%):Q", title="Volatility / Risk (%)"),
        y=alt.Y("Return (%):Q",     title="Expected Annual Return (%)"),
        color=alt.Color("Sharpe Ratio:Q", scale=alt.Scale(scheme="goldorange"), legend=alt.Legend(title="Sharpe Ratio")),
        tooltip=["Volatility (%):Q", "Return (%):Q", "Sharpe Ratio:Q"],
    )
    df_opt = pd.DataFrame([
        {"Volatility (%)": ms["volatility"]*100, "Return (%)": ms["return"]*100, "Portfolio": f"★ Max Sharpe ({ms['sharpe']:.3f})", "size": 300},
        {"Volatility (%)": mv["volatility"]*100, "Return (%)": mv["return"]*100, "Portfolio": f"◆ Min Volatility ({mv['volatility']*100:.1f}%)", "size": 200},
    ])
    opt_points = alt.Chart(df_opt).mark_point(filled=True, opacity=1).encode(
        x="Volatility (%):Q", y="Return (%):Q",
        color=alt.Color("Portfolio:N", scale=alt.Scale(range=[GOLD, TEAL]), legend=alt.Legend(title="Optimal")),
        size=alt.Size("size:Q", legend=None),
        tooltip=["Portfolio:N", "Volatility (%):Q", "Return (%):Q"],
    )
    chart = (frontier + opt_points).properties(height=480, background=SURFACE, padding={"left":20,"right":20,"top":20,"bottom":20}
    ).configure_axis(gridColor=BORDER, labelColor=MUTED, titleColor=MUTED, domainColor=BORDER, tickColor=BORDER
    ).configure_legend(labelColor=TEXT, titleColor=MUTED, fillColor=SURFACE, strokeColor=BORDER
    ).configure_view(strokeOpacity=0)
    st.altair_chart(chart, use_container_width=True)

# ── Tab 2: Optimal Weights ─────────────────────────────────────────────────────
with tab2:
    col_ms, col_mv = st.columns(2)
    for col, label, portfolio, color in [(col_ms, "Max Sharpe Ratio", ms, GOLD), (col_mv, "Min Volatility", mv, TEAL)]:
        with col:
            st.markdown(f'<div class="section-label">{label}</div><span style="font-family:monospace;font-size:0.8rem;color:{color};">Sharpe {portfolio["sharpe"]:.3f}</span> <span style="font-family:monospace;font-size:0.75rem;color:{MUTED};">· ret {portfolio["return"]*100:.1f}% · vol {portfolio["volatility"]*100:.1f}%</span>', unsafe_allow_html=True)
            df_w = pd.DataFrame({"Ticker": tickers, "Allocation (%)": portfolio["weights"]*100}).sort_values("Allocation (%)", ascending=False)
            bar = alt.Chart(df_w).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                x=alt.X("Allocation (%):Q", title="Allocation (%)"),
                y=alt.Y("Ticker:N", sort="-x", title=None),
                color=alt.value(color),
                tooltip=["Ticker:N", alt.Tooltip("Allocation (%):Q", format=".1f")],
            ).properties(height=max(180, len(tickers)*40), background=SURFACE, padding={"left":10,"right":30,"top":10,"bottom":10}
            ).configure_axis(gridColor=BORDER, labelColor=TEXT, titleColor=MUTED, domainColor=BORDER, tickColor=BORDER
            ).configure_view(strokeOpacity=0)
            st.altair_chart(bar, use_container_width=True)
            df_w["Allocation (%)"] = df_w["Allocation (%)"].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_w.set_index("Ticker"), use_container_width=True)

# ── Tab 3: Correlation ─────────────────────────────────────────────────────────
with tab3:
    rows = [{"Asset A": tickers[i], "Asset B": tickers[j], "ρ": round(float(corr.values[i,j]),3)}
            for i in range(len(tickers)) for j in range(len(tickers))]
    df_corr = pd.DataFrame(rows)
    heatmap = alt.Chart(df_corr).mark_rect().encode(
        x=alt.X("Asset A:N", title=None), y=alt.Y("Asset B:N", title=None),
        color=alt.Color("ρ:Q", scale=alt.Scale(domain=[-1,0,1], range=["#1a3a5c", SURFACE, "#7a3f1e"]), legend=alt.Legend(title="ρ")),
        tooltip=["Asset A:N","Asset B:N","ρ:Q"],
    )
    text = alt.Chart(df_corr).mark_text(fontSize=12, fontWeight="bold").encode(
        x="Asset A:N", y="Asset B:N",
        text=alt.Text("ρ:Q", format=".2f"), color=alt.value(TEXT),
    )
    corr_chart = (heatmap+text).properties(height=420, background=SURFACE, padding={"left":20,"right":20,"top":20,"bottom":20}
    ).configure_axis(labelColor=TEXT, titleColor=MUTED, domainColor=BORDER, tickColor=BORDER, gridColor=BORDER
    ).configure_legend(labelColor=TEXT, titleColor=MUTED, fillColor=SURFACE, strokeColor=BORDER
    ).configure_view(strokeOpacity=0)
    st.altair_chart(corr_chart, use_container_width=True)
    st.markdown(f'<div style="font-family:monospace;font-size:0.78rem;color:{MUTED};">ρ → +1: assets move together · ρ → −1: assets move inversely (max diversification benefit)</div>', unsafe_allow_html=True)

# ── Tab 4: Price History ───────────────────────────────────────────────────────
with tab4:
    prices_norm = optimizer.price_data / optimizer.price_data.iloc[0] * 100
    prices_norm.index = pd.to_datetime(prices_norm.index)
    df_prices = prices_norm.reset_index()
    df_prices = df_prices.rename(columns={df_prices.columns[0]: "Date"})
    df_prices = df_prices.melt(id_vars="Date", var_name="Ticker", value_name="Indexed Price")
    line = alt.Chart(df_prices).mark_line(strokeWidth=1.8).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Indexed Price:Q", title="Indexed Price (base = 100)"),
        color=alt.Color("Ticker:N", scale=alt.Scale(range=PALETTE[:len(tickers)]), legend=alt.Legend(title="Ticker")),
        tooltip=["Ticker:N", alt.Tooltip("Date:T", format="%b %Y"), alt.Tooltip("Indexed Price:Q", format=".1f")],
    ).properties(height=460, background=SURFACE, padding={"left":20,"right":20,"top":20,"bottom":20}
    ).configure_axis(gridColor=BORDER, labelColor=MUTED, titleColor=MUTED, domainColor=BORDER, tickColor=BORDER
    ).configure_legend(labelColor=TEXT, titleColor=MUTED, fillColor=SURFACE, strokeColor=BORDER
    ).configure_view(strokeOpacity=0)
    st.altair_chart(line, use_container_width=True)

st.markdown("---")
st.markdown(f'<div style="font-family:monospace;font-size:0.7rem;color:#3a3f48;text-align:center;padding:0.5rem 0 1rem;">SMART PORTFOLIO OPTIMIZER · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)
