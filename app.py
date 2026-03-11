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
PALETTE = [GOLD, TEAL, "#5b8dee", RED, "#a78bfa", "#f97316", "#84cc16", "#ec4899",
           "#06b6d4","#8b5cf6","#f43f5e","#10b981","#fb923c","#a3e635","#38bdf8","#e879f9"]

STOCK_UNIVERSE = {
    "Technology": {
        "Large Cap": ["AAPL","MSFT","NVDA","GOOGL","GOOG","META","AMZN","TSLA","AVGO","ORCL","ADBE","CRM","AMD","INTC","QCOM","TXN","NOW","INTU","IBM","AMAT","MU","LRCX","ADI","KLAC","MRVL","SNPS","CDNS","FTNT","PANW","CSCO"],
        "Mid Cap":   ["SNOW","DDOG","CRWD","ZS","NET","MDB","HUBS","BILL","TTD","GTLB","OKTA","SPLK","ESTC","CFLT","COUP","FIVN","PAYC","PCTY","RNG","SMAR","TOST","TWLO","VEEV","WEX","ZI","ZM","DOCN","HCP","APPF","AVLR"],
        "Small Cap": ["IONQ","SOUN","BBAI","KOPN","PDFS","SMTC","LPSN","KPLT","ACMR","ALGM","BAND","BIGC","BLKB","CEVA","COHU","CRUS","DIOD","FORM","ICHR","IMOS","IPWR","ITRN","KLIC","LIQT","LYTS","MCHX","MDXG","MFAC","MGNI","MRAM"],
    },
    "Healthcare": {
        "Large Cap": ["JNJ","UNH","LLY","ABBV","PFE","MRK","TMO","ABT","DHR","BMY","AMGN","GILD","CVS","CI","HUM","ELV","CNC","MOH","ISRG","SYK","ZBH","BAX","BDX","BSX","EW","HCA","IQV","MCK","MDT","RMD"],
        "Mid Cap":   ["ALGN","HOLX","ICLR","INSP","NVCR","PINC","PRVA","RCM","ACAD","ADMA","ADPT","ADUS","AFMD","AGEN","AGIO","AKBA","ALKS","ALNY","AMPH","ANAB","APLS","ARWR","ASMB","ASND","ATRC","AUPH","AVDL","AXNX","AZTA","BDSX"],
        "Small Cap": ["ACCD","ARDX","CDNA","CGEN","CLPT","CNMD","CODX","CRVS","ACET","ACHC","ACRS","ACRX","ACST","ADAP","ADCT","ADIL","ADMP","ADMS","ADNC","ADOC","ADPT","ADXN","AERI","AESI","AEVA","AFRI","AGEN","AGMH","AGNCN","AGNCM"],
    },
    "Finance": {
        "Large Cap": ["JPM","BAC","WFC","GS","MS","BLK","AXP","SPGI","CB","MMC","USB","PNC","TFC","COF","AIG","MET","PRU","AFL","ALL","TRV","ICE","CME","SCHW","BK","STT","NTRS","FITB","HBAN","KEY","RF"],
        "Mid Cap":   ["ALLY","CBSH","CFG","CMA","FHN","GBCI","IBOC","NBTB","PNFP","SFNC","ABCB","ABTX","ACBI","AFIN","AMNB","AMTB","ANCX","ANET","ANFH","ANGI","BOH","BOKF","BRO","BSVN","CADE","CALB","CASH","CATC","CBFV","CBNK"],
        "Small Cap": ["AROW","BYFC","CARV","CFFI","CLBK","COFS","CZWI","DCOM","EBTC","ACFC","ACGLO","ACGL","ACKB","ACLX","ACMC","ACNB","BCBP","BFIN","BFST","BGLC","BGNE","BHAT","BHLB","BHSE","BIDU","BIMI","BIOL","BIOX","BIRC"],
    },
    "Energy": {
        "Large Cap": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HES","PXD","DVN","HAL","BKR","FANG","MRO","APA","CTRA","EQT","AR","KMI","WMB","OKE","ET","EPD","MPLX","PAA","TRGP","LNG","CQP"],
        "Mid Cap":   ["CIVI","CRGY","GPRE","HESM","MNRL","MTDR","RRC","SM","VTLE","WTI","AMPY","BATL","BORR","CEIX","DNOW","ESTE","FLNG","GTE","HBT","IMPP","NRGU","REX","ROAN","ROCC","SBOW","SNDE","SRLP","STNG","TALO","TRMD"],
        "Small Cap": ["ACDC","ACER","ACES","ACEV","ACFC","NRGD","RINF","SITC","STNG","TELL","TPVG","TRMK","USWS","VAALCO","VAST","VERTEX","VET","VGAS","VKIN","VLRS"],
    },
    "Consumer Discretionary": {
        "Large Cap": ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","TJX","CMG","BKNG","MAR","HLT","YUM","DRI","ROST","ORLY","AZO","BBY","EBAY","F","GM","APTV","BWA","LEN","DHI","PHM","NVR","TOL","KBH"],
        "Mid Cap":   ["BYND","CAKE","CHUY","DENN","EAT","FAT","FRSH","JACK","KRUS","LOCO","CVCO","FIVE","FIZZ","FOXF","FRPH","FRPT","GDEN","GDRX","GFF","GFLU","GHM","GHRS","GIII","GKOS","GL","GLEO","GLNG","GLOG","GLOP","GLPG"],
        "Small Cap": ["ANDE","ARCH","BIRD","BOOT","CATO","CHRS","CLAR","CULP","DXLG","EXPR","ACBI","ACBK","ACCD","ACCE","BUCK","CATO","CHYS","CMPR","CONN","COOK","CROX","CTAS","CVNA","DECK","DENN","DHAT","DKNG","DMRC","DNKN","DNUT"],
    },
    "Consumer Staples": {
        "Large Cap": ["WMT","COST","PG","KO","PEP","PM","MO","MDLZ","CL","GIS","KHC","HSY","MKC","SJM","CAG","CPB","HRL","K","LW","KR"],
        "Mid Cap":   ["CELH","COTY","ELF","HAIN","HLF","IPAR","NOMD","OLLI","PRGO","SMPL","SPTN","TLRY","TROW","BYND","COKE","FARM","FIZZ","FLXS","FMNB","FMTB"],
        "Small Cap": ["ANDE","APPH","APRN","ARCO","CENT","CENTA","CHEF","CLOV","CLPS","CLRB","CLSD","CLSK","CLST","CLVS","CLWT","CLXT","CMCL","CMCO","CMCT","CMCX"],
    },
    "Industrials": {
        "Large Cap": ["HON","UPS","RTX","LMT","GE","CAT","DE","BA","MMM","EMR","ETN","PH","ROK","DOV","ITW","SWK","XYL","FAST","GWW","FDX","DAL","UAL","AAL","LUV","ALK","JBLU","SAVE","SKYW","MESA","AXON"],
        "Mid Cap":   ["AQUA","ARCB","ARLO","ARMK","ARNC","AROC","AZEK","AZPN","AZRE","AZTA","BCPC","CSTM","DXPE","EGL","EGAN","EGHT","EHAB","EHTH","EICA","EIDX","EIGI","EIKA","EILL","EIMB","EINK","EINV","EION","EIOV","EIPA","EIPO"],
        "Small Cap": ["ACXP","ADAC","ADAL","ADAM","ADAT","AEIS","AELT","AENZ","AERI","AESI","AEYE","AEZS","AFAC","AFAR","AFBI","AFCG","AFGH","AFHI","AFIB","AFIN"],
    },
    "Real Estate": {
        "Large Cap": ["AMT","PLD","CCI","EQIX","PSA","SPG","O","WELL","DLR","AVB","EQR","ESS","MAA","UDR","CPT","NNN","WPC","VICI","GLPI","BXP","SLG","VNO","KIM","REG","FRT","EXR","CUBE","LSI","NSA","LADR"],
        "Mid Cap":   ["ALEX","BRT","CHCT","CLDT","DEA","GMRE","HIW","INN","JBGS","KREF","AOMR","AHH","CLPR","CORR","DBRG","ELME","EPRT","ESRT","FCPT","GOOD","GPMT","ILPT","INVH","IOSP","IRET","JBGS","KREF","LAND","LMND","LMRK"],
        "Small Cap": ["ACEV","ACFC","ACGLO","ACGL","ACHC","ACIO","ACIX","ACKB","ACLX","ACMC","BRSP","BWXT","CBRL","CLDT","CLNC","CLPR","CLPS","CLRB","CLSD","CLSK"],
    },
    "ETFs & Index": {
        "Broad Market":  ["SPY","QQQ","IWM","DIA","VTI","VOO","IVV","SCHB","ITOT","VT","SCHX","SCHA","SCHM","VB","VO","VV","MGC","MGK","MGV","VONE"],
        "Sector ETFs":   ["XLK","XLF","XLE","XLV","XLC","XLI","XLY","XLP","XLU","XLRE","VGT","VFH","VDE","VHT","VOX","VIS","VCR","VDC","VPU","VNQ"],
        "Bond & Commodity": ["TLT","IEF","SHY","LQD","HYG","AGG","BND","BNDX","EMB","MUB","GLD","SLV","IAU","PDBC","GSG","DJP","COMT","COMB","BCI","RJI"],
        "International": ["EFA","EEM","VEA","VWO","IEFA","ACWI","IXUS","VXUS","SCZ","GXC","EWJ","EWG","EWU","EWC","EWA","EWZ","EWY","EWT","EWH","EWS"],
        "Thematic":      ["ARKK","ARKG","ARKW","ARKF","ARKQ","ARKX","BOTZ","ROBO","SNSR","AIQ","CLOUD","WCLD","CLOU","IGV","SKYY","HACK","BUG","CIBR","IHAK","FITE"],
    },
    "Crypto & Blockchain": {
        "Large Cap": ["COIN","MSTR","MARA","RIOT","HUT","CLSK","BTBT","CIFR","WGMI","BITO","GBTC","ETHE","BTCO","IBIT","FBTC","ARKB","HODL","EZBC","BRRR","DEFI"],
        "Mid Cap":   ["IREN","CORZ","SATO","BTCS","BKKT","GREE","DMGI","BTDR","WULF","NXRA","SOS","MIGI","BTCM","BTOG","HIVE","MGTI","SLNH","DIGI","CBIT","BTBT"],
        "Small Cap": ["BTCM","BTOG","HIVE","MGTI","SLNH","NKLA","PNTM","DIGI","CBIT","SOS","MIGI","HUTMF","SDIG","CIFR","APLD","NBTX","BTDG","BSRT","BSRR","BSVN"],
    },
    "Utilities": {
        "Large Cap": ["NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ED","EIX","ETR","PPL","FE","ES","CMS","NI","AEE","LNT","WEC","EVRG"],
        "Mid Cap":   ["CLNE","SPWR","FSLR","ENPH","RUN","NOVA","ARRY","CSIQ","JKS","DQ","AVA","BKH","GENIE","IDACORP","MGE","NJR","NWE","OGE","POR","SJW"],
        "Small Cap": ["AMRC","AMPE","AMPG","AMPH","AMPX","AMPY","AMRB","AMRN","AMRS","AMSC","AMSF","AMSG","AMST","AMSWA","AMTB","AMTBB","AMTD","AMTX","AMUB","AMTY"],
    },
    "Materials": {
        "Large Cap": ["LIN","APD","SHW","ECL","DD","DOW","NEM","FCX","NUE","STLD","RS","CMC","WRK","IP","PKG","SEE","SON","AVY","BLL","CCK"],
        "Mid Cap":   ["AXTA","BCPC","CLF","CSTM","GEF","HUN","KWR","MP","MEOH","OLN","TROX","VNTR","CC","CENX","CHEM","EMN","IOSP","KALU","KGC","LAUR"],
        "Small Cap": ["ASIX","LIQT","MTRN","NTIC","NVST","OESX","PRLB","PURE","SCCO","SHLX","SLCA","SQM","STCN","STLD","STRM","STRS","STRT","STRW","STSA","STTA"],
    },
    "Communications": {
        "Large Cap": ["GOOGL","META","NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","WBD","PARA","FOX","FOXA","IPG","OMC","NYT","NWSA","NWS","LBRDA","LBRDK"],
        "Mid Cap":   ["ATUS","CABO","CNSL","LUMN","SHEN","TDS","USM","OOMA","SATS","DISH","LMND","LMNL","LMND","LMNR","LMNS","LMNT","LMNX","LMNZ","LMOA","LMRK"],
        "Small Cap": ["ACNB","ACTG","ACTT","ACUA","ACUC","ACUN","ACUR","ACUS","ACVA","ACVF","IDT","LSXMA","LSXMB","LSXMK","LTRPA","LTRPB","LTRX","LUNA","LWAY","LWLG"],
    },
}

RISK_PROFILES = {
    "🟢 Conservative": {
        "desc": "Capital preservation. Low volatility, stable dividend payers.",
        "num_portfolios": 5000,
        "rf_rate": 0.045,
        "sectors": ["Real Estate","Finance","ETFs & Index","Utilities","Consumer Staples"],
        "caps": ["Large Cap","Broad Market","Sector ETFs","Bond & Commodity"],
    },
    "🟡 Moderate": {
        "desc": "Balanced growth and income. Mix of stability and upside.",
        "num_portfolios": 10000,
        "rf_rate": 0.04,
        "sectors": ["Technology","Healthcare","Finance","Consumer Discretionary","Industrials"],
        "caps": ["Large Cap","Mid Cap"],
    },
    "🔴 Aggressive": {
        "desc": "Maximum growth. High volatility, high potential returns.",
        "num_portfolios": 15000,
        "rf_rate": 0.02,
        "sectors": ["Technology","Crypto & Blockchain","Energy","Consumer Discretionary"],
        "caps": ["Mid Cap","Small Cap","Thematic"],
    },
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#0a0c10;--surface:#111318;--surface2:#181c24;--border:#1f2530;--gold:#c9a84c;--gold-dim:#7a6230;--teal:#3ecfb2;--text:#e8e4da;--muted:#6b7280;--radius:12px;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3{font-family:'DM Serif Display',serif!important;color:var(--text)!important;}
[data-testid="metric-container"]{background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.4rem!important;}
[data-testid="metric-container"] label{font-family:'DM Mono',monospace!important;font-size:0.7rem!important;letter-spacing:.12em;text-transform:uppercase;color:var(--muted)!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;font-size:1.8rem!important;color:var(--gold)!important;}
.stButton>button{background:var(--gold)!important;color:#0a0c10!important;border:none!important;border-radius:8px!important;font-family:'DM Mono',monospace!important;font-size:.82rem!important;letter-spacing:.08em;font-weight:500!important;padding:.6rem 1.6rem!important;width:100%;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{font-family:'DM Mono',monospace!important;font-size:.78rem!important;letter-spacing:.08em;color:var(--muted)!important;background:transparent!important;border:none!important;padding:.7rem 1.4rem!important;}
.stTabs [aria-selected="true"]{color:var(--gold)!important;border-bottom:2px solid var(--gold)!important;}
hr{border-color:var(--border)!important;}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.4rem 1.6rem;margin-bottom:1rem;}
.ticker-tag{display:inline-block;background:var(--surface2);border:1px solid var(--gold-dim);border-radius:6px;padding:2px 10px;font-family:'DM Mono',monospace;font-size:.78rem;color:var(--gold);margin:2px;}
.hero-title{font-family:'DM Serif Display',serif;font-size:2.6rem;line-height:1.1;color:var(--text);margin:0;}
.hero-sub{color:var(--muted);font-size:.95rem;margin-top:.4rem;}
.section-label{font-family:'DM Mono',monospace;font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:var(--gold-dim);margin-bottom:.5rem;}
input[type="text"]{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--border)!important;border-radius:6px!important;font-family:'DM Mono',monospace!important;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">◈ Portfolio Config</p>', unsafe_allow_html=True)

    # Risk profile
    st.markdown("### Risk Profile")
    risk_choice = st.radio("Risk", list(RISK_PROFILES.keys()), index=1, label_visibility="collapsed")
    risk = RISK_PROFILES[risk_choice]
    risk_color = {"🟢 Conservative": TEAL, "🟡 Moderate": GOLD, "🔴 Aggressive": RED}[risk_choice]
    st.markdown(f'<div style="background:{SURFACE};border:1px solid {risk_color};border-radius:8px;padding:.5rem .9rem;font-family:monospace;font-size:.75rem;color:{MUTED};margin-bottom:.5rem;">{risk["desc"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Selection mode
    st.markdown("### Stock Selection")
    mode = st.radio("Mode", ["🏭 Browse by Industry & Cap", "✏️ Enter Tickers Manually", "🔍 Search by Name/Ticker"], label_visibility="collapsed")

    tickers = []

    if mode == "🏭 Browse by Industry & Cap":
        all_sectors = list(STOCK_UNIVERSE.keys())
        sectors = st.multiselect("Industry", all_sectors, default=[s for s in risk["sectors"] if s in all_sectors][:2], label_visibility="visible")
        if sectors:
            all_caps = sorted(set(cap for s in sectors for cap in STOCK_UNIVERSE[s].keys()))
            caps = st.multiselect("Cap / Type", all_caps, default=[c for c in risk["caps"] if c in all_caps][:2], label_visibility="visible")
            if caps:
                pool = sorted(set(t for s in sectors for c in caps if c in STOCK_UNIVERSE[s] for t in STOCK_UNIVERSE[s][c]))
                st.markdown(f'<div style="font-family:monospace;font-size:.72rem;color:{MUTED};margin-bottom:.3rem;">{len(pool)} stocks available</div>', unsafe_allow_html=True)
                tickers = st.multiselect("Select Stocks", pool, default=pool[:6], label_visibility="visible")

    elif mode == "✏️ Enter Tickers Manually":
        raw = st.text_area("Tickers (comma or newline separated)", value="AAPL, MSFT, GOOGL, AMZN, META", height=100, label_visibility="collapsed")
        tickers = [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]

    else:  # Search mode
        search = st.text_input("Type ticker or company keyword", value="", label_visibility="collapsed", placeholder="e.g. apple, NVDA, health...")
        if search:
            query = search.upper().strip()
            matches = sorted(set(
                t for sector in STOCK_UNIVERSE.values()
                for cap_list in sector.values()
                for t in cap_list
                if query in t.upper()
            ))
            if matches:
                st.markdown(f'<div style="font-family:monospace;font-size:.72rem;color:{MUTED};">{len(matches)} matches</div>', unsafe_allow_html=True)
                tickers = st.multiselect("Results", matches, default=matches[:6], label_visibility="collapsed")
            else:
                st.warning("No matches found. Try a different keyword.")

    if tickers:
        st.markdown(" ".join(f'<span class="ticker-tag">{t}</span>' for t in tickers), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Date Range")
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("From", value=pd.Timestamp("2020-01-01"))
    with c2: end_date   = st.date_input("To",   value=pd.Timestamp("2025-01-01"))

    st.markdown("---")
    _sim_opts = [1_000, 2_500, 5_000, 10_000, 20_000]
    _sim_def  = min(_sim_opts, key=lambda x: abs(x - risk["num_portfolios"]))
    with st.expander("⚙️ Advanced Parameters"):
        risk_free_rate = st.slider("Risk-Free Rate", 0.0, 0.10, float(risk["rf_rate"]), 0.005, format="%.3f")
        num_portfolios = st.select_slider("Simulations", options=_sim_opts, value=_sim_def, format_func=lambda x: f"{x:,}")

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
    rc1, rc2, rc3 = st.columns(3)
    for col, key, clr in [(rc1,"🟢 Conservative",TEAL),(rc2,"🟡 Moderate",GOLD),(rc3,"🔴 Aggressive",RED)]:
        p = RISK_PROFILES[key]
        sel = f"border:2px solid {clr}" if key==risk_choice else f"border:1px solid {BORDER}"
        with col:
            st.markdown(f'<div class="card" style="{sel};text-align:center;min-height:130px;"><div style="font-size:1.8rem;margin-bottom:.4rem;">{key.split()[0]}</div><div style="font-family:\'DM Serif Display\',serif;font-size:1rem;margin-bottom:.4rem;color:{clr};">{key.split(" ",1)[1]}</div><div style="font-size:.78rem;color:{MUTED};">{p["desc"]}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    for col, icon, title, desc in [
        (i1,"◎","Efficient Frontier","Thousands of portfolios mapped across the risk-return spectrum"),
        (i2,"◈","Optimal Portfolios","Max Sharpe & Min Volatility via projected gradient optimisation"),
        (i3,"◉","870+ US Stocks","Browse by industry, cap size, or search any ticker"),
    ]:
        with col:
            st.markdown(f'<div class="card" style="text-align:center;min-height:120px;"><div style="font-size:1.8rem;margin-bottom:.5rem;color:{GOLD};">{icon}</div><div style="font-family:\'DM Serif Display\',serif;font-size:1rem;margin-bottom:.3rem;">{title}</div><div style="font-size:.8rem;color:{MUTED};">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;margin-top:2rem;color:{MUTED};font-family:monospace;font-size:.8rem;letter-spacing:.12em;">SELECT STOCKS & RISK PROFILE IN THE SIDEBAR → PRESS RUN</div>', unsafe_allow_html=True)
    st.stop()

if len(tickers) < 2:
    st.error("Please select at least 2 stocks to build a portfolio.")
    st.stop()

with st.spinner("Fetching data & running simulation…"):
    try:
        optimizer = PortfolioOptimizer(tickers, str(start_date), str(end_date), risk_free_rate)
        results   = optimizer.simulate_portfolios(num_portfolios)
        optimal   = optimizer.get_optimal_portfolios(results)
        corr      = optimizer.correlation_matrix()
    except Exception as e:
        st.error(f"**Error fetching data:** {e}  \n*Tip: some tickers may not be available on stooq. Try removing them.*")
        st.stop()

returns    = results["returns"]
volatility = results["volatility"]
sharpe     = results["sharpe"]
ms = optimal["max_sharpe"]
mv = optimal["min_volatility"]

st.markdown(f'<div style="background:{SURFACE};border:1px solid {risk_color};border-radius:10px;padding:.7rem 1.2rem;margin-bottom:1.2rem;font-family:monospace;font-size:.82rem;"><span style="color:{risk_color};font-weight:bold;">{risk_choice}</span>&nbsp;·&nbsp;<span style="color:{MUTED};">{risk["desc"]}</span>&nbsp;·&nbsp;<span style="color:{MUTED};">RF Rate {risk_free_rate*100:.1f}% · {num_portfolios:,} simulations · {len(tickers)} assets</span></div>', unsafe_allow_html=True)

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Assets",      len(tickers))
k2.metric("Simulations", f"{num_portfolios:,}")
k3.metric("Max Sharpe",  f"{ms['sharpe']:.3f}",           f"ret {ms['return']*100:.1f}%")
k4.metric("Min Vol",     f"{mv['volatility']*100:.1f}%",   f"ret {mv['return']*100:.1f}%")
k5.metric("Best Return", f"{returns.max()*100:.1f}%")
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["  EFFICIENT FRONTIER  ","  OPTIMAL WEIGHTS  ","  CORRELATION  ","  PRICE HISTORY  "])

with tab1:
    df_sim = pd.DataFrame({"Volatility (%)": volatility*100, "Return (%)": returns*100, "Sharpe Ratio": np.nan_to_num(sharpe, nan=0.0)})
    frontier = alt.Chart(df_sim).mark_circle(size=18, opacity=0.65).encode(
        x=alt.X("Volatility (%):Q", title="Volatility / Risk (%)"),
        y=alt.Y("Return (%):Q", title="Expected Annual Return (%)"),
        color=alt.Color("Sharpe Ratio:Q", scale=alt.Scale(scheme="goldorange"), legend=alt.Legend(title="Sharpe Ratio")),
        tooltip=["Volatility (%):Q","Return (%):Q","Sharpe Ratio:Q"],
    )
    df_opt = pd.DataFrame([
        {"Volatility (%)": ms["volatility"]*100,"Return (%)": ms["return"]*100,"Portfolio": f"★ Max Sharpe ({ms['sharpe']:.3f})","size":300},
        {"Volatility (%)": mv["volatility"]*100,"Return (%)": mv["return"]*100,"Portfolio": f"◆ Min Volatility ({mv['volatility']*100:.1f}%)","size":200},
    ])
    opt_pts = alt.Chart(df_opt).mark_point(filled=True,opacity=1).encode(
        x="Volatility (%):Q", y="Return (%):Q",
        color=alt.Color("Portfolio:N", scale=alt.Scale(range=[GOLD,TEAL]), legend=alt.Legend(title="Optimal")),
        size=alt.Size("size:Q", legend=None),
        tooltip=["Portfolio:N","Volatility (%):Q","Return (%):Q"],
    )
    st.altair_chart((frontier+opt_pts).properties(height=480,background=SURFACE,padding={"left":20,"right":20,"top":20,"bottom":20}).configure_axis(gridColor=BORDER,labelColor=MUTED,titleColor=MUTED,domainColor=BORDER,tickColor=BORDER).configure_legend(labelColor=TEXT,titleColor=MUTED,fillColor=SURFACE,strokeColor=BORDER).configure_view(strokeOpacity=0), use_container_width=True)

with tab2:
    col_ms, col_mv = st.columns(2)
    for col, label, portfolio, color in [(col_ms,"Max Sharpe Ratio",ms,GOLD),(col_mv,"Min Volatility",mv,TEAL)]:
        with col:
            st.markdown(f'<div class="section-label">{label}</div><span style="font-family:monospace;font-size:.8rem;color:{color};">Sharpe {portfolio["sharpe"]:.3f}</span> <span style="font-family:monospace;font-size:.75rem;color:{MUTED};">· ret {portfolio["return"]*100:.1f}% · vol {portfolio["volatility"]*100:.1f}%</span>', unsafe_allow_html=True)
            df_w = pd.DataFrame({"Ticker":tickers,"Allocation (%)":portfolio["weights"]*100}).sort_values("Allocation (%)",ascending=False)
            bar = alt.Chart(df_w).mark_bar(cornerRadiusTopRight=4,cornerRadiusBottomRight=4).encode(
                x=alt.X("Allocation (%):Q",title="Allocation (%)"),
                y=alt.Y("Ticker:N",sort="-x",title=None),
                color=alt.value(color),
                tooltip=["Ticker:N",alt.Tooltip("Allocation (%):Q",format=".1f")],
            ).properties(height=max(200,len(tickers)*38),background=SURFACE,padding={"left":10,"right":30,"top":10,"bottom":10}).configure_axis(gridColor=BORDER,labelColor=TEXT,titleColor=MUTED,domainColor=BORDER,tickColor=BORDER).configure_view(strokeOpacity=0)
            st.altair_chart(bar, use_container_width=True)
            df_w["Allocation (%)"] = df_w["Allocation (%)"].map(lambda x: f"{x:.1f}%")
            st.dataframe(df_w.set_index("Ticker"), use_container_width=True)

with tab3:
    rows = [{"Asset A":tickers[i],"Asset B":tickers[j],"ρ":round(float(corr.values[i,j]),3)} for i in range(len(tickers)) for j in range(len(tickers))]
    df_corr = pd.DataFrame(rows)
    heatmap = alt.Chart(df_corr).mark_rect().encode(
        x=alt.X("Asset A:N",title=None), y=alt.Y("Asset B:N",title=None),
        color=alt.Color("ρ:Q",scale=alt.Scale(domain=[-1,0,1],range=["#1a3a5c",SURFACE,"#7a3f1e"]),legend=alt.Legend(title="ρ")),
        tooltip=["Asset A:N","Asset B:N","ρ:Q"],
    )
    text = alt.Chart(df_corr).mark_text(fontSize=11,fontWeight="bold").encode(
        x="Asset A:N", y="Asset B:N", text=alt.Text("ρ:Q",format=".2f"), color=alt.value(TEXT),
    )
    st.altair_chart((heatmap+text).properties(height=max(380,len(tickers)*45),background=SURFACE,padding={"left":20,"right":20,"top":20,"bottom":20}).configure_axis(labelColor=TEXT,titleColor=MUTED,domainColor=BORDER,tickColor=BORDER,gridColor=BORDER).configure_legend(labelColor=TEXT,titleColor=MUTED,fillColor=SURFACE,strokeColor=BORDER).configure_view(strokeOpacity=0), use_container_width=True)
    st.markdown(f'<div style="font-family:monospace;font-size:.78rem;color:{MUTED};">ρ → +1: move together · ρ → −1: move inversely (max diversification)</div>', unsafe_allow_html=True)

with tab4:
    prices_norm = optimizer.price_data / optimizer.price_data.iloc[0] * 100
    prices_norm.index = pd.to_datetime(prices_norm.index)
    df_p = prices_norm.reset_index()
    df_p = df_p.rename(columns={df_p.columns[0]: "Date"})
    df_p = df_p.melt(id_vars="Date", var_name="Ticker", value_name="Indexed Price")
    line = alt.Chart(df_p).mark_line(strokeWidth=1.8).encode(
        x=alt.X("Date:T",title="Date"),
        y=alt.Y("Indexed Price:Q",title="Indexed Price (base = 100)"),
        color=alt.Color("Ticker:N",scale=alt.Scale(range=PALETTE[:len(tickers)]),legend=alt.Legend(title="Ticker")),
        tooltip=["Ticker:N",alt.Tooltip("Date:T",format="%b %Y"),alt.Tooltip("Indexed Price:Q",format=".1f")],
    ).properties(height=460,background=SURFACE,padding={"left":20,"right":20,"top":20,"bottom":20}).configure_axis(gridColor=BORDER,labelColor=MUTED,titleColor=MUTED,domainColor=BORDER,tickColor=BORDER).configure_legend(labelColor=TEXT,titleColor=MUTED,fillColor=SURFACE,strokeColor=BORDER).configure_view(strokeOpacity=0)
    st.altair_chart(line, use_container_width=True)

st.markdown("---")
st.markdown(f'<div style="font-family:monospace;font-size:.7rem;color:#3a3f48;text-align:center;padding:.5rem 0 1rem;">SMART PORTFOLIO OPTIMIZER · FOR EDUCATIONAL PURPOSES ONLY · NOT FINANCIAL ADVICE</div>', unsafe_allow_html=True)
