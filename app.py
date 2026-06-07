"""
Dashboard Stratégie 130/30 — point d'entrée Streamlit.
"""

import streamlit as st

from dashboard.components import GLOBAL_CSS
from dashboard.data import (
    list_backtest_configs, load_backtest,
    load_ic, load_sectors, load_signals, load_picks, load_returns,
    load_monthly_returns,
)
from dashboard.pages import performance, signals_ic, sectors, black_litterman, live_picks, stress, regime, summary

# ---------------------------------------------------------------------------
# Config globale
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Stratégie 130/30",
    page_icon="[130/30]",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chargement des données (mis en cache par Streamlit)
# ---------------------------------------------------------------------------

ic_s, ic_m   = load_ic()
sec_e, sec_s = load_sectors()
signals      = load_signals()
picks        = load_picks()
raw_returns   = load_returns()
monthly_ret   = load_monthly_returns()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Stratégie 130/30")
st.sidebar.caption("Cross-sectionnel | Multi-signal | S&P 500")

page = st.sidebar.radio("Navigation", [
    "Résumé",
    "Performance",
    "Signaux & IC",
    "Secteurs",
    "Black-Litterman",
    "Stress Tests",
    "Régime HMM",
    "Live Picks",
])

st.sidebar.divider()

_CONFIG_LABELS = {
    "sn1_bl0":     "130/30 — Référence",
    "sn1_bl0_hmm": "130/30 — HMM",
    "sn1_bl1_c5":  "130/30 — Black-Litterman",
    "lo20_sc3":    "Long-only 20 titres",
}
_configs = [c for c in list_backtest_configs() if c in _CONFIG_LABELS]
_selected_cfg = (
    st.sidebar.selectbox(
        "Configuration du backtest",
        options=_configs,
        format_func=lambda x: _CONFIG_LABELS.get(x, x),
    )
    if _configs else "sn1_bl0"
)

bt = load_backtest(_selected_cfg)

if bt:
    m = bt["metrics"]
    st.sidebar.metric("Sharpe",         f"{float(m['sharpe']):.3f}")
    st.sidebar.metric("Info Ratio",     f"{float(m['ir']):.3f}")
    st.sidebar.metric("Rendement ann.", f"{float(m['annual_return']):.1%}")
    st.sidebar.metric("Max Drawdown",   f"{float(m['max_dd']):.1%}")

# ---------------------------------------------------------------------------
# Routage
# ---------------------------------------------------------------------------

if page == "Résumé":
    summary.render(bt, _configs)

elif page == "Performance":
    if bt is None:
        st.warning("Lance d'abord `uv run python run_backtest.py` pour générer les résultats.")
    else:
        performance.render(bt, monthly_ret)

elif page == "Signaux & IC":
    if ic_m is None:
        st.warning("Lance `uv run python run_ic_analysis.py` pour générer l'analyse IC.")
    else:
        signals_ic.render(ic_s, ic_m, signals)

elif page == "Secteurs":
    if sec_s is None:
        st.warning("Lance `uv run python run_sector_analysis.py` pour générer l'analyse.")
    else:
        sectors.render(sec_e, sec_s)

elif page == "Black-Litterman":
    if raw_returns is None:
        st.warning("Données de rendements non disponibles.")
    else:
        black_litterman.render(raw_returns, picks, bt)

elif page == "Stress Tests":
    if bt is None:
        st.warning("Lance d'abord `uv run python run_backtest.py` pour générer les résultats.")
    else:
        stress.render(bt, monthly_ret)

elif page == "Régime HMM":
    regime.render(bt, monthly_ret)

elif page == "Live Picks":
    if picks.empty:
        st.warning("Lance `uv run python run_live_picks.py` pour générer les picks.")
    else:
        live_picks.render(picks)
