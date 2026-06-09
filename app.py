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
from dashboard.pages import (
    performance, signals_ic, sectors, black_litterman,
    live_picks, stress, regime, summary,
)

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
# Session state (requis par segmented_control)
# ---------------------------------------------------------------------------

_PAGES = [
    "Résumé",
    "Performance",
    "Signaux & IC",
    "Secteurs",
    "Régime",
    "Stress Tests",
    "Black-Litterman",
    "Live Picks",
]

if "page" not in st.session_state:
    st.session_state.page = "Résumé"

# ---------------------------------------------------------------------------
# Navigation horizontale (top bar)
# ---------------------------------------------------------------------------

page = st.segmented_control(
    "Navigation",
    options=_PAGES,
    key="nav_control",
    label_visibility="collapsed",
    default="Résumé",
)

# Fallback si None (premier chargement edge case)
if page is None:
    page = st.session_state.page
else:
    st.session_state.page = page

st.divider()

# ---------------------------------------------------------------------------
# Données (mises en cache par Streamlit)
# ---------------------------------------------------------------------------

ic_s, ic_m   = load_ic()
sec_e, sec_s = load_sectors()
signals      = load_signals()
picks        = load_picks()
raw_returns  = load_returns()
monthly_ret  = load_monthly_returns()

# ---------------------------------------------------------------------------
# Sidebar — config uniquement
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding: 12px 0 20px;">
        <div style="font-size: 1.55rem; font-weight: 800; color: #f1f5f9;
                    letter-spacing: -0.02em; line-height: 1.1;">
            Stratégie<br>130/30
        </div>
        <div style="height: 2px; margin: 10px 0 8px;
                    background: linear-gradient(90deg, #00D4AA 0%, transparent 100%);
                    border-radius: 1px;"></div>
        <div style="font-size: 0.7rem; color: #64748b;
                    letter-spacing: 0.07em; text-transform: uppercase;">
            Cross-sectionnel · Multi-signal · S&P 500
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
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
        "Configuration",
        options=_configs,
        format_func=lambda x: _CONFIG_LABELS.get(x, x),
    )
    if _configs else "sn1_bl0"
)

bt = load_backtest(_selected_cfg)

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

elif page == "Régime":
    regime.render(bt, monthly_ret)

elif page == "Stress Tests":
    if bt is None:
        st.warning("Lance d'abord `uv run python run_backtest.py` pour générer les résultats.")
    else:
        stress.render(bt, monthly_ret)

elif page == "Black-Litterman":
    if raw_returns is None:
        st.warning("Données de rendements non disponibles.")
    else:
        black_litterman.render(raw_returns, picks, bt)

elif page == "Live Picks":
    if picks.empty:
        st.warning("Lance `uv run python run_live_picks.py` pour générer les picks.")
    else:
        live_picks.render(picks)
