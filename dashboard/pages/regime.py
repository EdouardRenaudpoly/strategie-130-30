"""
Page Régime HMM — visualisation des états de marché détectés par HMM.

Contenu :
  - Indicateur du régime actuel (bull/bear + probabilité)
  - Historique des probabilités de régime (aire empilée)
  - Évolution des features HMM avec zones de régime
  - Poids des facteurs par régime (bull vs bear)
  - Distribution des rendements mensuels par régime
  - Matrice de transition
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.regime.hmm import RegimeDetector, build_features


# ---------------------------------------------------------------------------
# Chargement et calcul HMM (mis en cache)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _compute_regime_data() -> dict:
    """Retourne les features HMM et les probabilités calculées sur toute la période."""
    returns = pd.read_csv(
        "data/processed/sp500_returns.csv", index_col=0, parse_dates=True
    )
    monthly = (1 + returns).resample("MS").prod() - 1
    features = build_features(monthly)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        det = RegimeDetector(n_states=2, random_state=42)
        det.fit(features)
        proba = det.predict_proba(features)
        transition_matrix = det._model.transmat_

    regime = proba["p_bull"].apply(lambda p: "bull" if p >= 0.5 else "bear")

    return {
        "monthly_returns": monthly,
        "features": features,
        "proba": proba,
        "regime": regime,
        "transition_matrix": transition_matrix,
    }


@st.cache_data(ttl=3600)
def _load_factor_weights(cfg_tag: str) -> pd.DataFrame | None:
    """Charge les factor weights d'une config HMM."""
    import os

    path = f"data/processed/backtest/{cfg_tag}/factor_weights.csv"
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ---------------------------------------------------------------------------
# Helpers de visualisation
# ---------------------------------------------------------------------------


def _add_regime_shading(fig: go.Figure, regime: pd.Series, row: int = 1, col: int = 1):
    """Ajoute des bandes verticales pour les périodes bull (vert pâle)."""
    in_bull = False
    start_date = None
    dates = regime.index.tolist()
    for i, dt in enumerate(dates):
        if regime.iloc[i] == "bull" and not in_bull:
            in_bull = True
            start_date = dt
        elif regime.iloc[i] == "bear" and in_bull:
            in_bull = False
            fig.add_vrect(
                x0=start_date,
                x1=dt,
                fillcolor="rgba(0,180,0,0.08)",
                layer="below",
                line_width=0,
                row=row,
                col=col,
            )
    if in_bull and start_date is not None:
        fig.add_vrect(
            x0=start_date,
            x1=dates[-1],
            fillcolor="rgba(0,180,0,0.08)",
            layer="below",
            line_width=0,
            row=row,
            col=col,
        )


# ---------------------------------------------------------------------------
# Sections de la page
# ---------------------------------------------------------------------------


def _render_current_regime(proba: pd.DataFrame):
    last = proba.iloc[-1]
    last_date = proba.index[-1]
    state = "Bull" if last["p_bull"] >= 0.5 else "Bear"

    col1, col2, col3 = st.columns(3)
    col1.metric("Régime actuel", state)
    col2.metric("P(Bull)", f"{last['p_bull']:.1%}")
    col3.metric("Dernière observation", last_date.strftime("%Y-%m"))


def _render_regime_timeline(proba: pd.DataFrame, bt: dict | None):
    """Aire empilée p_bull / p_bear + rendement cumulé du portefeuille en overlay."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.45],
        vertical_spacing=0.04,
        subplot_titles=("Probabilités de régime HMM", "Rendement cumulé du portefeuille"),
    )

    # Aire bull
    fig.add_trace(
        go.Scatter(
            x=proba.index,
            y=proba["p_bull"],
            name="P(Bull)",
            fill="tozeroy",
            mode="lines",
            line=dict(color="#22c55e", width=1),
            fillcolor="rgba(34,197,94,0.25)",
        ),
        row=1,
        col=1,
    )
    # Ligne seuil 50%
    fig.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4, row=1, col=1)

    # Rendement cumulé du portefeuille (si dispo)
    if bt is not None:
        ret = bt["returns"]
        cum = (1 + ret).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values * 100,
                name="Portefeuille (%)",
                mode="lines",
                line=dict(color="#60a5fa", width=2),
            ),
            row=2,
            col=1,
        )
        bench_cum = (1 + bt["benchmark"]).cumprod() - 1
        fig.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values * 100,
                name="Benchmark (%)",
                mode="lines",
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        height=480,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.update_yaxes(title_text="P(Bull)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Rendement cumulé (%)", row=2, col=1)
    st.plotly_chart(fig, width="stretch")


def _render_features(features: pd.DataFrame, regime: pd.Series):
    """Évolution des 3 features HMM avec zones de régime."""
    feature_labels = {
        "ew_return": "Rendement EW mensuel",
        "ew_vol_12m": "Volatilité 12 mois (EW)",
        "cs_dispersion": "Dispersion cross-sectionelle",
    }
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=list(feature_labels.values()),
    )
    colors = ["#60a5fa", "#f97316", "#a78bfa"]
    for i, (col_name, label) in enumerate(feature_labels.items(), start=1):
        _add_regime_shading(fig, regime, row=i, col=1)
        fig.add_trace(
            go.Scatter(
                x=features.index,
                y=features[col_name],
                name=label,
                mode="lines",
                line=dict(color=colors[i - 1], width=1.5),
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=560,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.caption("Les zones vertes correspondent aux périodes bull détectées par le HMM.")
    st.plotly_chart(fig, width="stretch")


def _render_factor_weights_by_regime(factor_weights: pd.DataFrame, regime: pd.Series):
    """Poids moyens des facteurs en bull vs bear."""
    aligned_regime = regime.reindex(factor_weights.index).ffill()
    bull_mask = aligned_regime == "bull"

    fw_bull = factor_weights[bull_mask].mean()
    fw_bear = factor_weights[~bull_mask].mean()

    factors = factor_weights.columns.tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Bull",
            x=factors,
            y=fw_bull.values,
            marker_color="#22c55e",
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Bear",
            x=factors,
            y=fw_bear.values,
            marker_color="#ef4444",
            opacity=0.85,
        )
    )
    fig.update_layout(
        barmode="group",
        height=360,
        template="plotly_dark",
        xaxis_title="Facteur",
        yaxis_title="Coefficient Ridge moyen",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    # Tableau résumé
    diff = fw_bull - fw_bear
    summary = pd.DataFrame(
        {
            "Bull": fw_bull.round(5),
            "Bear": fw_bear.round(5),
            "Δ (bull−bear)": diff.round(5),
        }
    )
    summary.index.name = "Facteur"
    st.dataframe(summary.sort_values("Δ (bull−bear)", ascending=False), width=600)


def _render_return_distribution(monthly_returns: pd.DataFrame, regime: pd.Series):
    """Distribution des rendements EW mensuels par régime (box plots)."""
    ew = monthly_returns.mean(axis=1)
    aligned = regime.reindex(ew.index)
    bull_ret = ew[aligned == "bull"] * 100
    bear_ret = ew[aligned == "bear"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=bull_ret,
            name=f"Bull ({len(bull_ret)} mois)",
            marker_color="#22c55e",
            boxmean="sd",
        )
    )
    fig.add_trace(
        go.Box(
            y=bear_ret,
            name=f"Bear ({len(bear_ret)} mois)",
            marker_color="#ef4444",
            boxmean="sd",
        )
    )
    fig.update_layout(
        height=380,
        template="plotly_dark",
        yaxis_title="Rendement mensuel EW (%)",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Bull** — μ={bull_ret.mean():.2f}%  σ={bull_ret.std():.2f}%")
    with col2:
        st.caption(f"**Bear** — μ={bear_ret.mean():.2f}%  σ={bear_ret.std():.2f}%")


def _render_transition_matrix(transition_matrix: np.ndarray):
    """Heatmap de la matrice de transition."""
    labels = ["Bear", "Bull"]
    # L'indice 0 = bear dans l'ordre de la matrice de transition HMM
    # On affiche bull→bull, bull→bear, bear→bull, bear→bear
    # pour être lisible on garde l'ordre interne mais on label correctement
    fig = go.Figure(
        go.Heatmap(
            z=transition_matrix * 100,
            x=[f"→ {l}" for l in labels],
            y=[f"De {l}" for l in labels],
            colorscale="Blues",
            text=[[f"{v:.1f}%" for v in row] for row in transition_matrix * 100],
            texttemplate="%{text}",
            showscale=False,
        )
    )
    fig.update_layout(
        height=240,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def render(bt: dict | None, _monthly_ret: pd.DataFrame | None = None):
    st.header("Régime de Marché — HMM")
    st.caption(
        "HMM gaussien 2 états entraîné sur toute la période disponible (visualisation). "
        "Dans le backtest, le HMM est entraîné de façon walk-forward sans lookahead."
    )

    try:
        data = _compute_regime_data()
    except Exception as e:
        st.error(f"Erreur lors du calcul HMM : {e}")
        return

    proba = data["proba"]
    regime = data["regime"]
    features = data["features"]
    monthly_returns = data["monthly_returns"]
    transition_matrix = data["transition_matrix"]

    # Indicateur courant
    _render_current_regime(proba)
    st.divider()

    # Onglets
    tab1, tab2, tab3, tab4 = st.tabs([
        "Historique des régimes",
        "Features HMM",
        "Facteurs par régime",
        "Matrice de transition",
    ])

    with tab1:
        _render_regime_timeline(proba, bt)

    with tab2:
        _render_features(features, regime)

    with tab3:
        # Chercher une config HMM disponible
        hmm_configs = ["sn1_bl0_hmm", "lo20_sc3_hmm"]
        cfg_labels = {
            "sn1_bl0_hmm": "130/30 — Secteur-neutre + HMM",
            "lo20_sc3_hmm": "Long-only 20 titres — HMM",
        }
        fw: pd.DataFrame | None = None
        used_cfg: str = ""
        for cfg in hmm_configs:
            fw = _load_factor_weights(cfg)
            if fw is not None:
                used_cfg = cfg
                break

        if fw is not None and used_cfg:
            st.caption(f"Poids des facteurs issus de : **{cfg_labels.get(used_cfg, used_cfg)}**")
            _render_factor_weights_by_regime(fw, regime)
        else:
            st.info(
                "Lance `uv run python run_backtest.py` avec une config HMM pour voir "
                "les poids des facteurs par régime."
            )

    with tab4:
        _render_return_distribution(monthly_returns, regime)
        st.divider()
        st.subheader("Matrice de transition")
        st.caption("Probabilité de rester dans / passer à chaque régime d'un mois à l'autre.")
        _render_transition_matrix(transition_matrix)
