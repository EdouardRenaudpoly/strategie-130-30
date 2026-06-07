"""
Page Résumé — vue exécutive de la stratégie.

Sections :
  1. Métriques clés avec IC bootstrap 95%
  2. Rendement cumulé de toutes les configs vs benchmark (séparateur IS/OOS)
  3. Performance in-sample vs out-of-sample par config
  4. Importance des facteurs (factor weights moyens)
"""

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis.bootstrap import compute_bootstrap_ci, MetricCI

TRAIN_END = pd.Timestamp("2020-01-01")

_CONFIG_LABELS = {
    "sn1_bl0":     "130/30 — Référence",
    "sn1_bl0_hmm": "130/30 — HMM",
    "sn1_bl1_c5":  "130/30 — Black-Litterman",
    "lo20_sc3":    "Long-only 20 titres",
}
_CONFIG_COLORS = {
    "sn1_bl0":     "#60a5fa",
    "sn1_bl0_hmm": "#a78bfa",
    "sn1_bl1_c5":  "#34d399",
    "lo20_sc3":    "#f97316",
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _load_config(tag: str) -> dict | None:
    import os
    base = f"data/processed/backtest/{tag}"
    if not os.path.exists(f"{base}/portfolio_returns.csv"):
        return None
    ret   = pd.read_csv(f"{base}/portfolio_returns.csv", index_col=0, parse_dates=True).squeeze()
    bench = pd.read_csv(f"{base}/benchmark_returns.csv", index_col=0, parse_dates=True).squeeze()
    fw    = pd.read_csv(f"{base}/factor_weights.csv",    index_col=0, parse_dates=True)
    return {"returns": ret, "benchmark": bench, "factor_weights": fw, "tag": tag}


@st.cache_data(ttl=3600)
def _bootstrap(tag: str, n_boot: int = 1000) -> dict[str, MetricCI] | None:
    d = _load_config(tag)
    if d is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_bootstrap_ci(d["returns"], d["benchmark"], n_boot=n_boot)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _metric_card(label: str, ci: MetricCI, pct: bool = False):
    scale = 100 if pct else 1
    suffix = "%" if pct else ""
    d = 1 if pct else 3

    point_str = f"{ci.point * scale:.{d}f}{suffix}"
    ci_str    = f"IC 95 % : [{ci.lower * scale:.{d}f}{suffix}, {ci.upper * scale:.{d}f}{suffix}]"

    st.markdown(
        f"""
        <div style="
            background: #1e293b;
            border-radius: 10px;
            padding: 18px 20px 14px;
            border-left: 3px solid #60a5fa;
        ">
            <div style="color:#94a3b8; font-size:12px; text-transform:uppercase;
                        letter-spacing:0.08em; margin-bottom:6px;">{label}</div>
            <div style="color:#f1f5f9; font-size:28px; font-weight:700;
                        line-height:1;">{point_str}</div>
            <div style="color:#64748b; font-size:11px; margin-top:8px;">{ci_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics(tag: str):
    cis = _bootstrap(tag)
    if cis is None:
        st.warning("Données non disponibles pour cette config.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _metric_card("Sharpe Ratio", cis["sharpe"])
    with c2:
        _metric_card("Info Ratio", cis["ir"])
    with c3:
        _metric_card("Rendement annualisé", cis["annual_return"], pct=True)
    with c4:
        _metric_card("Max Drawdown", cis["max_drawdown"], pct=True)


def _render_cumulative_chart(available_tags: list[str]):
    fig = go.Figure()

    # Benchmark (une seule fois — même pour tous)
    bench_added = False
    for tag in available_tags:
        d = _load_config(tag)
        if d is None:
            continue
        bench = d["benchmark"]
        if not bench_added:
            cum_b = (1 + bench.dropna()).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=cum_b.index, y=cum_b.values * 100,
                name="Benchmark (EW S&P 500)",
                mode="lines",
                line=dict(color="#475569", width=1.5, dash="dot"),
            ))
            bench_added = True

        ret     = d["returns"].dropna()
        cum_ret = (1 + ret).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=cum_ret.index,
            y=cum_ret.values * 100,
            name=_CONFIG_LABELS.get(tag, tag),
            mode="lines",
            line=dict(color=_CONFIG_COLORS.get(tag, "#94a3b8"), width=2),
        ))

    # Séparateur IS / OOS
    fig.add_vline(
        x=TRAIN_END,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text="↑ Fin entraînement (jan. 2020)",
        annotation_position="top right",
        annotation_font_color="#f59e0b",
        annotation_font_size=11,
    )

    fig.update_layout(
        height=400,
        template="plotly_dark",
        yaxis_title="Rendement cumulé (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def _render_is_oos_table(available_tags: list[str]):
    rows = []
    for tag in available_tags:
        d = _load_config(tag)
        if d is None:
            continue
        ret   = d["returns"].dropna()
        bench = d["benchmark"].reindex(ret.index)

        is_ret  = ret[ret.index <  TRAIN_END]
        oos_ret = ret[ret.index >= TRAIN_END]
        is_b    = bench[bench.index <  TRAIN_END]
        oos_b   = bench[bench.index >= TRAIN_END]

        def sharpe(r):
            return r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan

        def ir(r, b):
            ex = r - b.reindex(r.index)
            return ex.mean() / ex.std() * np.sqrt(12) if ex.std() > 0 else np.nan

        rows.append({
            "Stratégie":         _CONFIG_LABELS.get(tag, tag),
            "Sharpe IS":         round(sharpe(is_ret),  3),
            "Sharpe OOS":        round(sharpe(oos_ret), 3),
            "IR IS":             round(ir(is_ret, is_b), 3),
            "IR OOS":            round(ir(oos_ret, oos_b), 3),
            "Rend. ann. IS (%)":  round(is_ret.mean()  * 1200, 1),
            "Rend. ann. OOS (%)": round(oos_ret.mean() * 1200, 1),
        })

    df = pd.DataFrame(rows).set_index("Stratégie")

    def color_oos(val):
        if not isinstance(val, float):
            return ""
        return "color: #22c55e" if val > 0 else "color: #ef4444"

    st.dataframe(
        df.style.map(color_oos, subset=["Sharpe OOS", "IR OOS", "Rend. ann. OOS (%)"]),
        width="stretch",
    )
    st.caption(
        f"IS = in-sample (avant {TRAIN_END.strftime('%b %Y')})  ·  "
        f"OOS = out-of-sample (après {TRAIN_END.strftime('%b %Y')})"
    )


def _render_factor_importance(tag: str):
    d = _load_config(tag)
    if d is None:
        return
    fw = d["factor_weights"]
    avg = fw.mean().sort_values(ascending=True)

    fig = go.Figure(go.Bar(
        x=avg.values,
        y=avg.index,
        orientation="h",
        marker_color=["#22c55e" if v > 0 else "#ef4444" for v in avg.values],
        text=[f"{v:.4f}" for v in avg.values],
        textposition="outside",
    ))
    fig.update_layout(
        height=320,
        template="plotly_dark",
        xaxis_title="Coefficient Ridge moyen",
        margin=dict(l=0, r=80, t=20, b=0),
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def render(bt: dict | None, available_tags: list[str]):
    # En-tête
    st.markdown("## Stratégie 130/30 Cross-Sectionnel")
    st.markdown(
        """
Stratégie quantitative cross-sectionelle qui combine **10 signaux** couvrant momentum,
qualité fondamentale (rentabilité brute, accruals, croissance des actifs), sentiment
(insider trading, SUE), et structure de marché (illiquidité, volatilité réalisée).
Les signaux sont agrégés chaque mois par **régression Ridge Fama-MacBeth** : les
coefficients factoriels sont estimés en walk-forward sur 36 mois glissants, puis pondérés
par similarité de régime HMM pour concentrer l'apprentissage sur les périodes
macroéconomiques proches du contexte actuel.

Le portefeuille 130/30 est construit par **optimisation quadratique sous contraintes** :
secteur-neutre (≤ 1 % d'écart par GICS), tracking error ≤ 6 %, turnover ≤ 30 %/mois,
coûts de transaction modélisés à 7 bps par transaction.
Toutes les données proviennent de sources publiques (yfinance · SEC EDGAR XBRL) — aucun
fournisseur payant, **entièrement reproductible**.
        """,
        help="Backtest walk-forward strict — aucun lookahead. "
             "Les IC bootstrap sont calculés par blocs de 12 mois pour préserver "
             "l'autocorrélation des rendements.",
    )
    st.divider()

    tags = [t for t in available_tags if t in _CONFIG_LABELS]
    if not tags:
        st.warning("Aucune config disponible. Lance `uv run python run_backtest.py`.")
        return

    # Config active (héritée du sélecteur sidebar)
    active_tag = None
    if bt is not None:
        # Identifier le tag de la config sélectionnée
        for tag in tags:
            d = _load_config(tag)
            if d is not None:
                ret_bt = bt["returns"]
                ret_d  = d["returns"]
                if ret_bt.index.equals(ret_d.index) and np.allclose(ret_bt.values, ret_d.values, atol=1e-10):
                    active_tag = tag
                    break
    if active_tag is None:
        active_tag = tags[0]

    # Section 1 — métriques avec IC
    st.subheader(f"Métriques — {_CONFIG_LABELS.get(active_tag, active_tag)}")
    st.caption("Estimations ponctuelles + intervalles de confiance bootstrap 95 % (1 000 réplications, blocs de 12 mois)")
    _render_metrics(active_tag)

    st.divider()

    # Section 2 — rendement cumulé toutes configs
    st.subheader("Rendement cumulé — toutes configurations")
    _render_cumulative_chart(tags)

    st.divider()

    # Section 3 — IS vs OOS
    st.subheader("Performance in-sample vs out-of-sample")
    _render_is_oos_table(tags)

    st.divider()

    # Section 4 — importance des facteurs
    st.subheader(f"Importance des facteurs — {_CONFIG_LABELS.get(active_tag, active_tag)}")
    st.caption("Moyenne des coefficients Ridge Fama-MacBeth sur toute la période walk-forward")
    _render_factor_importance(active_tag)
