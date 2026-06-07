import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.components import COLORS, explain


def _render_attribution(bt: dict, monthly_ret: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Attribution P&L par titre")
    explain(
        "L'attribution décompose le rendement mensuel du portefeuille titre par titre : "
        "<b>contribution = poids × rendement individuel</b>. Les positions longues avec "
        "un bon rendement contribuent positivement ; les shorts sur des titres qui baissent "
        "aussi. Cette vue permet d'identifier quel pari sectoriel ou individuel a dominé "
        "la performance d'un mois donné."
    )

    weights = bt["weights"]
    portfolio_returns = bt["returns"]

    # Mois disponibles = mois où on a un rendement de portefeuille ET des poids T-1
    earn_months = sorted(
        [t for t in portfolio_returns.index if t in monthly_ret.index],
        reverse=True,
    )
    if not earn_months:
        st.warning("Données de rendements individuels non disponibles.")
        return

    labels = [t.strftime("%B %Y") for t in earn_months]
    selected_label = st.selectbox("Mois à analyser", labels, index=0)
    earn_ts = earn_months[labels.index(selected_label)]

    # Poids au mois T-1 (rebalancement qui génère le rendement de earn_ts)
    rebal_ts = earn_ts - pd.DateOffset(months=1)
    rebal_ts = rebal_ts.replace(day=1)
    if rebal_ts not in weights.index:
        st.warning(f"Poids de rebalancement introuvables pour {rebal_ts.strftime('%B %Y')}.")
        return

    w    = weights.loc[rebal_ts]
    ret  = monthly_ret.loc[earn_ts]
    common = w.index.intersection(ret.index)
    w, ret = w[common], ret[common]

    contrib = (w * ret).dropna().sort_values(ascending=False)
    contrib = contrib[contrib != 0]

    total_portfolio = float(portfolio_returns.loc[earn_ts])
    total_attributed = float(contrib.sum())
    residual = total_portfolio - total_attributed

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Rendement total du portefeuille", f"{total_portfolio:.2%}")
    col_m2.metric("Contribution attribuée (positions connues)", f"{total_attributed:.2%}")
    if abs(residual) > 0.001:
        col_m3.metric("Résidu (coûts / positions non suivies)", f"{residual:.2%}")

    # Garder les top/bottom + tous les shorts significatifs
    top_n    = contrib[contrib > 0].head(15)
    bottom_n = contrib[contrib < 0].tail(10)
    display  = pd.concat([top_n, bottom_n]).sort_values(ascending=True)

    colors = [COLORS["long"] if v >= 0 else COLORS["short"] for v in display.values]

    fig_attr = go.Figure(go.Bar(
        x=display.values * 100,
        y=display.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in display.values * 100],
        textposition="outside",
        textfont=dict(size=10),
        customdata=[[float(w.get(t, 0)) * 100, float(ret.get(t, 0)) * 100]
                    for t in display.index],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Contribution : %{x:.2f}%<br>"
            "Poids : %{customdata[0]:.1f}%<br>"
            "Rendement titre : %{customdata[1]:.1f}%<extra></extra>"
        ),
    ))
    fig_attr.add_vline(x=0, line_color="white", line_width=1)
    fig_attr.update_layout(
        title=f"Top contributeurs et détracteurs — {selected_label}",
        xaxis_title="Contribution au rendement (%)",
        height=max(400, len(display) * 28),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=100),
    )
    st.plotly_chart(fig_attr, width="stretch")

    # Tableau détaillé
    with st.expander("Tableau complet des positions ce mois-là"):
        df_full = pd.DataFrame({
            "Poids (%)":       (w * 100).round(2),
            "Ret. titre (%)":  (ret * 100).round(2),
            "Contribution (%)": ((w * ret) * 100).round(3),
        }).dropna()
        df_full = df_full[df_full["Poids (%)"].abs() > 0.01].sort_values(
            "Contribution (%)", ascending=False
        )
        st.dataframe(
            df_full.style
            .background_gradient(subset=["Contribution (%)"], cmap="RdYlGn")
            .format({"Poids (%)": "{:.2f}", "Ret. titre (%)": "{:.2f}", "Contribution (%)": "{:+.3f}"}),
            width="stretch",
        )


def render(bt: dict, monthly_ret: pd.DataFrame | None = None) -> None:
    st.title("Performance du portefeuille")

    explain(
        "Cette page présente les résultats du backtest walk-forward : à chaque mois, "
        "le modèle utilise uniquement les données passées pour construire le portefeuille, "
        "puis mesure la performance réelle du mois suivant. Cela simule fidèlement ce qu'un "
        "gestionnaire aurait obtenu en temps réel, sans biais de regard en avant."
    )

    r  = bt["returns"]
    bm = bt["benchmark"]
    m  = bt["metrics"]

    bm_aligned = bm.reindex(r.index).fillna(0)
    beta  = float(np.cov(r.values, bm_aligned.values)[0, 1] / np.var(bm_aligned.values))
    alpha = float((r.mean() - beta * bm_aligned.mean()) * 12)
    var95 = float(np.percentile(r.values, 5))

    def kpi(col, label, value, help_text=""):
        col.metric(label, value, help=help_text)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Sharpe", f"{float(m['sharpe']):.3f}",
        "Rendement / écart-type annualisé. Au-dessus de 1.0 est excellent.")
    kpi(c2, "Info Ratio", f"{float(m['ir']):.3f}",
        "Rendement actif / tracking error. Mesure la valeur ajoutée vs le benchmark.")
    kpi(c3, "Rendement ann.", f"{float(m['annual_return']):.1%}",
        "Rendement mensuel moyen × 12.")
    kpi(c4, "Max Drawdown", f"{float(m['max_dd']):.1%}",
        "Perte maximale du pic au creux suivant.")

    c5, c6, c7, c8 = st.columns(4)
    kpi(c5, "Beta (β)", f"{beta:.3f}",
        "Sensibilité au marché. β < 1 : moins volatile que le benchmark. "
        "β = cov(r, r_bm) / var(r_bm).")
    kpi(c6, "Alpha (α) ann.", f"{alpha:.2%}",
        "Rendement non expliqué par le beta (Jensen). α = r - β·r_bm, annualisé.")
    kpi(c7, "VaR 95% (mensuel)", f"{var95:.2%}",
        "Dans 95 % des mois, la perte ne dépasse pas cette valeur.")
    kpi(c8, "Turnover moy.", f"{float(m['avg_turnover']):.1%}",
        "Fraction du portefeuille rééquilibrée chaque mois.")

    st.divider()

    explain(
        "La courbe de valeur montre la croissance de 1 $ investi depuis le debut du backtest. "
        "Le benchmark est un portefeuille equal-weight sur l'univers S&P 500 disponible à chaque date. "
        "L'écart entre les deux courbes représente l'alpha généré par le modèle."
    )

    cum_strat = (1 + r).cumprod()
    cum_bm    = (1 + bm_aligned).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_strat.index, y=cum_strat.values,
        name="Stratégie 130/30", line=dict(color=COLORS["strategy"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
    ))
    fig.add_trace(go.Scatter(
        x=cum_bm.index, y=cum_bm.values,
        name="Benchmark (equal-weight)", line=dict(color=COLORS["benchmark"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        title="Valeur du portefeuille (base 1)", height=400,
        xaxis_title="Date", yaxis_title="Valeur",
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        explain(
            "Le drawdown mesure la perte en cours de route : à chaque date, quelle fraction "
            "du capital de pointe a été perdue. Un drawdown de -35 % signifie qu'il a fallu "
            "remonter de 54 % pour revenir au niveau précédent."
        )
        drawdown = cum_strat / cum_strat.cummax() - 1
        fig_dd = go.Figure(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100,
            fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
            line=dict(color=COLORS["short"], width=1.5),
        ))
        fig_dd.update_layout(
            title="Drawdown", height=300, yaxis_title="%",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dd, width="stretch")

    with col2:
        explain(
            "Le Sharpe glissant sur 12 mois indique si le modèle génère de la valeur de façon "
            "consistante dans le temps. Un Sharpe qui reste positif à travers les cycles "
            "de marché est un signe de robustesse."
        )
        roll_sharpe = r.rolling(12).mean() / r.rolling(12).std() * np.sqrt(12)
        fig_rs = go.Figure(go.Scatter(
            x=roll_sharpe.index, y=roll_sharpe.values,
            line=dict(color=COLORS["accent"], width=2),
        ))
        fig_rs.add_hline(y=1.0, line_dash="dash", line_color=COLORS["strategy"],
                         annotation_text="Sharpe = 1")
        fig_rs.add_hline(y=0.0, line_dash="dot", line_color=COLORS["benchmark"])
        fig_rs.update_layout(
            title="Sharpe glissant 12 mois", height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rs, width="stretch")

    col_ret1, col_ret2 = st.columns(2)

    with col_ret1:
        st.subheader("Rendements annuels — stratégie vs benchmark")
        explain(
            "Comparaison année par année des rendements totaux. C'est la vue standard "
            "des rapports de gestion. Une barre verte qui dépasse le gris chaque année "
            "indique une génération d'alpha consistante à travers les cycles."
        )
        annual_strat = (1 + r).resample("YE").prod() - 1
        annual_bm    = (1 + bm_aligned).resample("YE").prod() - 1
        years        = annual_strat.index.year.tolist()

        fig_ann = go.Figure()
        fig_ann.add_trace(go.Bar(
            x=years, y=annual_bm.values * 100,
            name="Benchmark", marker_color=COLORS["benchmark"], opacity=0.7,
        ))
        fig_ann.add_trace(go.Bar(
            x=years, y=annual_strat.values * 100,
            name="Stratégie",
            marker_color=[COLORS["long"] if v >= 0 else COLORS["short"]
                          for v in annual_strat.values],
            text=[f"{v:.1f}%" for v in annual_strat.values * 100],
            textposition="outside",
        ))
        fig_ann.add_hline(y=0, line_color="white", line_width=0.5)
        fig_ann.update_layout(
            barmode="group", height=340, yaxis_title="%",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_ann, width="stretch")

    with col_ret2:
        st.subheader("Rendements mensuels — 24 derniers mois")
        explain(
            "Vue détaillée des 24 derniers mois. Permet de voir les clusters de "
            "mauvais mois (drawdowns) et la vitesse de récupération."
        )
        r_last24 = r.iloc[-24:] * 100
        fig_bar = go.Figure(go.Bar(
            x=r_last24.index.strftime("%b %Y"),
            y=r_last24.values,
            marker_color=[COLORS["long"] if v >= 0 else COLORS["short"]
                          for v in r_last24.values],
            text=[f"{v:.1f}%" for v in r_last24.values],
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig_bar.add_hline(y=0, line_color="white", line_width=0.5)
        fig_bar.update_layout(
            height=340, yaxis_title="%", xaxis_tickangle=-45,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, width="stretch")

    st.subheader("Poids des facteurs dans le temps (Ridge Fama-MacBeth)")
    explain(
        "Le modèle Fama-MacBeth estime chaque mois les coefficients d'une régression "
        "cross-sectionelle : pour tous les titres de l'univers, quel signal prédit le mieux "
        "le rendement du mois suivant ? Ridge régularise les coefficients pour éviter le "
        "surapprentissage. La courbe lissée (moyenne mobile 6M) montre quels facteurs ont "
        "été les plus persistants dans le temps — un coefficient stable est plus fiable "
        "qu'un coefficient qui change de signe d'un mois à l'autre."
    )
    fw = bt["factor_weights"].dropna()
    fig_fw = go.Figure()
    for col in fw.columns:
        fig_fw.add_trace(go.Scatter(
            x=fw.index, y=fw[col].rolling(6).mean(),
            name=col, mode="lines",
        ))
    fig_fw.update_layout(
        height=300,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig_fw, width="stretch")

    if monthly_ret is not None:
        _render_attribution(bt, monthly_ret)
