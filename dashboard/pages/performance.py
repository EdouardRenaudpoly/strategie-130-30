import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.components import COLORS


def _render_attribution(bt: dict, monthly_ret: pd.DataFrame) -> None:
    weights = bt["weights"]
    portfolio_returns = bt["returns"]

    earn_months = sorted(
        [t for t in portfolio_returns.index if t in monthly_ret.index],
        reverse=True,
    )
    if not earn_months:
        st.warning("Données de rendements individuels non disponibles.")
        return

    labels = [t.strftime("%B %Y") for t in earn_months]
    selected_label = st.selectbox("Mois à analyser", labels, index=0)
    earn_ts  = earn_months[labels.index(selected_label)]
    rebal_ts = (earn_ts - pd.DateOffset(months=1)).replace(day=1)

    if rebal_ts not in weights.index:
        st.warning(f"Poids de rebalancement introuvables pour {rebal_ts.strftime('%B %Y')}.")
        return

    w    = weights.loc[rebal_ts]
    ret  = monthly_ret.loc[earn_ts]
    common = w.index.intersection(ret.index)
    w, ret = w[common], ret[common]

    contrib = (w * ret).dropna().sort_values(ascending=False)
    contrib = contrib[contrib != 0]

    total_portfolio  = float(portfolio_returns.loc[earn_ts])
    total_attributed = float(contrib.sum())
    residual         = total_portfolio - total_attributed

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Rendement du portefeuille", f"{total_portfolio:.2%}")
    col_m2.metric("Contribution attribuée", f"{total_attributed:.2%}")
    if abs(residual) > 0.001:
        col_m3.metric("Résidu (coûts / autres)", f"{residual:.2%}")

    top_n    = contrib[contrib > 0].head(15)
    bottom_n = contrib[contrib < 0].tail(10)
    display  = pd.concat([top_n, bottom_n]).sort_values(ascending=True)
    colors   = [COLORS["long"] if v >= 0 else COLORS["short"] for v in display.values]

    fig = go.Figure(go.Bar(
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
    fig.add_vline(x=0, line_color="white", line_width=1)
    fig.update_layout(
        title=f"Contributeurs et détracteurs — {selected_label}",
        xaxis_title="Contribution au rendement (%)",
        height=max(380, len(display) * 26),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=80, r=100),
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Tableau complet des positions"):
        df_full = pd.DataFrame({
            "Poids (%)":        (w * 100).round(2),
            "Ret. titre (%)":   (ret * 100).round(2),
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

    r  = bt["returns"]
    bm = bt["benchmark"]
    m  = bt["metrics"]

    bm_aligned = bm.reindex(r.index).fillna(0)
    beta  = float(np.cov(r.values, bm_aligned.values)[0, 1] / np.var(bm_aligned.values))
    alpha = float((r.mean() - beta * bm_aligned.mean()) * 12)
    var95 = float(np.percentile(r.values, 5))

    tab1, tab2, tab3 = st.tabs(["Vue globale", "Analyse détaillée", "Attribution P&L"])

    # ------------------------------------------------------------------
    # Tab 1 — Vue globale
    # ------------------------------------------------------------------
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe", f"{float(m['sharpe']):.3f}",
                  help="Rendement / écart-type annualisé.")
        c2.metric("Info Ratio", f"{float(m['ir']):.3f}",
                  help="Rendement actif / tracking error vs benchmark.")
        c3.metric("Rendement ann.", f"{float(m['annual_return']):.1%}",
                  help="Rendement mensuel moyen × 12.")
        c4.metric("Max Drawdown", f"{float(m['max_dd']):.1%}",
                  help="Perte maximale du pic au creux.")

        cum_strat = (1 + r).cumprod()
        cum_bm    = (1 + bm_aligned).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cum_strat.index, y=cum_strat.values,
            name="Stratégie 130/30",
            line=dict(color=COLORS["strategy"], width=2.5),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.07)",
        ))
        fig.add_trace(go.Scatter(
            x=cum_bm.index, y=cum_bm.values,
            name="Benchmark (equal-weight)",
            line=dict(color=COLORS["benchmark"], width=1.5, dash="dot"),
        ))
        fig.update_layout(
            title="Valeur du portefeuille (base 1)",
            height=450,
            xaxis_title="Date", yaxis_title="Valeur",
            legend=dict(orientation="h", y=-0.12),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

        with st.expander("Métriques secondaires"):
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Beta (β)", f"{beta:.3f}",
                      help="Sensibilité au marché. β < 1 : moins volatile que le benchmark.")
            s2.metric("Alpha (α) ann.", f"{alpha:.2%}",
                      help="Rendement non expliqué par le beta (Jensen), annualisé.")
            s3.metric("VaR 95% (mensuel)", f"{var95:.2%}",
                      help="Dans 95 % des mois, la perte ne dépasse pas cette valeur.")
            s4.metric("Turnover moy.", f"{float(m['avg_turnover']):.1%}",
                      help="Fraction du portefeuille rééquilibrée chaque mois.")

    # ------------------------------------------------------------------
    # Tab 2 — Analyse détaillée
    # ------------------------------------------------------------------
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            drawdown = cum_strat / cum_strat.cummax() - 1
            fig_dd = go.Figure(go.Scatter(
                x=drawdown.index, y=drawdown.values * 100,
                fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
                line=dict(color=COLORS["short"], width=1.5),
            ))
            fig_dd.update_layout(
                title="Drawdown (%)", height=280, yaxis_title="%",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_dd, width="stretch")

        with col2:
            roll_sharpe = r.rolling(12).mean() / r.rolling(12).std() * np.sqrt(12)
            fig_rs = go.Figure(go.Scatter(
                x=roll_sharpe.index, y=roll_sharpe.values,
                line=dict(color=COLORS["accent"], width=2),
            ))
            fig_rs.add_hline(y=1.0, line_dash="dash", line_color=COLORS["strategy"],
                             annotation_text="Sharpe = 1")
            fig_rs.add_hline(y=0.0, line_dash="dot", line_color=COLORS["benchmark"])
            fig_rs.update_layout(
                title="Sharpe glissant 12 mois", height=280,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rs, width="stretch")

        col3, col4 = st.columns(2)

        with col3:
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
                title="Rendements annuels", barmode="group", height=300,
                yaxis_title="%",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_ann, width="stretch")

        with col4:
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
                title="Rendements mensuels — 24 derniers mois",
                height=300, yaxis_title="%", xaxis_tickangle=-45,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, width="stretch")

        with st.expander("Poids des facteurs dans le temps (Ridge Fama-MacBeth)"):
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
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_fw, width="stretch")

    # ------------------------------------------------------------------
    # Tab 3 — Attribution P&L
    # ------------------------------------------------------------------
    with tab3:
        if monthly_ret is not None:
            _render_attribution(bt, monthly_ret)
        else:
            st.info("Données de rendements individuels non disponibles.")
