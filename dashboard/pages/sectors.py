import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.components import COLORS, explain


def render(sec_e: pd.DataFrame | None, sec_s: pd.DataFrame) -> None:
    st.title("Expositions sectorielles")

    explain(
        "Une stratégie quantitative peut générer des biais sectoriels involontaires : si les "
        "titres tech ont tendance à avoir des scores d'alpha élevés, le portefeuille sera "
        "massivement surpondéré en tech, et la performance reflétera en grande partie le "
        "risque sectoriel plutôt que l'alpha pur. La neutralisation sectorielle corrige "
        "cela en centrant chaque signal à l'intérieur de chaque secteur avant la régression."
    )

    sec_s_clean = sec_s.drop(index="Unknown", errors="ignore")

    st.subheader("Exposition active moyenne par secteur")
    explain(
        "L'exposition active est la différence entre le poids du secteur dans le portefeuille "
        "et son poids dans le benchmark equal-weight. Une barre positive signifie une "
        "surpondération (overweight), négative une sous-pondération (underweight). "
        "Un gestionnaire actif cherche à avoir des expositions actives justifiées par un "
        "signal d'alpha, pas par un biais accidentel du modèle."
    )
    active = sec_s_clean["active_avg"].sort_values()
    colors = [COLORS["long"] if v > 0 else COLORS["short"] for v in active]

    fig_sec = go.Figure()
    fig_sec.add_trace(go.Bar(
        x=active.values * 100, y=active.index,
        orientation="h", marker_color=colors, name="Exposition active moyenne",
        text=[f"{v:+.1f}%" for v in active.values * 100],
        textposition="outside",
    ))
    fig_sec.add_vline(x=0, line_color="white", line_width=1)
    fig_sec.update_layout(
        height=420, xaxis_title="Exposition active (%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_sec, width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portefeuille vs benchmark par secteur")
        explain(
            "Comparaison directe des poids alloués à chaque secteur dans le portefeuille "
            "(moyenne sur toute la période) versus le benchmark equal-weight."
        )
        fig_lv = go.Figure()
        fig_lv.add_trace(go.Bar(
            name="Benchmark", x=sec_s_clean.index,
            y=sec_s_clean["benchmark"] * 100,
            marker_color=COLORS["benchmark"], opacity=0.6,
        ))
        fig_lv.add_trace(go.Bar(
            name="Portefeuille", x=sec_s_clean.index,
            y=sec_s_clean["portfolio_avg"] * 100,
            marker_color=COLORS["strategy"],
        ))
        fig_lv.update_layout(
            barmode="group", height=380, yaxis_title="%",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-35, legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_lv, width="stretch")

    with col2:
        if sec_e is not None and not sec_e.empty:
            st.subheader("Drift sectoriel dans le temps")
            explain(
                "Le drift montre comment l'allocation sectorielle évolue d'un mois à l'autre "
                "au fil du backtest. Un drift important indique que le modèle change fréquemment "
                "ses paris sectoriels, ce qui contribue au turnover et aux coûts de transaction."
            )
            sec_e_clean = sec_e.drop(columns=["Unknown"], errors="ignore")
            fig_area = go.Figure()
            for col in sec_e_clean.columns:
                fig_area.add_trace(go.Scatter(
                    x=sec_e_clean.index, y=sec_e_clean[col].rolling(3).mean(),
                    name=col, stackgroup="one", mode="lines",
                ))
            fig_area.update_layout(
                height=380, yaxis_title="Exposition nette",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
            )
            st.plotly_chart(fig_area, width="stretch")
