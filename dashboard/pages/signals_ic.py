import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.components import COLORS, explain


def render(ic_s: pd.DataFrame, ic_m: pd.DataFrame, signals: dict) -> None:
    st.title("Signaux & Information Coefficient")

    explain(
        "L'Information Coefficient (IC) est la corrélation de rang de Spearman entre le score "
        "d'un signal à la date t et le rendement réalisé au mois t+1. Un IC positif signifie "
        "que le signal classe correctement les titres : ceux qu'il désigne comme attractifs "
        "tendent effectivement à surperformer. L'ICIR (IC / écart-type de l'IC x sqrt(12)) "
        "mesure la régularité de ce signal dans le temps — une valeur supérieure à 0.5 est "
        "considérée comme solide en pratique."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("ICIR par signal")
        ic_sorted = ic_m.sort_values("ICIR", ascending=True)
        colors = [COLORS["long"] if v > 0 else COLORS["short"] for v in ic_sorted["ICIR"]]

        fig_icir = go.Figure(go.Bar(
            x=ic_sorted["ICIR"], y=ic_sorted.index,
            orientation="h", marker_color=colors,
            error_x=dict(
                type="data",
                array=(ic_sorted["mean_IC"] / ic_sorted["ICIR"].abs().clip(lower=0.001) * 0.1).abs(),
            ),
            text=[f"IC={v:.4f}  t={t:.2f}"
                  for v, t in zip(ic_sorted["mean_IC"], ic_sorted["t_stat"])],
            textposition="outside",
        ))
        fig_icir.add_vline(x=0, line_color="white", line_width=1)
        fig_icir.update_layout(
            height=380, xaxis_title="ICIR",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_icir, width="stretch")

    with col2:
        st.subheader("Résumé IC")
        explain(
            "Hit rate : fraction des mois où l'IC est positif. "
            "t-stat : significativité statistique de l'IC moyen. "
            "Une valeur |t| > 2 indique que le signal n'est pas du bruit pur."
        )
        display = ic_m[["mean_IC", "ICIR", "hit_rate", "t_stat"]].copy()
        display.columns = ["Mean IC", "ICIR", "Hit%", "t-stat"]
        display["Hit%"] = (display["Hit%"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            display.sort_values("ICIR", ascending=False).style
            .background_gradient(subset=["ICIR"], cmap="RdYlGn"),
            width="stretch",
        )

    st.subheader("IC cumulatif par signal")
    explain(
        "L'IC cumulatif est la somme des IC mensuels dans le temps. Une droite montante "
        "indique un signal qui prédit régulièrement dans le bon sens. Une ligne plate ou "
        "descendante révèle qu'un signal a perdu sa capacité prédictive — souvent parce "
        "que la prime de risque sous-jacente a été arbitragée par le marché."
    )
    ic_cum = ic_s.fillna(0).cumsum()
    fig_cum = go.Figure()
    for col in ic_cum.columns:
        fig_cum.add_trace(go.Scatter(
            x=ic_cum.index, y=ic_cum[col],
            name=col, mode="lines", line=dict(width=2),
        ))
    fig_cum.add_hline(y=0, line_dash="dot", line_color="white")
    fig_cum.update_layout(
        height=350, yaxis_title="IC cumulatif",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_cum, width="stretch")

    st.subheader("Corrélation entre signaux")
    explain(
        "Des signaux fortement corrélés entre eux apportent de l'information redondante : "
        "les inclure tous n'améliore pas le modèle et peut même nuire via le surapprentissage. "
        "Idéalement, les signaux sont faiblement corrélés — chacun capture alors une source "
        "d'alpha distincte. La régression Ridge atténue ce problème de colinéarité mais ne "
        "le résout pas complètement."
    )
    if signals:
        monthly = {
            name: df.resample("MS").last().stack().rename(name)
            for name, df in signals.items()
        }
        panel = pd.concat(monthly.values(), axis=1).dropna()
        corr  = panel.corr()

        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=11),
        ))
        fig_corr.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, width="stretch")
