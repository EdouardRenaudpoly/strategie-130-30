import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.components import COLORS, explain


def render(picks: pd.DataFrame) -> None:
    st.title("Live Picks — Portefeuille du mois")

    explain(
        "Les picks sont générés chaque début de mois par <code>run_live_picks.py</code>. "
        "Le modèle estime les factor weights Ridge sur les 36 derniers mois, calcule un "
        "score alpha pour chaque titre de l'univers, puis sélectionne les N meilleures "
        "opportunités longues. Les poids sont construits soit proportionnellement aux "
        "scores (avec plafonnage), soit via Black-Litterman avec les market caps comme prior. "
        "Ce portefeuille est conçu pour être déployé manuellement sur Wealthsimple (long-only)."
    )

    last_month = picks["month"].max()
    current    = picks[picks["month"] == last_month]
    longs      = current[current["side"] == "long"].copy()
    shorts     = current[current["side"] == "short"].copy()

    st.subheader(f"Picks de {last_month}")

    col1, col2 = st.columns([2, 1])

    with col1:
        explain(
            "Le treemap représente l'allocation : chaque rectangle est un titre, "
            "la surface est proportionnelle au poids dans le portefeuille, "
            "la couleur indique le secteur GICS. Survoler un titre affiche le "
            "score alpha, le secteur et le montant alloué."
        )
        if not longs.empty and longs["weight_pct"].notna().any():
            fig_tree = go.Figure(go.Treemap(
                labels=longs["ticker"],
                parents=longs["sector"].fillna("Other"),
                values=longs["weight_pct"].fillna(0),
                customdata=longs[["alpha_score", "amount_cad"]],
                hovertemplate=(
                    "<b>%{label}</b><br>Secteur: %{parent}<br>"
                    "Poids: %{value:.1f}%<br>"
                    "Score alpha: %{customdata[0]:.4f}<br>"
                    "Montant: $%{customdata[1]}<extra></extra>"
                ),
                texttemplate="<b>%{label}</b><br>%{value:.1f}%",
                marker=dict(colorscale="Teal"),
            ))
            fig_tree.update_layout(
                height=400, paper_bgcolor="rgba(0,0,0,0)",
                title="Allocation du portefeuille long",
            )
            st.plotly_chart(fig_tree, width="stretch")

    with col2:
        st.subheader("Positions longues")
        st.dataframe(
            longs[["ticker", "sector", "alpha_score", "weight_pct", "amount_cad"]]
            .rename(columns={"alpha_score": "Score", "weight_pct": "Poids %", "amount_cad": "$"})
            .set_index("ticker")
            .style.background_gradient(subset=["Score"], cmap="Greens"),
            width="stretch",
            height=380,
        )

        st.subheader("Shorts theoriques")
        explain(
            "Les shorts sont les titres avec les scores alpha les plus bas — "
            "ceux que le modèle juge les plus surévalués. Ils sont fournis "
            "à titre indicatif uniquement ; le portefeuille déployé est long-only."
        )
        st.dataframe(
            shorts[["ticker", "sector", "alpha_score"]]
            .rename(columns={"alpha_score": "Score"})
            .set_index("ticker")
            .style.background_gradient(subset=["Score"], cmap="Reds_r"),
            width="stretch",
        )

    if picks["month"].nunique() > 1:
        st.subheader("Historique des positions longues")
        explain(
            "L'historique montre comment les positions évoluent d'un mois à l'autre. "
            "Un titre qui reste dans le portefeuille plusieurs mois de suite indique "
            "que son signal alpha est persistant. Une rotation élevée peut signaler "
            "soit un modèle instable, soit une réponse saine aux nouvelles informations."
        )
        pivot_hist = picks[picks["side"] == "long"].pivot_table(
            index="month", columns="ticker", values="weight_pct", fill_value=0
        )
        fig_hist = go.Figure()
        for ticker in pivot_hist.columns:
            fig_hist.add_trace(go.Scatter(
                x=pivot_hist.index, y=pivot_hist[ticker],
                name=ticker, stackgroup="one", mode="lines",
            ))
        fig_hist.update_layout(
            height=350, yaxis_title="Poids (%)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_hist, width="stretch")

    st.divider()
    st.subheader("Prochain rebalancement")
    next_rebal = pd.Timestamp(last_month) + pd.DateOffset(months=1)
    days_left  = (next_rebal - pd.Timestamp.today()).days
    st.info(
        f"**{next_rebal.strftime('%d %B %Y')}** — dans {days_left} jours\n\n"
        "Commandes : `uv run python refresh.py --no-insider` "
        "puis `uv run python run_live_picks.py --bl`"
    )
