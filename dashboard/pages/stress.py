import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.components import COLORS, explain
from src.stress.scenarios import sector_shock, historical_crises, factor_knockout
from src.analysis.sector_analysis import load_sector_map


@st.cache_data
def _sector_map_cached(tickers: tuple) -> dict:
    return load_sector_map(list(tickers))


def render(bt: dict, monthly_ret: pd.DataFrame | None) -> None:
    st.title("Stress Tests & Scénarios")

    explain(
        "Les stress tests répondent à une question concrète : <b>que se passe-t-il si un scénario "
        "adverse se réalise ?</b> Contrairement aux métriques historiques qui décrivent ce qui s'est "
        "passé, les stress tests projettent le portefeuille actuel dans des scénarios hypothétiques "
        "ou historiques. Trois familles : chocs sectoriels configurables, replay de crises passées, "
        "et sensibilité aux facteurs du modèle."
    )

    if monthly_ret is None:
        st.warning("Données de rendements mensuels non disponibles.")
        return

    weights_df = bt["weights"]
    last_w     = weights_df.iloc[-1]
    fw_last    = bt["factor_weights"].dropna().iloc[-1]
    r          = bt["returns"]
    bm         = bt["benchmark"].reindex(r.index).fillna(0)

    tickers_tuple = tuple(last_w.index.tolist())
    sector_map    = _sector_map_cached(tickers_tuple)

    tab1, tab2, tab3 = st.tabs([
        "Choc sectoriel",
        "Crises historiques",
        "Knockout de facteur",
    ])

    # -----------------------------------------------------------------------
    # TAB 1 : Choc sectoriel
    # -----------------------------------------------------------------------
    with tab1:
        explain(
            "Simule ce qui arrive au portefeuille si un secteur entier subit un choc résiduel "
            "négatif — c'est-à-dire un rendement inférieur à sa moyenne historique. "
            "Les autres secteurs continuent à leur rendement moyen. Cela isole l'exposition "
            "directe du portefeuille au secteur choisi, indépendamment du bruit de marché."
        )

        sectors_present = sorted({
            s for t, s in sector_map.items()
            if t in last_w.index and last_w.get(t, 0) != 0 and s != "Unknown"
        })
        all_sectors = sorted({s for s in sector_map.values() if s != "Unknown"})

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            target_sector = st.selectbox(
                "Secteur choqué",
                options=all_sectors,
                index=all_sectors.index("Technology") if "Technology" in all_sectors else 0,
            )
        with col_s2:
            shock_pct = st.slider(
                "Choc total (%)",
                min_value=-60, max_value=-5, value=-30, step=5,
                format="%d%%",
            )
        with col_s3:
            n_months_shock = st.slider("Durée (mois)", min_value=1, max_value=6, value=2)

        result = sector_shock(
            weights        = last_w,
            sector_map     = sector_map,
            monthly_ret_history = monthly_ret.reindex(columns=last_w.index),
            target_sector  = target_sector,
            shock_total    = shock_pct / 100,
            n_months       = n_months_shock,
        )

        # Métriques clés
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Impact portefeuille", f"{result['impact']:.2%}",
                   help="Perte additionnelle due au choc vs scénario de base.")
        mc2.metric("Rendement stressé total", f"{result['cumulative_return']:.2%}",
                   help=f"Sur {n_months_shock} mois en incluant le choc.")
        mc3.metric("Rendement base (sans choc)", f"{result['baseline_return']:.2%}",
                   help="Ce qu'aurait fait le portefeuille à ses rendements moyens historiques.")

        exposed_w = sum(last_w.get(t, 0) for t in result["shocked_tickers"])
        mc4.metric(f"Exposition {target_sector}", f"{exposed_w:.1%}",
                   help="Fraction du portefeuille dans le secteur choqué.")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            explain(
                "Chaque barre est la contribution d'un titre au rendement du portefeuille "
                "pendant la période de stress. Les barres rouges foncées appartiennent au secteur "
                "choqué — leur rendement inclut le choc résiduel. Les barres grises sont les "
                "autres positions qui continuent à leur moyenne historique."
            )
            contrib = result["position_contributions"]
            contrib_nz = contrib[contrib.abs() > 1e-5].sort_values()
            is_shocked = contrib_nz.index.isin(result["shocked_tickers"])
            colors = [
                "#DC2626" if shocked else (COLORS["long"] if v >= 0 else "#94A3B8")
                for v, shocked in zip(contrib_nz.values, is_shocked)
            ]
            fig_contrib = go.Figure(go.Bar(
                x=contrib_nz.values * 100,
                y=contrib_nz.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f}%" for v in contrib_nz.values * 100],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<extra></extra>",
            ))
            fig_contrib.add_vline(x=0, line_color="white", line_width=1)
            fig_contrib.update_layout(
                title=f"Contribution par titre — choc {target_sector} {shock_pct}% sur {n_months_shock}m",
                xaxis_title="Contribution (%)",
                height=max(350, len(contrib_nz) * 24),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=80, r=80),
            )
            st.plotly_chart(fig_contrib, width="stretch")

        with col_right:
            explain(
                "Vue agrégée par secteur : exposition totale (poids cumulé) et contribution "
                "au rendement stressé. Le secteur choqué apparaît en rouge."
            )
            sec_df = result["sector_summary"].copy()
            sec_df = sec_df[sec_df["Poids (%)"] > 0.1].sort_values("Poids (%)", ascending=False)
            bar_colors = [
                "#DC2626" if choqué else COLORS["neutral"]
                for choqué in sec_df["Choqué"]
            ]
            fig_sec = go.Figure()
            fig_sec.add_trace(go.Bar(
                name="Poids (%)",
                x=sec_df.index,
                y=sec_df["Poids (%)"],
                marker_color=bar_colors,
            ))
            fig_sec.update_layout(
                title="Exposition sectorielle du portefeuille",
                height=320, yaxis_title="%",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_sec, width="stretch")

            # Titres exposés dans le secteur choqué
            if result["shocked_tickers"]:
                shocked_in_port = [t for t in result["shocked_tickers"] if last_w.get(t, 0) > 0.001]
                if shocked_in_port:
                    st.markdown(f"**Positions exposées à {target_sector} :**")
                    exp_df = pd.DataFrame({
                        "Poids": [f"{last_w[t]:.1%}" for t in shocked_in_port],
                        "Impact choc": [
                            f"{last_w[t] * (shock_pct/100/n_months_shock) * n_months_shock:.2%}"
                            for t in shocked_in_port
                        ],
                    }, index=shocked_in_port)
                    st.dataframe(exp_df, width="stretch")

    # -----------------------------------------------------------------------
    # TAB 2 : Crises historiques
    # -----------------------------------------------------------------------
    with tab2:
        explain(
            "Replay des périodes de stress présentes dans la fenêtre du backtest. "
            "Pour chaque crise, on mesure la performance effective du portefeuille tel qu'il "
            "était positionné à l'époque — sans rétroviseur. C'est la mesure la plus honnête "
            "de la résilience du modèle car elle utilise les vrais poids historiques."
        )

        monthly_ret_aligned = monthly_ret.reindex(columns=weights_df.columns)
        crises = historical_crises(
            portfolio_returns   = r,
            benchmark_returns   = bm,
            monthly_ret_history = monthly_ret_aligned,
            weights_history     = weights_df,
            sector_map          = sector_map,
        )

        if not crises:
            st.info("Aucune crise historique dans la période du backtest.")
        else:
            # Tableau récapitulatif
            summary_rows = []
            for c in crises:
                summary_rows.append({
                    "Crise":             c["label"],
                    "Période":           f"{c['start'].strftime('%b %Y')} → {c['end'].strftime('%b %Y')}",
                    "Durée":             f"{c['n_months']}m",
                    "Portefeuille":      f"{c['port_total']:+.1%}",
                    "Benchmark":         f"{c['bm_total']:+.1%}",
                    "Alpha actif":       f"{c['active_ret']:+.1%}",
                    "Max DD période":    f"{c['max_dd']:.1%}",
                })
            summary_df = pd.DataFrame(summary_rows).set_index("Crise")

            def color_cell(val):
                if isinstance(val, str) and val.startswith("+"):
                    return "color: #10B981"
                if isinstance(val, str) and val.startswith("-"):
                    return "color: #EF4444"
                return ""

            st.dataframe(
                summary_df.style.map(color_cell,
                    subset=["Portefeuille", "Benchmark", "Alpha actif", "Max DD période"]),
                width="stretch",
            )

            # Graphe détaillé par crise sélectionnée
            st.divider()
            selected_crisis_label = st.selectbox(
                "Détail d'une crise", [c["label"] for c in crises]
            )
            selected = next(c for c in crises if c["label"] == selected_crisis_label)

            st.markdown(f"*{selected['description']}*")

            col_d1, col_d2 = st.columns(2)

            with col_d1:
                explain(
                    "Rendements mensuels du portefeuille et du benchmark mois par mois "
                    "pendant la crise. Utile pour voir si la protection se manifeste dès le "
                    "premier mois ou seulement en milieu de crise."
                )
                monthly_port = selected["monthly_rets"]
                monthly_bm_c = bm.reindex(monthly_port.index).fillna(0)
                labels = [t.strftime("%b %Y") for t in monthly_port.index]

                fig_monthly = go.Figure()
                fig_monthly.add_trace(go.Bar(
                    name="Benchmark", x=labels, y=monthly_bm_c.values * 100,
                    marker_color=COLORS["benchmark"], opacity=0.7,
                ))
                fig_monthly.add_trace(go.Bar(
                    name="Portefeuille", x=labels,
                    y=monthly_port.values * 100,
                    marker_color=[COLORS["long"] if v >= 0 else COLORS["short"]
                                  for v in monthly_port.values],
                    text=[f"{v:+.1f}%" for v in monthly_port.values * 100],
                    textposition="outside",
                ))
                fig_monthly.update_layout(
                    barmode="group", height=320, yaxis_title="%",
                    title=f"Rendements mensuels — {selected_crisis_label}",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_monthly, width="stretch")

            with col_d2:
                if selected["sector_attr"]:
                    explain(
                        "Attribution sectorielle pendant la crise : quelle fraction de la "
                        "performance vient de chaque secteur ? Calculé comme poids moyen "
                        "du secteur × rendement moyen du secteur pendant la période."
                    )
                    sa = pd.Series(selected["sector_attr"]).sort_values()
                    fig_sa = go.Figure(go.Bar(
                        x=sa.values * 100,
                        y=sa.index,
                        orientation="h",
                        marker_color=[COLORS["long"] if v >= 0 else COLORS["short"]
                                      for v in sa.values],
                        text=[f"{v:+.2f}%" for v in sa.values * 100],
                        textposition="outside",
                        textfont=dict(size=10),
                    ))
                    fig_sa.add_vline(x=0, line_color="white", line_width=1)
                    fig_sa.update_layout(
                        title="Attribution sectorielle",
                        height=320,
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(r=80),
                    )
                    st.plotly_chart(fig_sa, width="stretch")

    # -----------------------------------------------------------------------
    # TAB 3 : Knockout de facteur
    # -----------------------------------------------------------------------
    with tab3:
        explain(
            "Que se passe-t-il si un signal du modèle cesse de fonctionner ? "
            "On recalcule les alpha scores en mettant le coefficient Ridge du facteur exclu à 0, "
            "puis on observe quels titres changeraient de rang dans le portefeuille. "
            "C'est une mesure directe de la dépendance du portefeuille à chaque signal."
        )

        factors = fw_last.index.tolist()
        col_ko1, col_ko2 = st.columns([2, 1])
        with col_ko1:
            excluded = st.selectbox("Facteur à supprimer", factors)
        with col_ko2:
            st.metric(
                f"Poids Ridge actuel ({excluded})",
                f"{fw_last.get(excluded, 0):.4f}",
                help="Coefficient moyen Ridge du dernier mois de backtest."
            )

        # Récupérer les signaux et scores du dernier mois
        from dashboard.data import load_signals
        signals_data = load_signals()

        last_date  = weights_df.index[-1]
        signals_at = {}
        for name, df in signals_data.items():
            if last_date in df.index:
                signals_at[name] = df.loc[last_date]
        sig_df = pd.DataFrame(signals_at).reindex(last_w.index).dropna()

        if sig_df.empty:
            st.warning("Signaux non disponibles pour la date du dernier rebalancement.")
        else:
            from sklearn.preprocessing import StandardScaler
            common_f = sig_df.columns.intersection(fw_last.index)
            X = pd.DataFrame(
                StandardScaler().fit_transform(sig_df[common_f]),
                index=sig_df.index, columns=common_f,
            )
            alpha_full = X @ fw_last[common_f]

            ko = factor_knockout(
                alpha_scores_full = alpha_full,
                factor_weights    = fw_last,
                current_signals   = sig_df[common_f],
                excluded_factor   = excluded,
                weights_full      = last_w,
                sector_map        = sector_map,
            )

            col_a, col_b = st.columns(2)

            with col_a:
                explain(
                    "Importance relative de chaque facteur dans la formation des alpha scores "
                    "au dernier rebalancement. Mesurée comme la contribution moyenne en valeur absolue "
                    "à travers tous les titres de l'univers. Un facteur dominant crée une fragilité "
                    "si son signal se dégrade."
                )
                fc = ko["factor_contribution"].sort_values(ascending=True)
                bar_colors_ko = [
                    "#DC2626" if f == excluded else COLORS["strategy"]
                    for f in fc.index
                ]
                fig_fc = go.Figure(go.Bar(
                    x=fc.values,
                    y=fc.index,
                    orientation="h",
                    marker_color=bar_colors_ko,
                    text=[f"{v:.4f}" for v in fc.values],
                    textposition="outside",
                ))
                fig_fc.update_layout(
                    title="Importance des facteurs (contribution aux scores)",
                    height=320,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(r=80),
                )
                st.plotly_chart(fig_fc, width="stretch")

            with col_b:
                explain(
                    "Changement de rang pour les titres actuellement en portefeuille. "
                    "Un delta positif signifie que le titre monterait dans le classement "
                    "sans ce facteur (il est pénalisé par lui). Négatif = il descend "
                    "(il bénéficiait de ce facteur)."
                )
                in_port    = ko["in_portfolio"]
                rank_df    = ko["rank_changes"]
                rank_port  = rank_df[rank_df["En portefeuille"]].sort_values("Delta rang")

                delta = rank_port["Delta rang"]
                fig_delta = go.Figure(go.Bar(
                    x=delta.values,
                    y=delta.index,
                    orientation="h",
                    marker_color=[COLORS["long"] if v > 0 else COLORS["short"]
                                  for v in delta.values],
                    text=[f"{int(v):+d}" for v in delta.values],
                    textposition="outside",
                ))
                fig_delta.add_vline(x=0, line_color="white", line_width=1)
                fig_delta.update_layout(
                    title=f"Δ rang des positions (sans '{excluded}')",
                    height=320,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(r=60),
                )
                st.plotly_chart(fig_delta, width="stretch")

            # Titres entrants / sortants du portefeuille
            top_orig = set(alpha_full.nlargest(len(in_port)).index)
            ko_w     = fw_last.copy()
            if excluded in ko_w.index:
                ko_w[excluded] = 0.0
            alpha_ko   = X @ ko_w[common_f]
            top_ko     = set(alpha_ko.nlargest(len(in_port)).index)
            entering   = top_ko - top_orig
            exiting    = top_orig - top_ko

            if entering or exiting:
                st.divider()
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    if exiting:
                        st.markdown(f"**Sortiraient du portefeuille sans `{excluded}` :**")
                        st.markdown(" · ".join(f"`{t}`" for t in sorted(exiting)))
                with col_e2:
                    if entering:
                        st.markdown(f"**Entreraient dans le portefeuille sans `{excluded}` :**")
                        st.markdown(" · ".join(f"`{t}`" for t in sorted(entering)))
