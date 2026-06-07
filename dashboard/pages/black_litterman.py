import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from sklearn.covariance import LedoitWolf

from dashboard.components import COLORS, explain


def render(raw_returns: pd.DataFrame, picks: pd.DataFrame, bt: dict | None) -> None:
    st.title("Black-Litterman & Frontière Efficiente")

    explain(
        "Black-Litterman (He & Litterman, 1992) résout un problème classique de l'optimisation "
        "moyenne-variance : les poids sont très sensibles aux estimations de rendement espéré, "
        "et de petites erreurs produisent des portefeuilles concentrés et peu robustes. "
        "BL part du principe que le marché est en équilibre — les rendements implicites "
        "<b>pi = lambda * Sigma * w_mkt</b> sont un prior raisonnable — puis incorpore nos vues "
        "quantitatives (les alpha scores Ridge) de façon bayésienne. Le résultat est un "
        "vecteur de rendements espérés plus stable, qui mélange l'information du marché et "
        "celle du modèle selon un paramètre de confiance."
    )

    if not picks.empty:
        long_picks = picks[picks["side"] == "long"]["ticker"].unique()[:8].tolist()
    else:
        long_picks = ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "JPM"]

    available = [t for t in long_picks if t in raw_returns.columns]
    if len(available) < 4:
        st.warning("Pas assez de tickers disponibles pour la frontière.")
        return

    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        tickers_sel = st.multiselect(
            "Tickers inclus dans la frontière efficiente",
            options=[t for t in raw_returns.columns if t in available] + list(raw_returns.columns[:30]),
            default=available,
            max_selections=12,
        )
    with col_ctrl2:
        confidence = st.slider(
            "Confiance dans les vues (0 = prior marché pur, 1 = vues pures)",
            0.0, 1.0, 0.5, 0.05,
        )

    if len(tickers_sel) < 3:
        st.info("Sélectionne au moins 3 tickers.")
        return

    explain(
        f"Confiance = {confidence:.0%} : le posterior BL donne "
        f"{'plus de poids aux vues quantitatives' if confidence > 0.5 else 'plus de poids au prior marché' if confidence < 0.5 else 'un poids égal au prior et aux vues'}. "
        "Augmenter la confiance rapproche les rendements espérés des alpha scores Ridge ; "
        "la diminuer ramène vers l'équilibre de marché CAPM."
    )

    hist    = (1 + raw_returns[tickers_sel]).resample("MS").prod() - 1
    hist    = hist.dropna().iloc[-48:]
    lw      = LedoitWolf()
    lw.fit(hist.values)
    cov     = lw.covariance_
    mu_hist = hist.mean().values * 12

    mcap_path = Path("data/raw/market_caps.json")
    if mcap_path.exists():
        mcaps_all = json.loads(mcap_path.read_text())
        mcap_vals = np.array([float(mcaps_all.get(t, 1e9)) for t in tickers_sel])
        mcap_vals = np.where(mcap_vals == 0, mcap_vals.mean(), mcap_vals)
        w_mkt = mcap_vals / mcap_vals.sum()
    else:
        w_mkt = np.ones(len(tickers_sel)) / len(tickers_sel)

    lam  = 2.5
    pi   = lam * cov @ w_mkt * 12
    views = mu_hist.copy()
    if views.std() > 0:
        views = views / views.std() * pi.std()

    tau             = 0.05
    tau_cov         = tau * cov
    confidence_clip = max(min(confidence, 0.9999), 0.001)
    omega_diag      = np.diag(tau_cov) * (1.0 / confidence_clip - 1.0)
    omega_inv       = np.diag(1.0 / omega_diag)
    tau_cov_inv     = np.linalg.inv(tau_cov)
    M               = np.linalg.inv(tau_cov_inv + omega_inv)
    mu_bl           = M @ (tau_cov_inv @ pi + omega_inv @ views)

    try:
        import cvxpy as cp

        n = len(tickers_sel)
        target_returns     = np.linspace(mu_bl.min() * 0.8, mu_bl.max() * 1.2, 50)
        frontier_vols, frontier_rets = [], []

        for target in target_returns:
            w    = cp.Variable(n)
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(w, cov * 12)),
                [cp.sum(w) == 1, w >= 0, mu_bl @ w == target],
            )
            prob.solve(solver=cp.CLARABEL, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                frontier_vols.append(np.sqrt(prob.value))
                frontier_rets.append(target)

        w_opt    = cp.Variable(n, nonneg=True)
        prob_opt = cp.Problem(
            cp.Maximize(mu_bl @ w_opt - lam / 2 * cp.quad_form(w_opt, cov * 12)),
            [cp.sum(w_opt) == 1, w_opt <= 0.40],
        )
        prob_opt.solve(solver=cp.CLARABEL, verbose=False)
        w_opt_val = w_opt.value if prob_opt.status in ("optimal", "optimal_inaccurate") else w_mkt

        ret_opt = float(mu_bl @ w_opt_val)
        vol_opt = float(np.sqrt(w_opt_val @ (cov * 12) @ w_opt_val))
        ret_mkt = float(mu_bl @ w_mkt)
        vol_mkt = float(np.sqrt(w_mkt @ (cov * 12) @ w_mkt))
        ret_eq  = float(mu_bl @ (np.ones(n) / n))
        vol_eq  = float(np.sqrt((np.ones(n) / n) @ (cov * 12) @ (np.ones(n) / n)))

        np.random.seed(42)
        mc_vols, mc_rets, mc_sharpes = [], [], []
        for _ in range(3000):
            w_r = np.random.dirichlet(np.ones(n))
            r_r = float(mu_bl @ w_r)
            v_r = float(np.sqrt(w_r @ (cov * 12) @ w_r))
            mc_rets.append(r_r)
            mc_vols.append(v_r)
            mc_sharpes.append(r_r / v_r if v_r > 0 else 0)

        explain(
            "Le nuage de points représente 3 000 portefeuilles construits avec des poids "
            "aléatoires (tirage Dirichlet). La couleur indique le ratio de Sharpe. La courbe "
            "verte est la frontière efficiente : pour chaque niveau de volatilité cible, elle "
            "donne le maximum de rendement atteignable par programmation quadratique. Tous les "
            "portefeuilles optimaux se trouvent sur cette courbe — tout ce qui est en dessous "
            "est sous-optimal car on peut améliorer le rendement sans ajouter de risque."
        )

        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=[v * 100 for v in mc_vols], y=[r * 100 for r in mc_rets],
            mode="markers",
            marker=dict(color=mc_sharpes, colorscale="Viridis", size=4, opacity=0.5,
                        colorbar=dict(title="Sharpe", x=1.02)),
            name="Portefeuilles aléatoires",
            hovertemplate="Vol: %{x:.1f}%<br>Ret: %{y:.1f}%",
        ))
        if frontier_vols:
            fig_ef.add_trace(go.Scatter(
                x=[v * 100 for v in frontier_vols], y=[r * 100 for r in frontier_rets],
                mode="lines", line=dict(color=COLORS["strategy"], width=3),
                name="Frontière efficiente (BL)",
            ))
        for label, ret, vol, color, size, sym in [
            ("Optimal BL",   ret_opt, vol_opt, COLORS["strategy"],  18, "star"),
            ("Market Cap",   ret_mkt, vol_mkt, COLORS["accent"],    14, "diamond"),
            ("Equal Weight", ret_eq,  vol_eq,  COLORS["benchmark"], 12, "circle"),
        ]:
            fig_ef.add_trace(go.Scatter(
                x=[vol * 100], y=[ret * 100],
                mode="markers+text",
                marker=dict(color=color, size=size, symbol=sym,
                            line=dict(color="white", width=1.5)),
                text=[label], textposition="top center", name=label,
            ))
        for i, t in enumerate(tickers_sel):
            fig_ef.add_trace(go.Scatter(
                x=[np.sqrt(cov[i, i] * 12) * 100], y=[float(mu_bl[i]) * 100],
                mode="markers+text",
                marker=dict(color="white", size=8, opacity=0.7),
                text=[t], textposition="top right", textfont=dict(size=10),
                name=t, showlegend=False,
            ))
        fig_ef.update_layout(
            title=f"Frontière efficiente Black-Litterman (confiance vues = {confidence:.0%})",
            xaxis_title="Volatilité annualisée (%)",
            yaxis_title="Rendement espéré annualisé (%)",
            height=580,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_ef, width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Poids optimaux BL")
            explain(
                "L'optimiseur maximise <b>mu_BL @ w - lambda/2 * w' Sigma w</b> sous contrainte "
                "de somme des poids = 1 et poids max 40 % par titre. Le paramètre lambda = 2.5 "
                "représente l'aversion au risque : augmenter lambda réduit la concentration."
            )
            w_df = pd.DataFrame({
                "Ticker":      tickers_sel,
                "Poids BL":    [f"{v:.1%}" for v in w_opt_val],
                "Market Cap":  [f"{v:.1%}" for v in w_mkt],
                "Ret. espéré": [f"{v:.1%}" for v in mu_bl],
            })
            st.dataframe(w_df.set_index("Ticker"), width="stretch")

        with col2:
            st.subheader("Prior vs vues vs posterior")
            explain(
                "Ce graphe illustre le coeur de BL : le prior (rendements d'équilibre CAPM) "
                "est tiré vers les vues quantitatives (alpha scores) pour donner le posterior. "
                "Un curseur de confiance élevé rapproche le posterior des vues ; bas, il reste "
                "proche du prior. La matrice Omega encode cette incertitude sur les vues."
            )
            blend_df = pd.DataFrame({
                "Prior pi":  pi * 100,
                "Vues":      views * 100,
                "Posterior": mu_bl * 100,
            }, index=tickers_sel)
            fig_blend = go.Figure()
            for col_name, color in [
                ("Prior pi",  COLORS["benchmark"]),
                ("Vues",      COLORS["accent"]),
                ("Posterior", COLORS["strategy"]),
            ]:
                fig_blend.add_trace(go.Bar(
                    name=col_name, x=blend_df.index, y=blend_df[col_name],
                    marker_color=color,
                ))
            fig_blend.update_layout(
                barmode="group", height=320, yaxis_title="Rendement espéré (%)",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_blend, width="stretch")

    except Exception as e:
        st.error(f"Erreur dans le calcul de la frontière : {e}")
        st.exception(e)
