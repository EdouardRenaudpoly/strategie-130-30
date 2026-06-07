"""
Black-Litterman portfolio construction.
He & Litterman (1999), Idzorek (2005).

Principe :
  1. Prior : rendements d'équilibre implicites du marché
             π = λ * Σ * w_mkt   (reverse-optimization CAPM)
  2. Vues  : nos alpha scores Ridge deviennent des vues absolues
             Q = alpha_scores,  P = I (une vue par titre)
  3. Posterior BL :
             μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
  4. Optimisation mean-variance :
             max μ_BL @ w - (λ/2) * w'Σw   sous contraintes 130/30

Paramètres clés :
  τ (tau)             : incertitude sur le prior, typiquement 1/T ≈ 0.03
  λ (risk_aversion)   : aversion au risque, typiquement 2.5
  view_confidence     : 0 = ignorer les vues, 1 = ignorer le prior
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def compute_equilibrium_returns(
    cov: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """π = λ * Σ * w_mkt"""
    return risk_aversion * cov @ market_weights


def compute_bl_posterior(
    pi: np.ndarray,
    cov: np.ndarray,
    views: np.ndarray,
    tau: float = 0.05,
    view_confidence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne (μ_BL, Σ_BL) — rendements et covariance postérieurs.

    P = I (vue absolue sur chaque titre)
    Ω = diag(τ * Σ) * (1/c - 1) — He-Litterman proportional uncertainty
        view_confidence=1 → Ω→0 → on croit totalement aux vues
        view_confidence=0 → Ω→∞ → on ignore les vues, reste au prior
    """
    n = len(pi)
    P = np.eye(n)                                # une vue par titre
    tau_cov = tau * cov

    # Matrice d'incertitude des vues (He-Litterman)
    if view_confidence >= 1.0:
        view_confidence = 0.9999
    omega_diag = np.diag(tau_cov) * (1.0 / view_confidence - 1.0)
    Omega = np.diag(omega_diag)

    # Posterior mean (formule BL standard)
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv   = np.diag(1.0 / omega_diag)

    M_inv = tau_cov_inv + P.T @ omega_inv @ P
    M     = np.linalg.inv(M_inv)

    mu_bl = M @ (tau_cov_inv @ pi + P.T @ omega_inv @ views)

    # Posterior covariance
    sigma_bl = cov + M

    return mu_bl, sigma_bl


def black_litterman_weights(
    returns_hist: pd.DataFrame,
    alpha_scores: pd.Series,
    market_weights: pd.Series | None = None,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    view_confidence: float = 0.5,
    max_position: float = 0.05,
    max_short: float = 0.05,
    max_turnover: float = 0.30,
    prev_weights: pd.Series | None = None,
) -> pd.Series | None:
    """
    Pipeline BL complet → poids optimaux 130/30.

    returns_hist  : rendements mensuels historiques (T x N)
    alpha_scores  : scores Ridge pour les tickers à inclure
    market_weights: poids benchmark (equal weight si None)
    view_confidence: 0.5 = blend équilibré prior/vues
    """
    import cvxpy as cp

    tickers = alpha_scores.index.tolist()
    hist    = returns_hist[tickers].dropna()
    if len(hist) < 12:
        return None

    # Covariance Ledoit-Wolf (shrinkage — plus stable que sample cov)
    lw = LedoitWolf()
    lw.fit(hist.values)
    cov = lw.covariance_

    # Poids benchmark
    if market_weights is not None:
        w_mkt = market_weights.reindex(tickers).fillna(0).values
        w_mkt = w_mkt / w_mkt.sum()
    else:
        w_mkt = np.ones(len(tickers)) / len(tickers)

    # Prior et vues
    pi    = compute_equilibrium_returns(cov, w_mkt, risk_aversion)
    views = alpha_scores.values

    # Normaliser les vues à la même échelle que π (évite que les vues dominent)
    if views.std() > 0:
        views = views / views.std() * pi.std()

    # Posterior BL
    mu_bl, sigma_bl = compute_bl_posterior(pi, cov, views, tau, view_confidence)
    mu_bl_series = pd.Series(mu_bl, index=tickers)

    # Optimisation mean-variance avec cvxpy
    n    = len(tickers)
    prev = prev_weights.reindex(tickers).fillna(0).values if prev_weights is not None else None
    mu   = mu_bl_series.values
    long_only = (max_short == 0.0)

    if long_only:
        w = cp.Variable(n, nonneg=True)
        constraints = [
            cp.sum(w) == 1.0,
            w <= max_position,
        ]
        if prev is not None and np.abs(prev).sum() > 1e-6:
            constraints.append(cp.sum(cp.abs(w - prev)) <= max_turnover)
        objective = cp.Maximize(
            mu @ w - (risk_aversion / 2) * cp.quad_form(w, sigma_bl)
        )
    else:
        w_long  = cp.Variable(n, nonneg=True)
        w_short = cp.Variable(n, nonneg=True)
        w       = w_long - w_short
        constraints = [
            cp.sum(w_long)  == 1.30,
            cp.sum(w_short) == 0.30,
            cp.sum(w)       == 1.0,
            w_long  <= max_position,
            w_short <= max_short,
        ]
        if prev is not None and np.abs(prev).sum() > 1e-6:
            constraints.append(cp.sum(cp.abs(w - prev)) <= max_turnover)
        objective = cp.Maximize(
            mu @ w - (risk_aversion / 2) * cp.quad_form(w, sigma_bl)
        )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None

    return pd.Series(w.value, index=tickers)
