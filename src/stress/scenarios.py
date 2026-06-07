"""
Stress tests quantitatifs pour la stratégie 130/30.

Trois familles :
  1. sector_shock       — choc sur un secteur donné (amplitude + durée configurables)
  2. historical_crises  — replay des crises historiques dans la période du backtest
  3. factor_knockout    — sensibilité du portefeuille à la suppression d'un signal
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Choc sectoriel
# ---------------------------------------------------------------------------

def sector_shock(
    weights: pd.Series,
    sector_map: dict[str, str],
    monthly_ret_history: pd.DataFrame,
    target_sector: str,
    shock_total: float,
    n_months: int = 1,
) -> dict:
    """
    Simule un choc résiduel sur un secteur sur n_months mois consécutifs.

    La logique est un choc résiduel : le secteur choqué fait shock_total de moins
    que son historique. Le reste du portefeuille performe à son rendement moyen
    historique. Cela isole l'impact du choc sectoriel en neutralisant le bruit.

    Retourne un dict avec :
      - monthly_portfolio_returns  : Series(n_months)
      - cumulative_return          : float
      - position_contributions     : Series (par ticker, cumulé)
      - sector_summary             : DataFrame (par secteur)
      - baseline_return            : float (sans choc)
    """
    sectors = pd.Series(sector_map)
    shocked_tickers = sectors[sectors == target_sector].index.intersection(weights.index)

    # Rendement mensuel moyen historique par ticker (base de référence)
    mean_ret = monthly_ret_history.reindex(columns=weights.index).mean()

    # Choc résiduel mensuel uniformément réparti
    shock_monthly = shock_total / n_months

    cumulative_port = 0.0
    cumulative_base = 0.0
    monthly_results = []
    contrib_total   = pd.Series(0.0, index=weights.index)

    for month in range(n_months):
        # Rendement de chaque ticker ce mois-ci
        ret = mean_ret.copy()
        ret[shocked_tickers] += shock_monthly   # choc résiduel en plus du mean

        # Rendement pondéré par les poids
        contrib = weights * ret
        port_ret = contrib.sum()
        base_ret = (weights * mean_ret).sum()

        cumulative_port += port_ret
        cumulative_base += base_ret
        monthly_results.append(port_ret)
        contrib_total   += contrib

    # Résumé par secteur
    sec_list = []
    for sec in sorted(set(sector_map.values())):
        tks = sectors[sectors == sec].index.intersection(weights.index)
        if tks.empty:
            continue
        w_sec    = weights[tks].sum()
        ret_sec  = (weights[tks] * mean_ret[tks]).sum() if not tks.empty else 0.0
        shock_sec = shock_monthly * len([t for t in shocked_tickers if t in tks]) / max(len(tks), 1) * w_sec if sec == target_sector else 0.0
        sec_list.append({
            "Secteur":         sec,
            "Poids (%)":       w_sec * 100,
            "Contrib. base":   ret_sec * n_months,
            "Contrib. choc":   contrib_total[tks].sum() if not tks.empty else 0.0,
            "Choqué":          sec == target_sector,
        })

    return {
        "monthly_portfolio_returns": pd.Series(monthly_results),
        "cumulative_return":         cumulative_port,
        "baseline_return":           cumulative_base,
        "impact":                    cumulative_port - cumulative_base,
        "position_contributions":    contrib_total.sort_values(),
        "sector_summary":            pd.DataFrame(sec_list).set_index("Secteur"),
        "shocked_tickers":           list(shocked_tickers),
    }


# ---------------------------------------------------------------------------
# 2. Crises historiques
# ---------------------------------------------------------------------------

CRISIS_WINDOWS = [
    {
        "label":       "COVID crash",
        "start":       "2020-01-01",
        "end":         "2020-04-01",
        "description": "Effondrement des marchés lors de la pandémie COVID-19.",
    },
    {
        "label":       "Bear market 2022",
        "start":       "2021-12-01",
        "end":         "2022-12-01",
        "description": "Correction prolongée liée au resserrement monétaire de la Fed.",
    },
    {
        "label":       "Choc de taux 2018",
        "start":       "2018-08-01",
        "end":         "2018-12-01",
        "description": "Correction de fin 2018 liée à la hausse des taux et aux tensions commerciales.",
    },
    {
        "label":       "Correction tech 2023",
        "start":       "2023-07-01",
        "end":         "2023-10-01",
        "description": "Correction du rally IA de 2023, taux réels élevés.",
    },
    {
        "label":       "Correction 2026-02",
        "start":       "2025-12-01",
        "end":         "2026-02-01",
        "description": "Correction liée aux incertitudes macro et rotation sectorielle.",
    },
]


def historical_crises(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    monthly_ret_history: pd.DataFrame,
    weights_history: pd.DataFrame,
    sector_map: dict[str, str],
) -> list[dict]:
    """
    Pour chaque crise prédéfinie présente dans la période du backtest,
    calcule les métriques de performance et l'attribution sectorielle.
    """
    results = []

    for crisis in CRISIS_WINDOWS:
        start = pd.Timestamp(crisis["start"])
        end   = pd.Timestamp(crisis["end"])

        # Filtrer sur la période du backtest
        mask_port = (portfolio_returns.index >= start) & (portfolio_returns.index <= end)
        mask_bm   = (benchmark_returns.index >= start) & (benchmark_returns.index <= end)

        r_crisis  = portfolio_returns[mask_port]
        bm_crisis = benchmark_returns[mask_bm]

        if r_crisis.empty:
            continue

        # Métriques globales
        port_total = float((1 + r_crisis).prod() - 1)
        bm_total   = float((1 + bm_crisis).prod() - 1) if not bm_crisis.empty else float("nan")
        active_ret = port_total - bm_total
        max_dd     = float((r_crisis.cumsum() - r_crisis.cumsum().cummax()).min())

        # Attribution sectorielle : poids moyen × rendement par secteur pendant la crise
        sectors   = pd.Series(sector_map)
        w_period  = weights_history[
            (weights_history.index >= start) & (weights_history.index <= end)
        ]
        w_avg = w_period.mean() if not w_period.empty else pd.Series(dtype=float)

        sector_attr = {}
        if not w_avg.empty:
            for sec in sorted(set(sector_map.values())):
                if sec == "Unknown":
                    continue
                tks = sectors[sectors == sec].index.intersection(w_avg.index).intersection(monthly_ret_history.columns)
                if tks.empty:
                    continue
                ret_sec = monthly_ret_history.loc[
                    (monthly_ret_history.index >= start) & (monthly_ret_history.index <= end), tks
                ].mean(axis=1)
                contrib = float((w_avg[tks].sum()) * ret_sec.sum())
                sector_attr[sec] = contrib

        results.append({
            "label":        crisis["label"],
            "description":  crisis["description"],
            "start":        start,
            "end":          end,
            "port_total":   port_total,
            "bm_total":     bm_total,
            "active_ret":   active_ret,
            "max_dd":       max_dd,
            "n_months":     len(r_crisis),
            "monthly_rets": r_crisis,
            "sector_attr":  sector_attr,
        })

    return results


# ---------------------------------------------------------------------------
# 3. Knockout de facteur
# ---------------------------------------------------------------------------

def factor_knockout(
    alpha_scores_full: pd.Series,
    factor_weights: pd.Series,
    current_signals: pd.DataFrame,
    excluded_factor: str,
    weights_full: pd.Series,
    sector_map: dict[str, str],
    n_positions: int | None = None,
    long_only: bool = False,
) -> dict:
    """
    Recalcule les alpha scores sans le facteur exclu.
    Compare le portefeuille résultant avec le portefeuille original.

    Retourne :
      - scores_original  : Series
      - scores_knockout  : Series
      - score_delta      : Series (changement de rang par ticker)
      - factor_contribution : Series (contribution de chaque facteur au score moyen)
    """
    from sklearn.preprocessing import StandardScaler

    # Contribution de chaque facteur au score moyen (en valeur absolue)
    if not current_signals.empty:
        X = pd.DataFrame(
            StandardScaler().fit_transform(current_signals),
            index=current_signals.index,
            columns=current_signals.columns,
        )
        common = X.columns.intersection(factor_weights.index)
        contributions = {}
        for f in common:
            contributions[f] = float((X[f] * factor_weights[f]).abs().mean())
        factor_contribution = pd.Series(contributions).sort_values(ascending=False)
    else:
        factor_contribution = pd.Series(dtype=float)

    # Scores sans le facteur exclu
    ko_weights = factor_weights.copy()
    if excluded_factor in ko_weights.index:
        ko_weights[excluded_factor] = 0.0

    common = current_signals.columns.intersection(ko_weights.index)
    if common.empty:
        return {
            "scores_original":      alpha_scores_full,
            "scores_knockout":      alpha_scores_full,
            "score_delta":          pd.Series(dtype=float),
            "factor_contribution":  factor_contribution,
            "rank_changes":         pd.DataFrame(),
        }

    X_ko = pd.DataFrame(
        StandardScaler().fit_transform(current_signals[common]),
        index=current_signals.index,
        columns=common,
    )
    scores_ko = X_ko @ ko_weights[common]

    # Changement de rang
    rank_orig = alpha_scores_full.rank(ascending=False)
    rank_ko   = scores_ko.rank(ascending=False)
    rank_delta = rank_orig - rank_ko   # positif = monte, négatif = descend

    # Pour les titres actuellement en portefeuille (top positions)
    if long_only and n_positions:
        in_portfolio = set(alpha_scores_full.nlargest(n_positions).index)
    else:
        in_portfolio = set(weights_full[weights_full.abs() > 0.01].index)

    rank_df = pd.DataFrame({
        "Rang original":  rank_orig,
        "Rang knockout":  rank_ko,
        "Delta rang":     rank_delta,
        "En portefeuille": [t in in_portfolio for t in rank_orig.index],
        "Score original": alpha_scores_full,
        "Score knockout": scores_ko,
    }).sort_values("Rang original")

    return {
        "scores_original":     alpha_scores_full,
        "scores_knockout":     scores_ko,
        "score_delta":         rank_delta,
        "factor_contribution": factor_contribution,
        "rank_changes":        rank_df,
        "in_portfolio":        in_portfolio,
    }
