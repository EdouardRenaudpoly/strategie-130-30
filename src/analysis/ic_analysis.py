"""
Information Coefficient (IC) analysis pour chaque signal.

IC mensuel = corrélation de Spearman cross-sectionelle entre
             signal(t) et rendement_forward(t+1).

Métriques produites par signal :
  mean_IC    : IC moyen (>0.02 = utile, >0.05 = très bon)
  ICIR       : mean(IC) / std(IC) * sqrt(12)  — consistance
  hit_rate   : % de mois où IC > 0
  t_stat     : significativité statistique
  n_months   : nombre d'observations valides
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats


def compute_ic_series(
    signal: pd.DataFrame,
    forward_returns: pd.DataFrame,
    universe_snapshots=None,
    min_obs: int = 30,
) -> pd.Series:
    """
    Calcule l'IC mensuel (Spearman) pour un signal donné.

    signal          : DataFrame (dates x tickers), fréquence mensuelle
    forward_returns : DataFrame (dates x tickers), rendements mensuels
    universe_snapshots : optionnel — restreint au bon univers à chaque date
    min_obs         : nb minimum de tickers communs pour calculer l'IC
    """
    if universe_snapshots is not None:
        from src.universe.historical_constituents import get_universe_at_date

    ic_values = {}
    dates = signal.index.intersection(forward_returns.index[:-1])

    for t in dates:
        # Rendement forward = mois suivant
        fwd_idx = forward_returns.index.searchsorted(t)
        if fwd_idx + 1 >= len(forward_returns):
            continue
        t_fwd = forward_returns.index[fwd_idx + 1]

        sig_row = signal.loc[t]
        ret_row = forward_returns.loc[t_fwd]

        # Restreindre à l'univers historique si disponible
        if universe_snapshots is not None:
            valid = get_universe_at_date(universe_snapshots, t)
            valid = [tk for tk in valid if tk in sig_row.index and tk in ret_row.index]
            sig_row = sig_row[valid]
            ret_row = ret_row[valid]

        # Aligner et dropna
        combined = pd.DataFrame({"sig": sig_row, "ret": ret_row}).dropna()
        if len(combined) < min_obs:
            continue

        ic, _ = stats.spearmanr(combined["sig"], combined["ret"])
        ic_values[t] = ic

    return pd.Series(ic_values, name="IC")


def ic_summary(ic_series: pd.Series) -> dict:
    """Calcule les métriques résumées d'une série d'IC."""
    n = len(ic_series.dropna())
    if n == 0:
        return {"mean_IC": np.nan, "ICIR": np.nan, "hit_rate": np.nan,
                "t_stat": np.nan, "n_months": 0}

    mean_ic = ic_series.mean()
    std_ic  = ic_series.std()
    icir    = (mean_ic / std_ic * np.sqrt(12)) if std_ic > 0 else np.nan
    hit     = (ic_series > 0).mean()
    t_stat  = (mean_ic / (std_ic / np.sqrt(n))) if std_ic > 0 else np.nan

    return {
        "mean_IC":   round(mean_ic, 4),
        "ICIR":      round(icir, 3),
        "hit_rate":  round(hit, 3),
        "t_stat":    round(t_stat, 2),
        "n_months":  n,
    }


def run_ic_analysis(
    signals: dict[str, pd.DataFrame],
    monthly_returns: pd.DataFrame,
    universe_snapshots=None,
    output_dir: str = "data/processed/analysis",
) -> pd.DataFrame:
    """
    Lance l'analyse IC sur tous les signaux.
    Retourne un DataFrame résumé et sauvegarde les séries IC individuelles.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    ic_series_all = {}

    for name, df in signals.items():
        print(f"  IC {name}...", end=" ", flush=True)

        # Les signaux fondamentaux sont mensuels (MS), les prix sont quotidiens
        # → rééchantillonner le signal sur le calendrier mensuel des rendements
        if not df.index.equals(monthly_returns.index):
            df = df.reindex(monthly_returns.index, method="ffill")

        ic_s = compute_ic_series(df, monthly_returns, universe_snapshots)
        ic_series_all[name] = ic_s
        results[name] = ic_summary(ic_s)

        print(f"mean={results[name]['mean_IC']:+.4f}  ICIR={results[name]['ICIR']:+.3f}  "
              f"hit={results[name]['hit_rate']:.0%}  t={results[name]['t_stat']:+.2f}  "
              f"n={results[name]['n_months']}")

    # Sauvegarde des séries IC
    ic_df = pd.DataFrame(ic_series_all)
    ic_df.to_csv(out / "ic_series.csv")

    # Résumé
    summary = pd.DataFrame(results).T
    summary.to_csv(out / "ic_summary.csv")

    return summary
