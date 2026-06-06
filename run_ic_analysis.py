import numpy as np
import pandas as pd
from pathlib import Path

from src.universe.historical_constituents import load_snapshots
from src.analysis.ic_analysis import run_ic_analysis

# --- Données ---
print("Chargement des données...")
returns = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, parse_dates=True)
monthly_returns = (1 + returns).resample("MS").prod() - 1

universe_snapshots = load_snapshots(min_date="2013-01-01")

# --- Signaux ---
def _load(path):
    p = Path(path)
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None

signals = {}
for name, path in [
    ("momentum",            "data/processed/signals/momentum.csv"),
    ("low_volatility",      "data/processed/signals/low_volatility.csv"),
    ("reversal",            "data/processed/signals/reversal.csv"),
    ("illiquidity",         "data/processed/signals/illiquidity.csv"),
    ("high_52w",            "data/processed/signals/high_52w.csv"),
    ("insider",             "data/processed/signals/insider_trading.csv"),
    ("gross_profitability", "data/processed/signals/gross_profitability.csv"),
    ("accruals",            "data/processed/signals/accruals.csv"),
    ("asset_growth",        "data/processed/signals/asset_growth.csv"),
]:
    df = _load(path)
    if df is not None:
        signals[name] = df
    else:
        print(f"  '{name}' absent — ignoré")

# --- IC Analysis ---
print(f"\nAnalyse IC sur {len(signals)} signaux...")
summary = run_ic_analysis(
    signals=signals,
    monthly_returns=monthly_returns,
    universe_snapshots=universe_snapshots,
)

# --- Résumé final ---
print("\n" + "="*65)
print(f"{'Signal':<22} {'Mean IC':>8} {'ICIR':>7} {'Hit%':>6} {'t-stat':>7} {'N':>5}")
print("-"*65)
for sig, row in summary.sort_values("ICIR", ascending=False).iterrows():
    print(f"{sig:<22} {row['mean_IC']:>+8.4f} {row['ICIR']:>+7.3f} "
          f"{row['hit_rate']:>6.0%} {row['t_stat']:>+7.2f} {int(row['n_months']):>5}")
print("="*65)

# IC cumulatif (pour visualisation future)
ic_df = pd.read_csv("data/processed/analysis/ic_series.csv", index_col=0, parse_dates=True)
print(f"\nSéries IC sauvegardées : data/processed/analysis/ic_series.csv")
print(f"Résumé sauvegardé     : data/processed/analysis/ic_summary.csv")
