import pandas as pd
import numpy as np
from pathlib import Path

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.universe.historical_constituents import load_snapshots
from src.analysis.sector_analysis import load_sector_map, sector_exposure_summary, compute_sector_exposures

# --- Données ---
print("Chargement des données...")
returns = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, parse_dates=True)
volume  = pd.read_csv("data/processed/sp500_volume.csv",  index_col=0, parse_dates=True)
universe_snapshots = load_snapshots(min_date="2013-01-01")

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
    ("gross_profitability", "data/processed/signals/gross_profitability.csv"),
    ("accruals",            "data/processed/signals/accruals.csv"),
    ("asset_growth",        "data/processed/signals/asset_growth.csv"),
]:
    df = _load(path)
    if df is not None:
        signals[name] = df

tickers = returns.columns.tolist()
volume  = volume.reindex(columns=tickers)
signals = {n: df.reindex(columns=tickers) for n, df in signals.items()}
benchmark_weights = pd.Series(1 / len(tickers), index=tickers)

# --- Backtest ---
print("Backtest walk-forward...")
config = BacktestConfig(
    estimation_window=36, ridge_alpha=1.0,
    max_position=0.05, max_short=0.05, max_te=0.06,
    max_turnover=0.30, transaction_cost_bps=7, train_end="2020-01-01",
)
engine = BacktestEngine(returns, signals, volume, benchmark_weights, config, universe_snapshots)
results = engine.run()
weights = results.weights

# --- Secteurs ---
print("\nFetch des secteurs GICS...")
sector_map = load_sector_map(tickers)

known = {t: s for t, s in sector_map.items() if s != "Unknown"}
print(f"  {len(known)}/{len(tickers)} tickers avec secteur connu")

# --- Résumé exposition ---
print("\nCalcul des expositions sectorielles...")
summary = sector_exposure_summary(weights, sector_map, benchmark_weights)

print("\n" + "="*72)
print(f"{'Secteur':<30} {'Portfolio':>9} {'Benchmark':>9} {'Actif moy':>9} {'Max|actif|':>10}")
print("-"*72)
for sector, row in summary.iterrows():
    if sector == "Unknown":
        continue
    bar = "▲" if row["active_avg"] > 0.01 else ("▼" if row["active_avg"] < -0.01 else " ")
    print(f"{bar} {sector:<28} {row['portfolio_avg']:>+9.3f} {row['benchmark']:>9.3f} "
          f"{row['active_avg']:>+9.3f} {row['active_max']:>10.3f}")
print("="*72)

# --- Expositions long / short séparées ---
print("\n--- Expositions LONGUES par secteur (moyenne) ---")
w_long  = weights.clip(lower=0)
w_short = weights.clip(upper=0)

exp_long  = compute_sector_exposures(w_long,  sector_map)
exp_short = compute_sector_exposures(w_short, sector_map)

long_avg  = exp_long.mean().sort_values(ascending=False)
short_avg = exp_short.mean().sort_values(ascending=False)

for sector in long_avg.index:
    if sector == "Unknown":
        continue
    print(f"  {sector:<30}  long={long_avg[sector]:+.3f}  short={short_avg[sector]:+.3f}  "
          f"net={long_avg[sector]+short_avg[sector]:+.3f}")

# --- Concentration : check si un secteur domine ---
print("\n--- Alertes de concentration ---")
active = summary["active_avg"].drop("Unknown", errors="ignore")
overweight  = active[active >  0.05]
underweight = active[active < -0.05]

if overweight.empty and underweight.empty:
    print("  Aucune exposition active > 5% — portefeuille bien diversifié")
else:
    for s, v in overweight.items():
        print(f"  ⚠ SUREXPOSÉ   : {s:30s} actif moyen = {v:+.1%}")
    for s, v in underweight.items():
        print(f"  ⚠ SOUS-EXPOSÉ : {s:30s} actif moyen = {v:+.1%}")

# Sauvegarde
exposures = compute_sector_exposures(weights, sector_map)
exposures.to_csv("data/processed/analysis/sector_exposures.csv")
summary.to_csv("data/processed/analysis/sector_summary.csv")
print("\nSauvegardé : data/processed/analysis/sector_exposures.csv")
