import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.universe.historical_constituents import load_snapshots
from src.analysis.sector_analysis import load_sector_map

args = sys.argv[1:]
sector_neutral = "--no-sector-neutral" not in args  # activé par défaut
print(f"Mode : neutralisation sectorielle {'ON' if sector_neutral else 'OFF'}")

# --- Constituants historiques ---
print("Chargement des constituants historiques S&P 500...")
universe_snapshots = load_snapshots(min_date="2013-01-01")
print(f"  {len(universe_snapshots)} snapshots | {universe_snapshots.iloc[0]['date'].date()} → {universe_snapshots.iloc[-1]['date'].date()}")

# --- Secteurs GICS (cache local) ---
print("Chargement des secteurs GICS...")
_tickers_for_sectors = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, nrows=0).columns.tolist()
sector_map = load_sector_map(_tickers_for_sectors)
print(f"  {sum(1 for s in sector_map.values() if s != 'Unknown')}/{len(sector_map)} tickers avec secteur")

# --- Chargement des données ---
print("Chargement des données...")
returns = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, parse_dates=True)
volume = pd.read_csv("data/processed/sp500_volume.csv", index_col=0, parse_dates=True)

def _load_signal(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    return None

signals = {}
for name, path in [
    ("momentum",       "data/processed/signals/momentum.csv"),
    ("low_volatility", "data/processed/signals/low_volatility.csv"),
    ("reversal",       "data/processed/signals/reversal.csv"),
    ("illiquidity",    "data/processed/signals/illiquidity.csv"),
    ("high_52w",            "data/processed/signals/high_52w.csv"),
    ("insider",             "data/processed/signals/insider_trading.csv"),
    ("gross_profitability", "data/processed/signals/gross_profitability.csv"),
    ("accruals",            "data/processed/signals/accruals.csv"),
    ("asset_growth",        "data/processed/signals/asset_growth.csv"),
]:
    df = _load_signal(path)
    if df is not None:
        signals[name] = df
        print(f"  Signal '{name}' chargé : {df.shape}")
    else:
        print(f"  Signal '{name}' absent (pas encore calculé)")

# Aligner tous les DataFrames sur le même univers de tickers
tickers = returns.columns.tolist()
volume = volume.reindex(columns=tickers)
signals = {name: df.reindex(columns=tickers) for name, df in signals.items()}

# Benchmark : equal weight
benchmark_weights = pd.Series(1 / len(tickers), index=tickers)

# --- Config ---
config = BacktestConfig(
    estimation_window=36,
    ridge_alpha=1.0,
    max_position=0.05,
    max_short=0.05,
    max_te=0.06,
    max_turnover=0.30,
    transaction_cost_bps=7,
    train_end="2020-01-01",
    sector_neutral=sector_neutral,
)

# --- Run ---
print("Lancement du backtest walk-forward...")
engine = BacktestEngine(
    returns=returns,
    signals=signals,
    volume=volume,
    benchmark_weights=benchmark_weights,
    config=config,
    universe_snapshots=universe_snapshots,
    sector_map=sector_map,
)

results = engine.run()

# --- Résultats ---
r = results.returns
monthly_returns = (1 + returns).resample("MS").prod() - 1
benchmark_returns = monthly_returns.mean(axis=1).reindex(r.index)

excess = r - benchmark_returns.values
sharpe = r.mean() / r.std() * np.sqrt(12)
ir = excess.mean() / excess.std() * np.sqrt(12)
max_dd = (r.cumsum() - r.cumsum().cummax()).min()
avg_turnover = results.turnover.mean()
avg_cost = results.costs.mean()

print("\n--- Résultats ---")
print(f"Période          : {r.index[0].date()} → {r.index[-1].date()}")
print(f"Nb rebalancements: {len(r)}")
print(f"Sharpe           : {sharpe:.3f}")
print(f"Info Ratio       : {ir:.3f}")
print(f"Max Drawdown     : {max_dd:.2%}")
print(f"Rendement annuel : {r.mean() * 12:.2%}")
print(f"Turnover moyen   : {avg_turnover:.2%}")
print(f"Coût moyen/mois  : {avg_cost:.4%}")
print(f"\nFacteur weights (moyenne) :")
print(results.factor_weights.mean().round(4))

# --- Diagnostic poids ---
w = results.weights
print(f"\n--- Diagnostic poids ---")
print(f"Nb tickers long  (w > 0) en moyenne : {(w > 1e-4).sum(axis=1).mean():.1f}")
print(f"Nb tickers short (w < 0) en moyenne : {(w < -1e-4).sum(axis=1).mean():.1f}")
print(f"Somme longs  (moyenne)  : {w.clip(lower=0).sum(axis=1).mean():.3f}")
print(f"Somme shorts (moyenne)  : {w.clip(upper=0).sum(axis=1).mean():.3f}")

# Fréquence d'apparition dans le portefeuille long / short
w = results.weights
long_freq  = (w > 1e-4).mean().sort_values(ascending=False)
short_freq = (w < -1e-4).mean().sort_values(ascending=False)
print(f"\nTop 10 tickers les plus souvent longs (% des mois) :")
print(long_freq.head(10).mul(100).round(1).to_string())
print(f"\nTop 10 tickers les plus souvent shorts (% des mois) :")
print(short_freq.head(10).mul(100).round(1).to_string())

# Distribution des rendements mensuels
print(f"\n--- Distribution rendements mensuels ---")
print(f"Médiane : {r.median():.2%}")
print(f"% mois positifs : {(r > 0).mean():.1%}")
print(r.describe().round(4))
