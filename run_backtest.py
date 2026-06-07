import sys
import numpy as np
import pandas as pd
from pathlib import Path

from src.backtest.engine import BacktestEngine, BacktestConfig
from src.universe.historical_constituents import load_snapshots
from src.analysis.sector_analysis import load_sector_map, load_market_cap_weights

args = sys.argv[1:]

# Configs à exécuter.  Par défaut : les 4 combinaisons sector-neutral × BL.
# Passer --single pour n'exécuter que la config demandée via les flags habituels.
single_mode    = "--single" in args
sector_neutral = "--no-sector-neutral" not in args
use_bl         = "--bl" in args
bl_confidence  = float(next((args[i+1] for i, a in enumerate(args) if a == "--bl-confidence"), 0.5))

if single_mode:
    CONFIGS = [
        dict(sector_neutral=sector_neutral, use_bl=use_bl, bl_confidence=bl_confidence),
    ]
else:
    CONFIGS = [
        dict(sector_neutral=True,  use_bl=False, bl_confidence=0.5),
        dict(sector_neutral=False, use_bl=False, bl_confidence=0.5),
        dict(sector_neutral=True,  use_bl=True,  bl_confidence=0.5),
        dict(sector_neutral=True,  use_bl=False, bl_confidence=0.5, long_only=True, n_positions=10),
        dict(sector_neutral=True,  use_bl=False, bl_confidence=0.5, long_only=True, n_positions=20, sector_cap=3, rank_buffer=1.5),
        dict(sector_neutral=True,  use_bl=False, bl_confidence=0.5, long_only=True, n_positions=20, sector_cap=3, rank_buffer=1.5, volatility_sizing=True),
        dict(sector_neutral=True,  use_bl=False, bl_confidence=0.5, use_hmm=True),
        dict(sector_neutral=True,  use_bl=False, bl_confidence=0.5, long_only=True, n_positions=20, sector_cap=3, rank_buffer=1.5, use_hmm=True),
    ]

# --- Données partagées (chargées une seule fois) ---
print("Chargement des constituants historiques S&P 500...")
universe_snapshots = load_snapshots(min_date="2013-01-01")
print(f"  {len(universe_snapshots)} snapshots | "
      f"{universe_snapshots.iloc[0]['date'].strftime('%Y-%m-%d')} → "
      f"{universe_snapshots.iloc[-1]['date'].strftime('%Y-%m-%d')}")

print("Chargement des secteurs GICS...")
_tickers_for_sectors = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, nrows=0).columns.tolist()
sector_map = load_sector_map(_tickers_for_sectors)
print(f"  {sum(1 for s in sector_map.values() if s != 'Unknown')}/{len(sector_map)} tickers avec secteur")

print("Chargement des market caps (pour BL)...")
mcap_weights = load_market_cap_weights(_tickers_for_sectors)

print("Chargement des données de prix et signaux...")
returns = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, parse_dates=True)
volume  = pd.read_csv("data/processed/sp500_volume.csv",  index_col=0, parse_dates=True)

def _load_signal(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    return None

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
    ("sue",                 "data/processed/signals/sue.csv"),
]:
    df = _load_signal(path)
    if df is not None:
        signals[name] = df
        print(f"  Signal '{name}' chargé : {df.shape}")
    else:
        print(f"  Signal '{name}' absent")

tickers = returns.columns.tolist()
volume  = volume.reindex(columns=tickers)
signals = {name: df.reindex(columns=tickers) for name, df in signals.items()}
benchmark_weights = pd.Series(1 / len(tickers), index=tickers)

# --- Boucle sur les configs ---
for cfg in CONFIGS:
    sn  = cfg["sector_neutral"]
    bl  = cfg["use_bl"]
    blc = cfg["bl_confidence"]
    lo   = cfg.get("long_only", False)
    n    = cfg.get("n_positions", 10)
    scap = cfg.get("sector_cap", 0)
    rbuf = cfg.get("rank_buffer", 1.0)
    vs   = cfg.get("volatility_sizing", False)
    hmm  = cfg.get("use_hmm", False)

    if lo:
        tag = f"lo{n}"
        if scap > 0:
            tag += f"_sc{scap}"
        if vs:
            tag += "_vs"
        if hmm:
            tag += "_hmm"
    else:
        tag = f"sn{'1' if sn else '0'}_bl{'1' if bl else '0'}"
        if bl:
            tag += f"_c{int(blc*10)}"
        if hmm:
            tag += "_hmm"

    print(f"\n{'='*60}")
    if lo:
        print(f"  Config : LONG-ONLY top-{n}  sc={scap}  buf={rbuf}  vs={vs}  hmm={hmm}  sn={sn}")
    else:
        print(f"  Config : sector_neutral={sn}  BL={bl}  HMM={hmm}"
              + (f"  (confidence={blc})" if bl else ""))
    print(f"  Sauvegarde → data/processed/backtest/{tag}/")
    print(f"{'='*60}")

    config = BacktestConfig(
        estimation_window=36,
        ridge_alpha=1.0,
        max_position=1.0 / n if lo else 0.05,
        max_short=0.0 if lo else 0.05,
        max_te=0.06,
        max_turnover=0.50 if lo else 0.30,
        transaction_cost_bps=7,
        train_end="2020-01-01",
        sector_neutral=bool(sn),
        use_bl=bool(bl),
        bl_confidence=blc,
        long_only=bool(lo),
        n_positions=int(n),
        sector_cap=int(scap),
        rank_buffer=float(rbuf),
        volatility_sizing=bool(vs),
        use_hmm=bool(hmm),
    )

    engine = BacktestEngine(
        returns=returns,
        signals=signals,
        volume=volume,
        benchmark_weights=benchmark_weights,
        config=config,
        universe_snapshots=universe_snapshots,
        sector_map=sector_map,
        mcap_weights=mcap_weights if bl else None,
    )

    results = engine.run()

    # --- Métriques ---
    r = results.returns
    monthly_returns = (1 + returns).resample("MS").prod() - 1
    # portfolio_returns[t] = rendement gagné APRÈS le rebalancement de t,
    # c'est-à-dire le rendement du mois t+1. On décale le benchmark d'un mois
    # pour aligner les deux séries sur la même période économique.
    benchmark_returns = monthly_returns.mean(axis=1).shift(-1).reindex(r.index)
    excess    = r - benchmark_returns.values
    sharpe    = r.mean() / r.std() * np.sqrt(12)
    ir        = excess.mean() / excess.std() * np.sqrt(12)
    max_dd    = (r.cumsum() - r.cumsum().cummax()).min()
    avg_to    = results.turnover.mean()
    avg_cost  = results.costs.mean()

    print(f"\n  Sharpe           : {sharpe:.3f}")
    print(f"  Info Ratio       : {ir:.3f}")
    print(f"  Max Drawdown     : {max_dd:.2%}")
    print(f"  Rendement annuel : {r.mean() * 12:.2%}")
    print(f"  Turnover moyen   : {avg_to:.2%}")
    print(f"  Coût moyen/mois  : {avg_cost:.4%}")

    # --- Sauvegarde ---
    out = Path(f"data/processed/backtest/{tag}")
    out.mkdir(parents=True, exist_ok=True)

    r.to_csv(out / "portfolio_returns.csv")
    benchmark_returns.to_csv(out / "benchmark_returns.csv")
    results.weights.to_csv(out / "weights.csv")
    results.factor_weights.to_csv(out / "factor_weights.csv")
    results.turnover.to_csv(out / "turnover.csv")
    results.costs.to_csv(out / "costs.csv")
    pd.Series({
        "sharpe": sharpe, "ir": ir, "max_dd": max_dd,
        "annual_return": r.mean() * 12, "avg_turnover": avg_to,
        "avg_cost": avg_cost, "sector_neutral": sn, "use_bl": bl,
        "bl_confidence": blc,
    }).to_csv(out / "metrics.csv", header=False)

    # Lien symbolique "default" → première config (sn1_bl0)
    default_link = Path("data/processed/backtest/default")
    if not default_link.exists():
        default_link.symlink_to(tag)

    print(f"  Sauvegardé dans {out}/")

print(f"\nTous les backtests terminés.")
print("Configs disponibles :", [cfg for cfg in Path("data/processed/backtest").iterdir()
                                  if cfg.is_dir()] if Path("data/processed/backtest").exists() else [])
