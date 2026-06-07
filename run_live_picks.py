"""
Picks mensuels du modèle 130/30 — version déployable.

Usage :
  uv run python run_live_picks.py                  # capital par défaut $2000
  uv run python run_live_picks.py --capital 5000   # capital personnalisé
  uv run python run_live_picks.py --longs 10       # top 10 longs au lieu de 8

Sorties :
  - Top N longs à acheter avec poids suggérés
  - Top 3 shorts théoriques (pour référence)
  - Delta vs mois précédent (entrants / sortants)
  - Sauvegarde dans data/processed/live_picks_YYYY-MM.csv
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.universe.historical_constituents import load_snapshots, get_universe_at_date
from src.analysis.sector_analysis import load_sector_map, load_market_cap_weights

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

args = sys.argv[1:]
capital        = float(next((args[i+1] for i, a in enumerate(args) if a == "--capital"),      2000))
n_longs        = int(next((args[i+1]   for i, a in enumerate(args) if a == "--longs"),         10))
max_weight     = float(next((args[i+1] for i, a in enumerate(args) if a == "--max-weight"),  0.20))
bl_confidence  = float(next((args[i+1] for i, a in enumerate(args) if a == "--bl-confidence"), 0.5))
sector_neutral = "--no-sector-neutral" not in args
use_bl         = "--bl" in args

# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------

print("Chargement des données...")
returns = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, parse_dates=True)
monthly_returns = (1 + returns).resample("MS").prod() - 1

universe_snapshots = load_snapshots(min_date="2013-01-01")
sector_map = load_sector_map(returns.columns.tolist())

def _load(path):
    p = Path(path)
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None

raw_signals = {}
for name, path in [
    ("momentum",            "data/processed/signals/momentum.csv"),
    ("low_volatility",      "data/processed/signals/low_volatility.csv"),
    ("reversal",            "data/processed/signals/reversal.csv"),
    ("illiquidity",         "data/processed/signals/illiquidity.csv"),
    ("high_52w",            "data/processed/signals/high_52w.csv"),
    ("gross_profitability", "data/processed/signals/gross_profitability.csv"),
    ("accruals",            "data/processed/signals/accruals.csv"),
    ("asset_growth",        "data/processed/signals/asset_growth.csv"),
    ("insider",             "data/processed/signals/insider_trading.csv"),
]:
    df = _load(path)
    if df is not None:
        raw_signals[name] = df.reindex(columns=returns.columns)

signal_names = list(raw_signals.keys())
today = pd.Timestamp.today().normalize()
print(f"  Signaux disponibles : {signal_names}")

# ---------------------------------------------------------------------------
# Univers courant
# ---------------------------------------------------------------------------

current_universe = get_universe_at_date(universe_snapshots, today)
current_universe = [t for t in current_universe if t in returns.columns]
print(f"  Univers courant : {len(current_universe)} tickers")

# ---------------------------------------------------------------------------
# Neutralisation sectorielle intra-secteur
# ---------------------------------------------------------------------------

_sector_series_cache = pd.Series({t: sector_map.get(t, "Unknown") for t in returns.columns})

def sector_neutralize(X: pd.DataFrame) -> pd.DataFrame:
    sectors = _sector_series_cache.reindex(X.index)
    X = X.copy()
    for col in X.columns:
        vals = X[col]
        grp = vals.groupby(sectors)
        counts = grp.transform("count")
        means  = grp.transform("mean")
        stds   = grp.transform("std").fillna(0)
        eligible = counts >= 3
        X.loc[eligible & (stds > 0), col] = ((vals - means) / stds).loc[eligible & (stds > 0)]
        X.loc[eligible & (stds == 0), col] = 0.0
    return X

# ---------------------------------------------------------------------------
# Estimation des factor weights (Fama-MacBeth + Ridge, 36 derniers mois)
# ---------------------------------------------------------------------------

print("\nEstimation des factor weights (36 derniers mois)...")
WINDOW = 36
monthly_signals = {
    name: df.resample("MS").last()
    for name, df in raw_signals.items()
}

monthly_coefs = []
window_months = monthly_returns[monthly_returns.index < today].iloc[-WINDOW:]

for date_t in window_months.index:
    future = window_months.index[window_months.index > date_t]
    if len(future) == 0:
        continue
    y = window_months.loc[future[0], current_universe].dropna()

    X_dict = {}
    for name, df in monthly_signals.items():
        if date_t in df.index:
            X_dict[name] = df.loc[date_t]

    if not X_dict:
        continue

    X = pd.DataFrame(X_dict).reindex(y.index).dropna()
    y = y[X.index]
    if len(y) < 50:
        continue

    if sector_neutral:
        X = sector_neutralize(X)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    monthly_coefs.append(pd.Series(ridge.coef_, index=X.columns))

factor_weights = pd.DataFrame(monthly_coefs).mean()
print(f"  Factor weights estimés sur {len(monthly_coefs)} mois")
for name, w in factor_weights.sort_values(ascending=False).items():
    print(f"    {name:<25} {w:+.5f}")

# ---------------------------------------------------------------------------
# Scores alpha courants
# ---------------------------------------------------------------------------

print("\nCalcul des scores alpha courants...")

# Signaux à la dernière date disponible
X_now_dict = {}
for name, df in monthly_signals.items():
    past = df[df.index < today]
    if past.empty:
        continue
    X_now_dict[name] = past.iloc[-1].reindex(current_universe)

X_now = pd.DataFrame(X_now_dict).dropna()
print(f"  {len(X_now)} tickers avec signaux complets")

if sector_neutral:
    X_now = sector_neutralize(X_now)

common_factors = X_now.columns.intersection(factor_weights.index)
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X_now[common_factors]),
    index=X_now.index,
    columns=common_factors,
)
alpha_scores = (X_scaled @ factor_weights[common_factors]).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# Picks
# ---------------------------------------------------------------------------

prices = pd.read_csv("data/processed/sp500_prices.csv", index_col=0, parse_dates=True)
latest_prices = prices.iloc[-1]

def enrich(tickers, scores):
    rows = []
    for t in tickers:
        price = latest_prices.get(t, np.nan)
        sector = sector_map.get(t, "Unknown")
        rows.append({"ticker": t, "alpha_score": round(scores[t], 4),
                     "sector": sector, "price_usd": round(price, 2) if pd.notna(price) else None})
    return pd.DataFrame(rows)

top_longs  = alpha_scores.head(n_longs).index.tolist()
top_shorts = alpha_scores.tail(3).index.tolist()

longs_df  = enrich(top_longs,  alpha_scores)
shorts_df = enrich(top_shorts, alpha_scores)

# Construction du portefeuille
if use_bl:
    print("\nConstruction Black-Litterman...")
    print("  Chargement des market caps (prior)...")
    mcap_weights = load_market_cap_weights(top_longs)
    from src.optimization.black_litterman import black_litterman_weights
    bl_w = black_litterman_weights(
        returns_hist=monthly_returns[top_longs],
        alpha_scores=alpha_scores[top_longs],
        market_weights=mcap_weights,
        tau=0.05,
        risk_aversion=2.5,
        view_confidence=bl_confidence,
        max_position=max_weight,
        max_short=0.0,        # long-only dans le script picks
        max_turnover=1.0,     # pas de contrainte de turnover sur picks
    )
    if bl_w is not None:
        weights = bl_w.clip(lower=0)
        weights = weights / weights.sum()
        print(f"  BL convergé  (confidence={bl_confidence})")
    else:
        print("  BL n'a pas convergé — fallback proportionnel")
        use_bl = False  # type: ignore[assignment]

if not use_bl:
    # Poids proportionnels aux scores avec plafonnage itératif
    long_scores = alpha_scores[top_longs]
    long_scores = long_scores - long_scores.min() + 0.01
    weights = long_scores / long_scores.sum()
    for _ in range(20):
        capped    = weights.clip(upper=max_weight)
        excess    = weights[weights > max_weight] - max_weight
        if excess.empty:
            break
        free      = weights[weights < max_weight]
        weights   = capped.copy()
        weights[free.index] += excess.sum() * (free / free.sum())
    weights = weights.clip(lower=0.05)
    weights = weights / weights.sum()

longs_df["weight_pct"] = (weights[top_longs].values * 100).round(1)
longs_df["amount_cad"] = (weights[top_longs].values * capital).round(0).astype(int)
longs_df["shares"] = (longs_df["amount_cad"] / longs_df["price_usd"]).round(2)

# ---------------------------------------------------------------------------
# Delta vs mois précédent
# ---------------------------------------------------------------------------

picks_dir = Path("data/processed/live_picks")
picks_dir.mkdir(parents=True, exist_ok=True)

last_month = today - pd.DateOffset(months=1)
prev_file  = picks_dir / f"picks_{last_month.strftime('%Y-%m')}.csv"

if prev_file.exists():
    prev_picks = pd.read_csv(prev_file)
    prev_longs = set(prev_picks[prev_picks["side"] == "long"]["ticker"])
    curr_longs = set(top_longs)
    entrants   = sorted(curr_longs - prev_longs)
    sortants   = sorted(prev_longs - curr_longs)
    unchanged  = sorted(curr_longs & prev_longs)
else:
    entrants = sortants = unchanged = []

# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

month_label = today.strftime("%B %Y")
mode_str = f"BL (c={bl_confidence})" if use_bl else f"Proportionnel (max={max_weight:.0%})"
print(f"\n{'='*65}")
print(f"  PICKS {month_label.upper()}  |  Capital : ${capital:,.0f}  |  {mode_str}  |  SN={'ON' if sector_neutral else 'OFF'}")
print(f"{'='*65}")

print(f"\n▲  TOP {n_longs} LONGS\n")
print(f"  {'Ticker':<8} {'Secteur':<28} {'Score':>7} {'Poids':>6} {'$CAD':>6} {'Actions':>7}")
print(f"  {'-'*67}")
for _, row in longs_df.iterrows():
    flag = " ← NEW" if row["ticker"] in entrants else ""
    print(f"  {row['ticker']:<8} {row['sector']:<28} {row['alpha_score']:>+7.4f} "
          f"{row['weight_pct']:>5.1f}% {row['amount_cad']:>6} {row['shares']:>7.2f}{flag}")

print(f"\n▼  TOP 3 SHORTS (théoriques — pour référence)\n")
print(f"  {'Ticker':<8} {'Secteur':<28} {'Score':>7}")
print(f"  {'-'*46}")
for _, row in shorts_df.iterrows():
    print(f"  {row['ticker']:<8} {row['sector']:<28} {row['alpha_score']:>+7.4f}")

if prev_file.exists():
    print(f"\n  Changements vs mois précédent :")
    if entrants:
        print(f"    ✚ Entrants  : {', '.join(entrants)}")
    if sortants:
        print(f"    ✖ Sortants  : {', '.join(sortants)}")
    if not entrants and not sortants:
        print(f"    Aucun changement")
else:
    print(f"\n  (Premier run — pas de comparaison disponible)")

print(f"\n{'='*62}\n")

# ---------------------------------------------------------------------------
# Sauvegarde
# ---------------------------------------------------------------------------

this_month = today.strftime("%Y-%m")
out_file   = picks_dir / f"picks_{this_month}.csv"

rows = []
for _, r in longs_df.iterrows():
    rows.append({"month": this_month, "side": "long",  "ticker": r["ticker"],
                 "sector": r["sector"], "alpha_score": r["alpha_score"],
                 "weight_pct": r["weight_pct"], "amount_cad": r["amount_cad"]})
for _, r in shorts_df.iterrows():
    rows.append({"month": this_month, "side": "short", "ticker": r["ticker"],
                 "sector": r["sector"], "alpha_score": r["alpha_score"],
                 "weight_pct": None, "amount_cad": None})

pd.DataFrame(rows).to_csv(out_file, index=False)
print(f"Sauvegardé : {out_file}")
print(f"Prochain rebalancement : {(today + pd.DateOffset(months=1)).strftime('%Y-%m-%d')}")
