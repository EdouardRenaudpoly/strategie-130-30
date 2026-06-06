"""
Analyse des expositions sectorielles (GICS) du portefeuille 130/30.

Deux niveaux :
  1. Analyse pure  : exposition nette par secteur à chaque rebalancement
  2. Neutralisation : soustraction de la moyenne sectorielle des signaux
                      avant de les passer au Ridge (z-score intra-secteur)
"""

import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path


_SECTOR_CACHE = Path("data/raw/sectors.json")

GICS_SECTORS = [
    "Communication Services", "Consumer Discretionary", "Consumer Staples",
    "Energy", "Financials", "Health Care", "Industrials", "Information Technology",
    "Materials", "Real Estate", "Utilities",
]


# ---------------------------------------------------------------------------
# Fetch secteurs
# ---------------------------------------------------------------------------

def load_sector_map(tickers: list[str], force: bool = False) -> dict[str, str]:
    """
    Retourne {ticker: sector} via yfinance. Cache local dans data/raw/sectors.json.
    Les tickers sans secteur disponible reçoivent 'Unknown'.
    """
    if _SECTOR_CACHE.exists() and not force:
        cached = json.loads(_SECTOR_CACHE.read_text())
        missing = [t for t in tickers if t not in cached]
        if not missing:
            return {t: cached.get(t, "Unknown") for t in tickers}
        tickers_to_fetch = missing
        sector_map = cached
    else:
        tickers_to_fetch = tickers
        sector_map = {}

    print(f"  Fetch secteurs pour {len(tickers_to_fetch)} tickers...")
    for i, ticker in enumerate(tickers_to_fetch):
        try:
            info = yf.Ticker(ticker).info
            sector_map[ticker] = info.get("sector") or "Unknown"
        except Exception:
            sector_map[ticker] = "Unknown"
        if i % 50 == 0 and i > 0:
            print(f"    [{i}/{len(tickers_to_fetch)}]")
        time.sleep(0.05)

    _SECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _SECTOR_CACHE.write_text(json.dumps(sector_map))
    return {t: sector_map.get(t, "Unknown") for t in tickers}


# ---------------------------------------------------------------------------
# Exposition sectorielle
# ---------------------------------------------------------------------------

def compute_sector_exposures(
    weights: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Calcule l'exposition nette par secteur à chaque date de rebalancement.
    weights : DataFrame (dates x tickers) — poids actifs (sum longs ~1.3, sum shorts ~-0.3)
    Retourne DataFrame (dates x secteurs).
    """
    sectors = sorted(set(sector_map.values()))
    exposures = pd.DataFrame(index=weights.index, columns=sectors, dtype=float)

    for date in weights.index:
        row = weights.loc[date].dropna()
        for sector in sectors:
            tickers_in = [t for t in row.index if sector_map.get(t) == sector]
            exposures.loc[date, sector] = row[tickers_in].sum() if tickers_in else 0.0

    return exposures


def sector_exposure_summary(
    weights: pd.DataFrame,
    sector_map: dict[str, str],
    benchmark_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Résumé des expositions actives moyennes par secteur.
    Si benchmark_weights fourni, calcule l'exposition active (vs benchmark).
    """
    exposures = compute_sector_exposures(weights, sector_map)

    # Exposition benchmark (poids égaux par défaut)
    if benchmark_weights is not None:
        bm_exp = {}
        for sector in exposures.columns:
            tickers_in = [t for t in benchmark_weights.index if sector_map.get(t) == sector]
            bm_exp[sector] = benchmark_weights[tickers_in].sum() if tickers_in else 0.0
        bm_series = pd.Series(bm_exp)
    else:
        n_tickers = len([t for t in weights.columns if t in sector_map])
        bm_series = pd.Series(
            {s: len([t for t in weights.columns if sector_map.get(t) == s]) / n_tickers
             for s in exposures.columns}
        )

    summary = pd.DataFrame({
        "portfolio_avg": exposures.mean(),
        "benchmark":     bm_series,
        "active_avg":    exposures.mean() - bm_series,
        "active_std":    exposures.std(),
        "active_max":    (exposures - bm_series).abs().max(),
    }).sort_values("active_avg", ascending=False)

    return summary


# ---------------------------------------------------------------------------
# Neutralisation sectorielle des signaux
# ---------------------------------------------------------------------------

def neutralize_signal(
    signal: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Neutralise un signal cross-sectionnel par secteur :
    pour chaque date, soustrait la moyenne sectorielle (z-score intra-secteur).
    Résultat : un signal "pure alpha" sans biais sectoriel.
    """
    neutralized = signal.copy()

    for date in signal.index:
        row = signal.loc[date]
        for sector in set(sector_map.values()):
            tickers_in = [t for t in row.index if sector_map.get(t) == sector and pd.notna(row[t])]
            if len(tickers_in) < 3:
                continue
            vals = row[tickers_in]
            std = vals.std()
            if std > 0:
                neutralized.loc[date, tickers_in] = (vals - vals.mean()) / std
            else:
                neutralized.loc[date, tickers_in] = 0.0

    return neutralized
