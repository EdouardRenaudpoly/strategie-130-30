"""
Refresh complet des données et signaux.

Usage :
  uv run python refresh.py                         # tout rafraîchir
  uv run python refresh.py --no-prices             # signaux seulement
  uv run python refresh.py --no-insider            # sauter insider (lent au 1er run)
  uv run python refresh.py --no-fundamentals       # sauter XBRL (lent au 1er run)

Signaux produits :
  data/processed/signals/momentum.csv
  data/processed/signals/low_volatility.csv
  data/processed/signals/reversal.csv
  data/processed/signals/illiquidity.csv
  data/processed/signals/high_52w.csv
  data/processed/signals/insider_trading.csv     (EDGAR, gratuit)
  data/processed/signals/gross_profitability.csv (SEC XBRL, gratuit)
  data/processed/signals/accruals.csv            (SEC XBRL, gratuit)

Fiabilité des sources :
  yfinance      → officieux, communauté le répare vite si ça casse
  SEC EDGAR     → officiel US, très stable depuis 2009
  Wikipedia     → stable, maintenu par la communauté
"""

import sys
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path

from src.universe.historical_constituents import load_snapshots, all_unique_tickers
from src.signals.insider_trading import get_cik_map
from src.signals.momentum import momentum
from src.signals.low_volatility import low_volatility
from src.signals.reversal import reversal
from src.signals.illiquidity import illiquidity
from src.signals.high_52w import high_52w

args = sys.argv[1:]
skip_prices       = "--no-prices"       in args
skip_insider      = "--no-insider"      in args
skip_fundamentals = "--no-fundamentals" in args

Path("data/processed/signals").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Check de fraîcheur — plante si les données sont trop vieilles
# ---------------------------------------------------------------------------

def check_freshness(prices: pd.DataFrame, signals: dict[str, pd.DataFrame]) -> None:
    """
    Vérifie que les données sont suffisamment récentes pour un refresh mensuel.
    Affiche des avertissements clairs plutôt que de continuer silencieusement.
    """
    today = pd.Timestamp.today().normalize()
    errors = []

    # Prix : acceptable jusqu'à 5 jours de retard (weekends + fériés)
    price_lag = (today - prices.index[-1]).days
    if price_lag > 5:
        errors.append(f"PRIX : dernière date {prices.index[-1].date()} — {price_lag} jours de retard")

    # Signaux : tolérances différentes selon le type
    thresholds = {
        "momentum":          40,   # mensuel
        "low_volatility":    40,
        "reversal":          40,
        "illiquidity":       40,
        "high_52w":          40,
        "insider_trading":   40,   # mis à jour mensuellement
        "gross_profitability": 400, # annuel — tolérance 13 mois
        "accruals":          400,
        "asset_growth":      400,
    }

    for name, df in signals.items():
        if df.empty:
            continue
        lag = (today - df.index[-1]).days
        threshold = thresholds.get(name, 40)
        if lag > threshold:
            errors.append(f"SIGNAL '{name}' : dernière date {df.index[-1].date()} — {lag} jours de retard")

    if errors:
        print("\n" + "="*60)
        print("ALERTES DE FRAÎCHEUR DES DONNÉES :")
        for e in errors:
            print(f"  ⚠  {e}")
        print("="*60)
        raise SystemExit(
            "\nRefresh interrompu : données trop vieilles.\n"
            "Lance 'uv run python refresh.py' pour mettre à jour."
        )
    else:
        print("  Fraîcheur OK — toutes les données sont à jour.")


# ---------------------------------------------------------------------------
# Univers historique
# ---------------------------------------------------------------------------

print("Chargement de l'univers historique...")
snapshots = load_snapshots(min_date="2013-01-01")
tickers   = all_unique_tickers(snapshots)
print(f"  {len(tickers)} tickers uniques")

# CIK map chargé une fois (utilisé par insider + fundamentals)
cik_map = get_cik_map()

# ---------------------------------------------------------------------------
# Prix et volume
# ---------------------------------------------------------------------------

if not skip_prices:
    print("Téléchargement des prix et volumes...")
    end   = date.today()
    start = end - timedelta(days=365 * 11)
    raw   = yf.download(tickers, start=start, end=end, auto_adjust=True)  # type: ignore[index]

    prices  = raw["Close"].dropna(how="all", axis=1)
    volume  = raw["Volume"].dropna(how="all", axis=1)
    returns = prices.pct_change().dropna(how="all")

    prices.to_csv("data/processed/sp500_prices.csv")
    volume.to_csv("data/processed/sp500_volume.csv")
    returns.to_csv("data/processed/sp500_returns.csv")
    print(f"  {prices.shape[1]} tickers | {prices.index[0].date()} → {prices.index[-1].date()}")
else:
    print("Chargement des prix existants...")
    prices  = pd.read_csv("data/processed/sp500_prices.csv",  index_col=0, parse_dates=True)
    volume  = pd.read_csv("data/processed/sp500_volume.csv",  index_col=0, parse_dates=True)
    returns = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, parse_dates=True)

# ---------------------------------------------------------------------------
# Signaux prix (rapides — yfinance)
# ---------------------------------------------------------------------------

print("Calcul des signaux prix...")
momentum(prices).to_csv("data/processed/signals/momentum.csv")
low_volatility(returns).to_csv("data/processed/signals/low_volatility.csv")
reversal(returns).to_csv("data/processed/signals/reversal.csv")
illiquidity(returns, volume).to_csv("data/processed/signals/illiquidity.csv")
high_52w(prices).to_csv("data/processed/signals/high_52w.csv")
print("  momentum, low_vol, reversal, illiquidity, high_52w : OK")

# ---------------------------------------------------------------------------
# Insider trading (SEC EDGAR Form 4)
# ---------------------------------------------------------------------------

if not skip_insider:
    print("Signal insider trading (EDGAR)...")
    print("  1er run : ~20-30 min | Suivants : <1 min (cache)")
    from src.signals.insider_trading import compute_insider_signal
    insider = compute_insider_signal(
        tickers=prices.columns.tolist(),
        start_date="2013-01-01",
        verbose=True,
    )
    if not insider.empty:
        insider.to_csv("data/processed/signals/insider_trading.csv")
        print(f"  insider_trading : OK {insider.shape}")
    else:
        print("  insider_trading : aucune donnée")
else:
    print("Insider trading ignoré (--no-insider)")

# ---------------------------------------------------------------------------
# Fondamentaux via SEC EDGAR XBRL (remplace SimFin)
# ---------------------------------------------------------------------------

if not skip_fundamentals:
    print("Signaux fondamentaux (SEC EDGAR XBRL)...")
    print("  1er run : ~15-20 min | Suivants : <2 min (cache 30 jours)")
    from src.signals.fundamentals import compute_fundamentals
    fund = compute_fundamentals(
        tickers=prices.columns.tolist(),
        cik_map=cik_map,
        start_year=2013,
        verbose=True,
    )
    for name, df in fund.items():
        if not df.empty:
            df.to_csv(f"data/processed/signals/{name}.csv")
            print(f"  {name} : OK {df.shape}")
        else:
            print(f"  {name} : aucune donnée")
else:
    print("Fondamentaux ignorés (--no-fundamentals)")

# ---------------------------------------------------------------------------
# Check de fraîcheur final
# ---------------------------------------------------------------------------

print("\nVérification de la fraîcheur des données...")
all_signals = {}
for sig_path in Path("data/processed/signals").glob("*.csv"):
    df = pd.read_csv(sig_path, index_col=0, parse_dates=True)
    all_signals[sig_path.stem] = df

check_freshness(prices, all_signals)

print("\nRefresh terminé.")
