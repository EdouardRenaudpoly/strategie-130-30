import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.universe.historical_constituents import load_snapshots, all_unique_tickers

# Univers historique — tous les tickers qui ont été dans l'indice depuis 2013
snapshots = load_snapshots(min_date="2013-01-01")
tickers = all_unique_tickers(snapshots)
print(f"Tickers historiques uniques : {len(tickers)}")

end = date.today()
start = end - timedelta(days=365 * 11)  # 11 ans pour avoir assez d'historique

print(f"Téléchargement prix {start} → {end}...")
raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

prices = raw["Close"]
volume = raw["Volume"]

# Retirer les colonnes entièrement vides (tickers sans données yfinance — délistés, etc.)
prices = prices.dropna(how="all", axis=1)
volume = volume.dropna(how="all", axis=1)

print(f"Tickers avec données : {prices.shape[1]} / {len(tickers)}")
print(f"Période : {prices.index[0].date()} → {prices.index[-1].date()}")

Path("../../data/processed").mkdir(parents=True, exist_ok=True)
prices.to_csv("../../data/processed/sp500_prices.csv")
volume.to_csv("../../data/processed/sp500_volume.csv")
print("Sauvegardé : sp500_prices.csv, sp500_volume.csv")
