import pandas as pd

prices = pd.read_csv("../../data/processed/sp500_prices.csv", index_col=0, parse_dates=True)

returns = prices.pct_change().dropna(how="all")

returns.to_csv("../../data/processed/sp500_returns.csv")
