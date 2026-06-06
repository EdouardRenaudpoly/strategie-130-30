import pandas as pd


def momentum(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    # Le mois le plus récent est exclu (skip=21 jours) car il tend à mean-revert
    # à court terme — l'inclure dégraderait le signal (Jegadeesh & Titman, 1993)
    return prices.shift(skip) / prices.shift(lookback) - 1
