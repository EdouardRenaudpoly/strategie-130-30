import pandas as pd


def high_52w(prices: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Ratio prix courant / plus haut sur 52 semaines.
    Proche de 1.0 = près du plus haut → signal positif (George & Hwang 2004).
    Corrélé au momentum mais capture la résistance psychologique au niveau du plus haut.
    """
    return prices / prices.rolling(window, min_periods=int(window * 0.75)).max()
