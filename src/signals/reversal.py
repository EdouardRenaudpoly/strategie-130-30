import pandas as pd


def reversal(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    # Rendement du dernier mois inversé — les titres qui ont récemment
    # sous-performé tendent à mean-revert à court terme (Jegadeesh, 1990).
    # C'est exactement le mois exclu du momentum 12-1, exploité séparément
    return -returns.rolling(window).sum()
