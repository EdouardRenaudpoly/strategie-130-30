import numpy as np
import pandas as pd


def illiquidity(returns: pd.DataFrame, volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    # Ratio d'Amihud : |rendement| / volume en dollars — mesure l'impact
    # prix par dollar échangé. Un ratio élevé = titre illiquide.
    # Signal positif : la prime d'illiquidité récompense les investisseurs
    # qui supportent le risque de liquidité (Amihud, 2002)
    amihud = returns.abs() / volume.replace(0, np.nan)
    return amihud.rolling(window).mean()
