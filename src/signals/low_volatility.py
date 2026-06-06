import numpy as np
import pandas as pd


def low_volatility(returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    # Volatilité annualisée sur fenêtre glissante, signal inversé :
    # un titre moins volatile reçoit un score plus élevé.
    # L'anomalie low-vol contredit le CAPM — les actifs moins risqués
    # surperforment risk-adjusted (Black, Jensen & Scholes, 1972)
    vol = returns.rolling(window).std() * np.sqrt(252)
    return -vol
