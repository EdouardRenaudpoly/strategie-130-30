import numpy as np
import pandas as pd


def transaction_cost(
    prev_weights: pd.Series,
    new_weights: pd.Series,
    cost_bps: float = 7,
) -> float:
    """
    Coût de transaction proportionnel au turnover.
    7bps par transaction est une approximation conservative pour large caps US.
    """
    turnover = np.abs(new_weights - prev_weights.reindex(new_weights.index).fillna(0)).sum()
    return turnover * cost_bps / 10_000


def borrow_cost(
    weights: pd.Series,
    avg_daily_volume: pd.Series | None,
) -> float:
    """
    Approximation du coût d'emprunt sur les positions short.
    Les vrais taux sont propriétaires (prime broker) — on utilise la liquidité
    comme proxy : moins un titre est liquide, plus il est cher à shorter.
    """
    short_mask = weights < 0
    if not short_mask.any():
        return 0.0

    short_weights = weights[short_mask].abs()

    if avg_daily_volume is None:
        # Fallback : taux flat 50bps/an sur tous les shorts
        annual_rate = 0.005
    else:
        adv = avg_daily_volume.reindex(short_weights.index).fillna(0)
        # Taux annuel selon liquidité
        rates = np.where(adv > 50e6, 0.003, np.where(adv > 10e6, 0.015, 0.05))
        annual_rate = (short_weights.values * rates).sum() / short_weights.sum()

    # Mensualiser le taux annuel
    return float((short_weights * annual_rate / 12).sum())


def total_frictions(
    prev_weights: pd.Series,
    new_weights: pd.Series,
    avg_daily_volume: pd.Series | None = None,
    transaction_cost_bps: float = 7,
) -> dict[str, float]:
    tc = transaction_cost(prev_weights, new_weights, transaction_cost_bps)
    bc = borrow_cost(new_weights, avg_daily_volume)
    return {"transaction_cost": tc, "borrow_cost": bc, "total": tc + bc}
