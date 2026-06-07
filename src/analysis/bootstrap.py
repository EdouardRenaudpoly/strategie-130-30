"""
Bootstrap par blocs pour intervalles de confiance sur métriques de portefeuille.

Utilise le block bootstrap (blocs de 12 mois) pour préserver l'autocorrélation
des rendements — plus honnête que le bootstrap i.i.d. sur des séries temporelles.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MetricCI:
    point: float
    lower: float
    upper: float
    ci_level: float = 0.95

    def fmt(self, pct: bool = False, decimals: int = 3) -> str:
        scale = 100 if pct else 1
        p, lo, hi = self.point * scale, self.lower * scale, self.upper * scale
        d = decimals if not pct else 1
        return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _block_bootstrap_sample(
    returns: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(returns)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    blocks = [returns[s : s + block_size] for s in starts]
    return np.concatenate(blocks)[:n]


def _sharpe(r: np.ndarray) -> float:
    std = r.std()
    return float(r.mean() / std * np.sqrt(12)) if std > 0 else 0.0


def _ir(r: np.ndarray, bench: np.ndarray) -> float:
    excess = r - bench
    std = excess.std()
    return float(excess.mean() / std * np.sqrt(12)) if std > 0 else 0.0


def _annual_return(r: np.ndarray) -> float:
    return float(r.mean() * 12)


def _max_drawdown(r: np.ndarray) -> float:
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    dd = (cum - running_max) / running_max
    return float(dd.min())


def compute_bootstrap_ci(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    block_size: int = 12,
    seed: int = 42,
) -> dict[str, MetricCI]:
    """
    Calcule les IC bootstrap par blocs pour Sharpe, IR, rendement annuel et max drawdown.

    Paramètres
    ----------
    returns   : rendements mensuels du portefeuille
    benchmark : rendements mensuels du benchmark (requis pour IR)
    n_boot    : nombre de réplications bootstrap
    ci_level  : niveau de confiance (0.95 = IC 95%)
    block_size: taille des blocs en mois (12 = un an)
    """
    r = returns.dropna().values
    b = benchmark.reindex(returns.index).dropna().values if benchmark is not None else None
    if b is not None and len(b) != len(r):
        min_len = min(len(r), len(b))
        r = r[-min_len:]
        b = b[-min_len:]

    alpha = (1 - ci_level) / 2
    rng = np.random.default_rng(seed)

    boot_sharpe = np.empty(n_boot)
    boot_ir     = np.empty(n_boot)
    boot_ann    = np.empty(n_boot)
    boot_dd     = np.empty(n_boot)

    for i in range(n_boot):
        rs = _block_bootstrap_sample(r, block_size, rng)
        boot_sharpe[i] = _sharpe(rs)
        boot_ann[i]    = _annual_return(rs)
        boot_dd[i]     = _max_drawdown(rs)
        if b is not None:
            bs = _block_bootstrap_sample(b, block_size, rng)
            boot_ir[i] = _ir(rs, bs)
        else:
            boot_ir[i] = np.nan

    def ci(arr: np.ndarray, point: float) -> MetricCI:
        return MetricCI(
            point=point,
            lower=float(np.nanpercentile(arr, alpha * 100)),
            upper=float(np.nanpercentile(arr, (1 - alpha) * 100)),
            ci_level=ci_level,
        )

    return {
        "sharpe":         ci(boot_sharpe, _sharpe(r)),
        "ir":             ci(boot_ir,     _ir(r, b) if b is not None else np.nan),
        "annual_return":  ci(boot_ann,    _annual_return(r)),
        "max_drawdown":   ci(boot_dd,     _max_drawdown(r)),
    }
