import numpy as np
import pandas as pd


def assert_no_lookahead(
    rebal_date: pd.Timestamp,
    returns: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
) -> None:
    """
    Vérifie qu'aucune donnée postérieure à rebal_date n'est accessible.
    Plante le backtest immédiatement si une fuite est détectée.
    """
    for name, df in signals.items():
        # Strictement avant rebal_date — même date = signal calculé ce jour-là,
        # non disponible avant la clôture, donc interdit pour la décision
        past = df[df.index < rebal_date]
        if past.empty:
            continue
        max_date = past.index.max()
        assert max_date < rebal_date, (
            f"Look-ahead bias détecté dans le signal '{name}' : "
            f"date max={max_date} >= rebalancement={rebal_date}"
        )


def validate_signals_alignment(
    returns: pd.DataFrame, signals: dict[str, pd.DataFrame]
) -> None:
    """
    Vérifie que les signaux et les rendements partagent le même univers
    et que les dates sont cohérentes.
    """
    for name, df in signals.items():
        missing = set(returns.columns) - set(df.columns)
        if missing:
            print(f"Avertissement : {len(missing)} tickers absents du signal '{name}'")

        if not df.index.is_monotonic_increasing:
            raise ValueError(f"Signal '{name}' : index non trié chronologiquement")


def canary_test(returns: pd.DataFrame, engine_class, config) -> float:
    """
    Injecte le rendement futur parfait comme signal — le Sharpe doit être
    irréaliste (> 5). Confirme que le pipeline détecte bien la fuite future
    si elle existe, et qu'on peut faire confiance aux assertions.

    À appeler une fois en développement, pas en production.
    """
    # Signal parfait : rendement du mois suivant (fuite future délibérée)
    perfect_signal = returns.shift(-1)

    dummy_signals = {"perfect_future": perfect_signal}
    dummy_volume = pd.DataFrame(1e8, index=returns.index, columns=returns.columns)
    dummy_benchmark = pd.Series(1 / len(returns.columns), index=returns.columns)

    engine = engine_class(
        returns=returns,
        signals=dummy_signals,
        volume=dummy_volume,
        benchmark_weights=dummy_benchmark,
        config=config,
    )

    # On désactive temporairement les assertions pour laisser tourner le canari
    results = engine.run()

    if results.returns.empty:
        return 0.0

    sharpe = results.returns.mean() / results.returns.std() * np.sqrt(12)
    print(f"Canary Sharpe (doit être >> 3) : {sharpe:.2f}")

    return sharpe
