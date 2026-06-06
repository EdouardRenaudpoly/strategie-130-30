"""
Purged K-Fold Cross-Validation pour séries temporelles financières.
Lopez de Prado, "Advances in Financial Machine Learning" (2018), Ch. 7.

Problème du K-Fold standard sur séries temporelles :
  Si le label de l'obs. t est le rendement forward t→t+1, une observation
  dans le fold de train peut "voir" le futur du fold de test via autocorrélation.

Solution :
  1. Purge  : supprimer du train les obs. dont les labels chevauchent le test
  2. Embargo : supprimer du train les obs. juste après le test (autocorrélation résiduelle)

Compatible avec l'API sklearn (split / get_n_splits).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr


class PurgedKFold:
    """
    K-Fold avec purge + embargo, adapté aux données financières mensuelles.

    n_splits    : nombre de folds
    embargo_td  : timedelta d'embargo après chaque fold de test
                  (ex: pd.DateOffset(months=1))
    """

    def __init__(self, n_splits: int = 5, embargo_td=None):
        self.n_splits   = n_splits
        self.embargo_td = embargo_td or pd.DateOffset(months=1)

    def split(self, X: pd.DataFrame, y=None, groups=None):
        """
        Génère (train_indices, test_indices) avec purge et embargo.
        X doit être un DataFrame avec un DatetimeIndex trié.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X doit avoir un DatetimeIndex.")

        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            # Délimitation du fold test
            t0 = k * fold_size
            t1 = (k + 1) * fold_size if k < self.n_splits - 1 else n

            test_idx  = indices[t0:t1]
            test_start = X.index[t0]
            test_end   = X.index[t1 - 1]

            # Embargo : exclure les obs. juste après le test
            embargo_end = test_end + self.embargo_td

            # Train : tout ce qui est avant test_start OU après embargo_end
            # (pas de purge explicite car label = 1 mois → obs. immédiatement
            #  avant le test est déjà exclue par le fold boundary)
            train_mask = (X.index < test_start) | (X.index > embargo_end)
            train_idx  = indices[train_mask]

            if len(train_idx) < 10:
                continue

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# ---------------------------------------------------------------------------
# Tuning du ridge_alpha via Purged K-Fold
# ---------------------------------------------------------------------------

def tune_ridge_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: list[float] | None = None,
    n_splits: int = 5,
    embargo_months: int = 1,
    verbose: bool = True,
) -> tuple[float, pd.DataFrame]:
    """
    Sélectionne le meilleur ridge_alpha par Purged K-Fold CV.

    Métrique : IC moyen (Spearman) sur les folds de validation.

    X       : (n_obs x n_signals) — signaux z-scorés
    y       : (n_obs,)             — rendements forward cross-sectionnels
    alphas  : grille de régularisation à tester
    Retourne (best_alpha, DataFrame des scores par alpha et fold)
    """
    if alphas is None:
        alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    pkf = PurgedKFold(n_splits=n_splits, embargo_td=pd.DateOffset(months=embargo_months))
    records = []

    for alpha in alphas:
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(pkf.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Supprimer les NaN
            mask_train = y_train.notna() & X_train.notna().all(axis=1)
            mask_test  = y_test.notna()  & X_test.notna().all(axis=1)
            if mask_train.sum() < 5 or mask_test.sum() < 5:
                continue

            model = Ridge(alpha=alpha)
            model.fit(X_train[mask_train], y_train[mask_train])

            y_pred = model.predict(X_test[mask_test])
            ic, _ = spearmanr(y_pred, y_test[mask_test])
            fold_scores.append(ic)
            records.append({"alpha": alpha, "fold": fold, "IC": ic})

        mean_ic = np.mean(fold_scores) if fold_scores else np.nan
        if verbose:
            print(f"  alpha={alpha:>7.2f}  mean_IC={mean_ic:+.4f}  "
                  f"std={np.std(fold_scores):.4f}  n_folds={len(fold_scores)}")

    scores_df = pd.DataFrame(records)
    if scores_df.empty:
        return 1.0, scores_df

    best_alpha = (
        scores_df.groupby("alpha")["IC"].mean()
        .idxmax()
    )
    return best_alpha, scores_df
