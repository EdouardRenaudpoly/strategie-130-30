"""
Calibration du ridge_alpha via Purged K-Fold CV — approche correcte.

La bonne formulation pour notre stratégie cross-sectionnelle :
  - Chaque "observation" du CV est un MOIS entier (pas un ticker individuel)
  - À chaque mois t du fold train, on fait une régression cross-sectionnelle
    Ridge(X_t, y_t) où X_t = (n_tickers x n_signals) et y_t = rendements forward
  - Le score sur le fold test = IC moyen sur les mois de test

Cela évite le leakage : tous les tickers d'un mois t sont soit dans le train,
soit dans le test — pas répartis entre les deux.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from src.backtest.purged_kfold import PurgedKFold
from src.universe.historical_constituents import load_snapshots, get_universe_at_date

# --- Données ---
print("Chargement des données...")
returns = pd.read_csv("data/processed/sp500_returns.csv", index_col=0, parse_dates=True)
monthly_returns = (1 + returns).resample("MS").prod() - 1
universe_snapshots = load_snapshots(min_date="2013-01-01")

def _load(path):
    p = Path(path)
    return pd.read_csv(p, index_col=0, parse_dates=True) if p.exists() else None

raw_signals = {}
for name, path in [
    ("momentum",            "data/processed/signals/momentum.csv"),
    ("low_volatility",      "data/processed/signals/low_volatility.csv"),
    ("reversal",            "data/processed/signals/reversal.csv"),
    ("illiquidity",         "data/processed/signals/illiquidity.csv"),
    ("high_52w",            "data/processed/signals/high_52w.csv"),
    ("gross_profitability", "data/processed/signals/gross_profitability.csv"),
    ("accruals",            "data/processed/signals/accruals.csv"),
    ("asset_growth",        "data/processed/signals/asset_growth.csv"),
]:
    df = _load(path)
    if df is not None:
        raw_signals[name] = df.reindex(columns=monthly_returns.columns)

signal_names = list(raw_signals.keys())

# --- Préparer les données par mois ---
# Pour chaque mois t : dict de (X_t, y_t) cross-sectionnels z-scorés
print("Préparation des snapshots mensuels...")

monthly_data = {}   # {date: (X_df, y_series)}
months = monthly_returns.index[:-1]

for t in months:
    valid_tickers = get_universe_at_date(universe_snapshots, t)
    valid_tickers = [tk for tk in valid_tickers if tk in monthly_returns.columns]
    if len(valid_tickers) < 50:
        continue

    t_fwd = monthly_returns.index[monthly_returns.index.get_loc(t) + 1]
    fwd_ret = monthly_returns.loc[t_fwd, valid_tickers].dropna()
    if len(fwd_ret) < 50:
        continue
    common = fwd_ret.index.tolist()

    # Signaux z-scorés cross-sectionnellement
    sig_dict = {}
    for name, df in raw_signals.items():
        if t not in df.index:
            continue
        s = df.loc[t, common]
        std = s.std()
        if std > 0:
            sig_dict[name] = ((s - s.mean()) / std)

    if len(sig_dict) < 3:
        continue

    X_t = pd.DataFrame(sig_dict, index=common).fillna(0.0)[signal_names]
    X_t = X_t[[c for c in signal_names if c in X_t.columns]]

    fwd_std = fwd_ret.std()
    y_t = (fwd_ret - fwd_ret.mean()) / fwd_std if fwd_std > 0 else fwd_ret * 0

    monthly_data[t] = (X_t, y_t)

all_months = pd.DatetimeIndex(sorted(monthly_data.keys()))
print(f"  {len(all_months)} mois disponibles | {all_months[0].date()} → {all_months[-1].date()}")

# DataFrame factice pour PurgedKFold (on a besoin d'un DatetimeIndex)
dummy = pd.DataFrame(index=all_months)


def score_alpha(alpha: float, train_months, test_months) -> float:
    """Entraîne Ridge sur train_months, mesure IC moyen sur test_months."""
    # Empiler toutes les obs. du fold train
    X_train_list, y_train_list = [], []
    for t in train_months:
        if t not in monthly_data:
            continue
        X_t, y_t = monthly_data[t]
        X_train_list.append(X_t)
        y_train_list.append(y_t)

    if not X_train_list:
        return np.nan

    X_train = pd.concat(X_train_list).fillna(0.0)
    y_train = pd.concat(y_train_list)

    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    # IC moyen sur les mois de test
    ics = []
    for t in test_months:
        if t not in monthly_data:
            continue
        X_t, y_t = monthly_data[t]
        cols = [c for c in X_train.columns if c in X_t.columns]
        if not cols:
            continue
        y_pred = model.predict(X_t[cols])
        ic, _ = spearmanr(y_pred, y_t)
        if not np.isnan(ic):
            ics.append(ic)

    return np.mean(ics) if ics else np.nan


# --- Purged K-Fold CV ---
alphas = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
n_splits = 5

print(f"\nPurged K-Fold CV (n_splits={n_splits}, embargo=1 mois)...")
print(f"Chaque fold = bloc de ~{len(all_months)//n_splits} mois consécutifs\n")

pkf = PurgedKFold(n_splits=n_splits, embargo_td=pd.DateOffset(months=1))
records = []

for alpha in alphas:
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(pkf.split(dummy)):
        train_months = all_months[train_idx]
        test_months  = all_months[test_idx]
        ic = score_alpha(alpha, train_months, test_months)
        if not np.isnan(ic):
            fold_scores.append(ic)
            records.append({"alpha": alpha, "fold": fold, "IC": ic})

    mean_ic = np.mean(fold_scores) if fold_scores else np.nan
    std_ic  = np.std(fold_scores)  if fold_scores else np.nan
    print(f"  alpha={alpha:>7.2f}  mean_IC={mean_ic:+.4f}  std={std_ic:.4f}  n_folds={len(fold_scores)}")

# --- Résultat ---
scores_df = pd.DataFrame(records)
by_alpha  = scores_df.groupby("alpha")["IC"].agg(["mean", "std", "count"])
best_alpha = float(by_alpha["mean"].idxmax())

print(f"\n{'='*55}")
print(f"{'Alpha':>8}  {'Mean IC':>8}  {'Std IC':>7}  {'ICIR':>6}  {'N':>4}")
print("-"*55)
for alpha, row in by_alpha.iterrows():
    icir   = row["mean"] / row["std"] * np.sqrt(n_splits) if row["std"] > 0 else np.nan
    marker = " ← optimal" if alpha == best_alpha else ""
    print(f"{alpha:>8.2f}  {row['mean']:>+8.4f}  {row['std']:>7.4f}  {icir:>+6.3f}  {int(row['count']):>4}{marker}")
print(f"{'='*55}")
print(f"\n→ Meilleur ridge_alpha = {best_alpha}")
print(f"  Mets à jour BacktestConfig(ridge_alpha={best_alpha}) dans run_backtest.py")

scores_df.to_csv("data/processed/analysis/purged_kfold_scores.csv", index=False)
