# Stratégie 130/30 cross-sectionnel — S&P 500

Modèle d'alpha quantitatif multi-signaux avec construction de portefeuille 130/30.
Signaux : momentum, low volatility, reversal, illiquidity, high 52w, gross profitability,
accruals, asset growth, insider trading.
Optimisation : Fama-MacBeth + Ridge, neutralisation sectorielle GICS, Black-Litterman optionnel.

---

## Installation

```bash
git clone <repo>
cd strategie-130-30
uv sync
cp .env.example .env   # puis remplir SEC_USER_AGENT_EMAIL
```

---

## Commandes principales

### 1. Refresh des données et signaux

```bash
# Refresh complet (prix + tous les signaux)
uv run python refresh.py

# Sauter le téléchargement insider (lent au premier run — ~8h)
uv run python refresh.py --no-insider

# Sauter les fondamentaux SEC XBRL (lent au premier run)
uv run python refresh.py --no-fundamentals

# Signaux seulement, sans re-télécharger les prix
uv run python refresh.py --no-prices
```

> Le signal insider trading peut être lancé en arrière-plan avec tmux :
> ```bash
> tmux new -s insider
> uv run python refresh.py --no-prices --no-fundamentals
> # Ctrl+B puis D pour détacher
> ```

---

### 2. Backtest walk-forward

```bash
# Backtest standard (neutralisation sectorielle activée par défaut)
uv run python run_backtest.py

# Sans neutralisation sectorielle
uv run python run_backtest.py --no-sector-neutral

# Avec construction Black-Litterman
uv run python run_backtest.py --bl

# BL avec confiance personnalisée dans les vues (0.0 = prior pur, 1.0 = vues pures)
uv run python run_backtest.py --bl --bl-confidence 0.7
```

Résultats sauvegardés dans `data/processed/backtest/` :
- `portfolio_returns.csv`, `benchmark_returns.csv`
- `weights.csv`, `factor_weights.csv`
- `turnover.csv`, `costs.csv`, `metrics.csv`

---

### 3. Analyse IC (Information Coefficient)

```bash
uv run python run_ic_analysis.py
```

Sorties dans `data/processed/analysis/` :
- `ic_series.csv` — IC mensuel par signal
- `ic_summary.csv` — mean IC, ICIR, hit rate, t-stat

---

### 4. Analyse sectorielle

```bash
uv run python run_sector_analysis.py
```

Sorties dans `data/processed/analysis/` :
- `sector_exposures.csv` — exposition par secteur à chaque rebalancement
- `sector_summary.csv` — exposition active moyenne vs benchmark

---

### 5. Tuning Ridge (Purged K-Fold CV)

```bash
uv run python run_tune_alpha.py
```

Compare plusieurs valeurs de `ridge_alpha` via Purged K-Fold avec embargo de 1 mois.
Résultats dans `data/processed/analysis/purged_kfold_scores.csv`.

---

### 6. Live picks mensuels

```bash
# Picks proportionnels, capital $2000, top 8 longs
uv run python run_live_picks.py

# Avec Black-Litterman et market cap weights
uv run python run_live_picks.py --bl

# Capital et nombre de longs personnalisés
uv run python run_live_picks.py --capital 5000 --longs 10

# Poids max par position (sans BL)
uv run python run_live_picks.py --max-weight 0.15

# Confiance BL personnalisée
uv run python run_live_picks.py --bl --bl-confidence 0.7

# Sans neutralisation sectorielle
uv run python run_live_picks.py --no-sector-neutral
```

Sorties dans `data/processed/live_picks/picks_YYYY-MM.csv`.

---

### 7. Dashboard Streamlit

```bash
uv run streamlit run app.py
```

Pages disponibles :
- **Performance** — courbe de valeur, heatmap mensuelle, Sharpe glissant (nécessite le backtest)
- **Signaux & IC** — ICIR par signal, IC cumulatif, corrélation entre signaux
- **Secteurs** — expositions actives, long vs benchmark, drift dans le temps
- **Black-Litterman** — frontière efficiente interactive, nuage Monte Carlo, curseur confiance
- **Live Picks** — treemap allocation, table longs/shorts, historique

---

## Workflow mensuel (rebalancement)

```bash
# 1. Rafraîchir les données (~5 min hors insider)
uv run python refresh.py --no-insider

# 2. Générer les nouveaux picks
uv run python run_live_picks.py --bl

# 3. Ouvrir le dashboard pour visualiser
uv run streamlit run app.py
```

---

## Variables d'environnement

Créer un fichier `.env` à la racine (ne jamais committer) :

```
SEC_USER_AGENT_EMAIL=ton-email@example.com
```

`SEC_USER_AGENT_EMAIL` est requis par l'API SEC EDGAR (signaux fondamentaux et insider trading).

---

## Structure du projet

```
strategie-130-30/
├── app.py                        # Dashboard Streamlit
├── refresh.py                    # Refresh données + signaux
├── run_backtest.py               # Backtest walk-forward
├── run_ic_analysis.py            # Analyse IC par signal
├── run_sector_analysis.py        # Analyse sectorielle
├── run_tune_alpha.py             # Tuning Ridge (Purged K-Fold)
├── run_live_picks.py             # Picks mensuels déployables
├── src/
│   ├── backtest/
│   │   ├── engine.py             # BacktestEngine + BacktestConfig
│   │   ├── purged_kfold.py       # PurgedKFold CV avec embargo
│   │   ├── frictions.py          # Coûts de transaction + borrow
│   │   └── validator.py          # Assertions anti-look-ahead
│   ├── signals/
│   │   ├── momentum.py
│   │   ├── low_volatility.py
│   │   ├── reversal.py
│   │   ├── illiquidity.py
│   │   ├── high_52w.py
│   │   ├── fundamentals.py       # gross_profitability, accruals, asset_growth (SEC XBRL)
│   │   └── insider_trading.py    # Form 4 SEC EDGAR
│   ├── optimization/
│   │   └── black_litterman.py    # BL + Ledoit-Wolf + cvxpy
│   ├── analysis/
│   │   ├── ic_analysis.py        # IC / ICIR / t-stat
│   │   └── sector_analysis.py    # GICS, market caps, neutralisation
│   └── universe/
│       └── historical_constituents.py  # S&P 500 historique (anti-survivorship bias)
├── data/
│   ├── raw/                      # Prix, secteurs, market caps (ignoré par git)
│   └── processed/                # Signaux, backtest, picks (ignoré par git)
└── pyproject.toml
```
