import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.backtest.frictions import transaction_cost, borrow_cost
from src.backtest.validator import assert_no_lookahead, validate_signals_alignment


@dataclass
class BacktestConfig:
    estimation_window: int = 36      # mois de données pour Fama-MacBeth
    ridge_alpha: float = 1.0         # régularisation Ridge
    max_position: float = 0.05       # max poids par titre
    max_short: float = 0.05          # max short par titre
    max_te: float = 0.06             # tracking error max vs benchmark
    max_turnover: float = 0.30       # turnover max par rebalancement
    transaction_cost_bps: float = 7  # coûts de transaction en bps
    train_end: str = "2020-01-01"    # frontière train / test
    sector_neutral: bool = True      # neutralisation intra-secteur des signaux


@dataclass
class BacktestResults:
    returns: pd.Series = field(default_factory=pd.Series)
    weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    turnover: pd.Series = field(default_factory=pd.Series)
    factor_weights: pd.DataFrame = field(default_factory=pd.DataFrame)
    costs: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    def __init__(
        self,
        returns: pd.DataFrame,
        signals: dict[str, pd.DataFrame],
        volume: pd.DataFrame,
        benchmark_weights: pd.Series,
        config: BacktestConfig | None = None,
        universe_snapshots: "pd.DataFrame | None" = None,
        sector_map: "dict[str, str] | None" = None,
    ):
        self.returns = returns
        self.volume = volume
        self.benchmark_weights = benchmark_weights
        self.config = config or BacktestConfig()
        self.results = BacktestResults()
        self.universe_snapshots = universe_snapshots
        self.sector_map = sector_map

        self.monthly_returns = (1 + returns).resample("MS").prod() - 1
        self.monthly_signals = {
            name: df.resample("MS").last()
            for name, df in signals.items()
        }
        self.signals = signals

    def run(self) -> BacktestResults:
        rebal_dates = self._rebalancing_dates()
        prev_weights = pd.Series(0.0, index=self.returns.columns)

        portfolio_returns = {}
        portfolio_weights = {}
        portfolio_turnover = {}
        portfolio_costs = {}
        portfolio_factor_weights = {}

        for t in rebal_dates:
            # --- Univers historiquement correct à la date t ---
            if self.universe_snapshots is not None:
                from src.universe.historical_constituents import get_universe_at_date
                valid_tickers = get_universe_at_date(self.universe_snapshots, t)
                valid_tickers = [tk for tk in valid_tickers if tk in self.returns.columns]
                if len(valid_tickers) < 50:
                    continue
            else:
                valid_tickers = self.returns.columns.tolist()

            # --- Assertion look-ahead ---
            assert_no_lookahead(t, self.returns, self.signals)

            # Filtrer returns et signaux sur l'univers valide
            monthly_returns_t = self.monthly_returns[valid_tickers]
            monthly_signals_t = {
                name: df[valid_tickers] if all(c in df.columns for c in valid_tickers)
                      else df.reindex(columns=valid_tickers)
                for name, df in self.monthly_signals.items()
            }

            # --- Fenêtre d'estimation mensuelle (données passées uniquement) ---
            window_returns = monthly_returns_t[monthly_returns_t.index < t].iloc[-self.config.estimation_window:]
            window_signals = {
                name: df[df.index < t].iloc[-self.config.estimation_window:]
                for name, df in monthly_signals_t.items()
            }

            if len(window_returns) < 12:
                # Pas assez d'historique pour estimer le modèle
                continue

            # --- Fama-MacBeth + Ridge ---
            factor_weights = self._estimate_factor_weights(window_signals, window_returns)
            portfolio_factor_weights[t] = factor_weights

            # --- Score alpha au temps t ---
            current_signals = self._get_current_signals(t, valid_tickers)
            if current_signals is None:
                continue

            # Aligner les colonnes du signal sur les facteurs estimés
            common_factors = current_signals.columns.intersection(factor_weights.index)
            if common_factors.empty:
                continue

            # Neutralisation sectorielle + z-score (même pipeline que l'estimation)
            X_raw = current_signals[common_factors]
            if self.config.sector_neutral:
                X_raw = self._sector_neutralize(X_raw)
            X_scaled = pd.DataFrame(
                StandardScaler().fit_transform(X_raw),
                index=X_raw.index,
                columns=X_raw.columns,
            )
            alpha_scores = X_scaled @ factor_weights[common_factors]

            # --- Optimisation 130/30 ---
            new_weights = self._optimize(alpha_scores, prev_weights)
            if new_weights is None:
                continue

            # --- Frictions ---
            tc = transaction_cost(prev_weights, new_weights, self.config.transaction_cost_bps)
            bc = borrow_cost(new_weights, self.volume.loc[t] if t in self.volume.index else None)
            total_cost = tc + bc

            # --- Rendement mensuel du portefeuille ---
            next_dates = self.monthly_returns.index[self.monthly_returns.index > t]
            if len(next_dates) == 0:
                break

            next_t = next_dates[0]
            period_returns = self.monthly_returns.loc[next_t]
            common = new_weights.index.intersection(period_returns.index)

            gross_return = (new_weights[common] * period_returns[common]).sum()
            net_return = gross_return - total_cost

            portfolio_returns[t] = net_return
            portfolio_weights[t] = new_weights
            portfolio_turnover[t] = np.abs(new_weights - prev_weights).sum()
            portfolio_costs[t] = total_cost

            prev_weights = new_weights

        self.results.returns = pd.Series(portfolio_returns)
        self.results.weights = pd.DataFrame(portfolio_weights).T
        self.results.turnover = pd.Series(portfolio_turnover)
        self.results.costs = pd.Series(portfolio_costs)
        self.results.factor_weights = pd.DataFrame(portfolio_factor_weights).T

        return self.results

    def _rebalancing_dates(self) -> list:
        # Premier jour de bourse de chaque mois
        return (
            self.returns
            .resample("MS")
            .first()
            .dropna(how="all")
            .index
            .tolist()
        )

    def _sector_neutralize(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Neutralisation intra-secteur : pour chaque signal, soustrait la moyenne
        du secteur et divise par l'écart-type du secteur (z-score intra-secteur).
        Les tickers sans secteur connu sont conservés tels quels.
        """
        if self.sector_map is None:
            return X
        X = X.copy()
        for col in X.columns:
            for sector in set(self.sector_map.values()):
                members = [t for t in X.index if self.sector_map.get(t) == sector]
                if len(members) < 3:
                    continue
                vals = X.loc[members, col].dropna()
                if len(vals) < 2:
                    continue
                std = vals.std()
                if std > 0:
                    X.loc[vals.index, col] = (vals - vals.mean()) / std
                else:
                    X.loc[vals.index, col] = 0.0
        return X

    def _estimate_factor_weights(
        self, window_signals: dict, window_returns: pd.DataFrame
    ) -> pd.Series:
        """
        Fama-MacBeth : régression cross-sectionelle à chaque mois passé,
        moyenne des coefficients → poids des facteurs.
        Ridge pour régulariser quand les signaux sont corrélés.
        """
        monthly_coefs = []

        for date in window_returns.index:
            # Rendement forward (mois suivant cette date dans la fenêtre)
            future_dates = window_returns.index[window_returns.index > date]
            if len(future_dates) == 0:
                continue
            y = window_returns.loc[future_dates[0]]

            # Signaux à cette date
            X_dict = {}
            for name, df in window_signals.items():
                if date in df.index:
                    X_dict[name] = df.loc[date]

            if not X_dict:
                continue

            X = pd.DataFrame(X_dict).dropna()
            y = y[X.index].dropna()
            X = X.loc[y.index]

            if len(y) < 50:
                continue

            # Neutralisation sectorielle (optionnel)
            if self.config.sector_neutral:
                X = self._sector_neutralize(X)

            # Z-score cross-sectionnel pour comparabilité entre signaux
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X), index=X.index, columns=X.columns
            )

            ridge = Ridge(alpha=self.config.ridge_alpha)
            ridge.fit(X_scaled, y)
            monthly_coefs.append(pd.Series(ridge.coef_, index=X.columns))

        if not monthly_coefs:
            return pd.Series(dtype=float)

        # Moyenne des coefficients mensuels — procédure Fama-MacBeth
        return pd.DataFrame(monthly_coefs).mean()

    def _get_current_signals(
        self, t: pd.Timestamp, valid_tickers: list[str] | None = None
    ) -> pd.DataFrame | None:
        """Signaux à la dernière date disponible strictement avant t."""
        frames = {}
        for name, df in self.signals.items():
            past = df.loc[:t].iloc[:-1]
            if past.empty:
                return None
            row = past.iloc[-1]
            if valid_tickers is not None:
                row = row.reindex(valid_tickers)
            frames[name] = row

        result = pd.DataFrame(frames).dropna()
        return result if not result.empty else None

    def _optimize(
        self, alpha_scores: pd.Series, prev_weights: pd.Series
    ) -> pd.Series | None:
        """Délègue à src/optimization — importé ici pour éviter circularité."""
        try:
            import cvxpy as cp

            tickers = alpha_scores.index
            n = len(tickers)
            # Décomposition explicite long/short — cp.pos()/cp.neg() dans une
            # égalité violent les règles DCP de cvxpy
            w_long = cp.Variable(n, nonneg=True)
            w_short = cp.Variable(n, nonneg=True)
            w = w_long - w_short

            prev = prev_weights.reindex(tickers).fillna(0).values
            alpha = alpha_scores.values

            constraints = [
                cp.sum(w_long) == 1.30,
                cp.sum(w_short) == 0.30,
                cp.sum(w) == 1,
                w_long <= self.config.max_position,
                w_short <= self.config.max_short,
            ]

            # Turnover uniquement si portfolio existant — inapplicable au 1er rebalancement
            # car sum(|w - 0|) = 1.60 pour un 130/30, ce qui violerait la contrainte
            if np.abs(prev).sum() > 1e-6:
                constraints.append(cp.sum(cp.abs(w - prev)) <= self.config.max_turnover)

            objective = cp.Maximize(alpha @ w)
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status not in ("optimal", "optimal_inaccurate"):
                return None

            return pd.Series(w.value, index=tickers)

        except Exception:
            return None
