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
    use_bl: bool = False             # Black-Litterman pour la construction du portefeuille
    bl_tau: float = 0.05             # incertitude sur le prior BL
    bl_confidence: float = 0.5      # confiance dans les vues (0=prior pur, 1=vues pures)
    use_hmm: bool = False            # pondérer les coefficients Fama-MacBeth par régime HMM
    hmm_sharpness: float = 2.0       # accentuation de la pondération (1=linéaire, 2=quadratique)
    long_only: bool = False          # pas de shorts — equal-weight top-N
    n_positions: int = 10            # nombre de titres longs (utilisé si long_only=True)
    sector_cap: int = 0              # max titres par secteur (0 = pas de limite)
    rank_buffer: float = 1.0         # un titre existant est conservé s'il reste dans top n×rank_buffer
    volatility_sizing: bool = False  # pondérer par 1/vol au lieu de equal-weight


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
        mcap_weights: "pd.Series | None" = None,
    ):
        self.returns = returns
        self.volume = volume
        self.benchmark_weights = benchmark_weights
        self.config = config or BacktestConfig()
        self.results = BacktestResults()
        self.universe_snapshots = universe_snapshots
        self.sector_map = sector_map
        self.mcap_weights = mcap_weights
        # Precompute sector Series for vectorized groupby in _sector_neutralize
        self._sector_series: pd.Series | None = None

        self.monthly_returns = (1 + returns).resample("MS").prod() - 1
        self.monthly_signals = {
            name: df.resample("MS").last()
            for name, df in signals.items()
        }
        self.signals = signals

    def run(self) -> BacktestResults:
        import time as _time
        rebal_dates = self._rebalancing_dates()
        prev_weights = pd.Series(0.0, index=self.returns.columns)

        # Pré-calcul des features de régime (partagées entre toutes les dates)
        _hmm_features = None
        if self.config.use_hmm:
            from src.regime.hmm import build_features
            _hmm_features = build_features(self.monthly_returns)

        portfolio_returns = {}
        portfolio_weights = {}
        portfolio_turnover = {}
        portfolio_costs = {}
        portfolio_factor_weights = {}

        n_total = len(rebal_dates)
        _t0 = _time.time()
        for _i, t in enumerate(rebal_dates):
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

            # --- Régime HMM (optionnel) ---
            month_weights = None
            if self.config.use_hmm and _hmm_features is not None:
                from src.regime.hmm import RegimeDetector, regime_sample_weights
                feat_window = _hmm_features[_hmm_features.index < t].iloc[-self.config.estimation_window:]
                if len(feat_window) >= 12:
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            det = RegimeDetector(n_states=2, random_state=42)
                            det.fit(feat_window)
                        proba_hist    = det.predict_proba(feat_window)
                        current_proba = proba_hist.iloc[-1]
                        month_weights = regime_sample_weights(
                            proba_hist, current_proba,
                            sharpness=self.config.hmm_sharpness,
                        )
                    except Exception:
                        month_weights = None

            # --- Fama-MacBeth + Ridge ---
            factor_weights = self._estimate_factor_weights(window_signals, window_returns, month_weights)
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

            # --- Optimisation ---
            self._current_t = t
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

            # Progress log every 20 rebalancings
            if (_i + 1) % 20 == 0 or (_i + 1) == n_total:
                elapsed = _time.time() - _t0
                pct = (_i + 1) / n_total
                eta = elapsed / pct * (1 - pct)
                print(f"  [{_i+1}/{n_total}]  {t.date()}  "
                      f"({pct:.0%})  écoulé {elapsed:.0f}s  ETA {eta:.0f}s")

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
        Neutralisation intra-secteur via pandas groupby (vectorisé).
        Soustrait la moyenne sectorielle et divise par l'écart-type (z-score).
        """
        if self.sector_map is None:
            return X

        # Build sector label Series aligned to X.index (cached across calls)
        if self._sector_series is None or not X.index.isin(self._sector_series.index).all():
            self._sector_series = pd.Series(
                {t: self.sector_map.get(t, "Unknown") for t in self.returns.columns}
            )
        sectors = self._sector_series.reindex(X.index)

        X = X.copy()
        for col in X.columns:
            vals = X[col]
            grp = vals.groupby(sectors)
            counts = grp.transform("count")
            means  = grp.transform("mean")
            stds   = grp.transform("std").fillna(0)

            eligible = counts >= 3
            X.loc[eligible & (stds > 0), col] = (
                (vals - means) / stds
            ).loc[eligible & (stds > 0)]
            X.loc[eligible & (stds == 0), col] = 0.0
        return X

    def _estimate_factor_weights(
        self,
        window_signals: dict,
        window_returns: pd.DataFrame,
        month_weights: pd.Series | None = None,
    ) -> pd.Series:
        """
        Fama-MacBeth : régression cross-sectionelle à chaque mois passé,
        moyenne (éventuellement pondérée par régime HMM) des coefficients.
        Ridge pour régulariser quand les signaux sont corrélés.
        """
        monthly_coefs = []
        coef_dates    = []

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
            coef_dates.append(date)

        if not monthly_coefs:
            return pd.Series(dtype=float)

        coef_df = pd.DataFrame(monthly_coefs, index=coef_dates)

        # Moyenne pondérée par régime si HMM activé
        if month_weights is not None and not month_weights.empty:
            w = month_weights.reindex(coef_df.index).fillna(month_weights.mean())
            w = w.clip(lower=0)
            w_sum = w.sum()
            if w_sum > 0:
                return (coef_df.multiply(w, axis=0)).sum() / w_sum

        return coef_df.mean()

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
        if self.config.long_only:
            n           = self.config.n_positions
            sector_cap  = self.config.sector_cap
            rank_buffer = self.config.rank_buffer

            ranked       = alpha_scores.sort_values(ascending=False)
            buffer_size  = int(n * rank_buffer)
            top_buffer   = set(ranked.head(buffer_size).index)

            # Conserver les positions existantes encore dans le buffer de rang
            existing = set(prev_weights[prev_weights > 1e-6].index)
            retained = existing & top_buffer

            # Décompte sectoriel des positions retenues
            sec_counts: dict[str, int] = {}
            if sector_cap > 0 and self.sector_map:
                for tk in retained:
                    sec = self.sector_map.get(tk, "Unknown")
                    sec_counts[sec] = sec_counts.get(sec, 0) + 1

            # Compléter jusqu'à n_positions par alpha décroissant
            selected = set(retained)
            for tk in ranked.index:
                if len(selected) >= n:
                    break
                if tk in selected:
                    continue
                if sector_cap > 0 and self.sector_map:
                    sec = self.sector_map.get(tk, "Unknown")
                    if sec_counts.get(sec, 0) >= sector_cap:
                        continue
                    sec_counts[sec] = sec_counts.get(sec, 0) + 1
                selected.add(tk)

            if not selected:
                return None

            selected_list = list(selected)

            if self.config.volatility_sizing:
                # Pondération inverse de la volatilité (risk parity au niveau titre)
                # Utilise les 12 derniers mois de rendements mensuels disponibles avant t
                t_cur = getattr(self, "_current_t", None)
                hist_idx = self.monthly_returns.index < t_cur if t_cur is not None \
                           else pd.Series([True] * len(self.monthly_returns.index))
                hist = self.monthly_returns[hist_idx].iloc[-12:]
                vol = hist[selected_list].std() * np.sqrt(12)
                vol = vol.replace(0, np.nan).dropna().clip(lower=0.02)
                if vol.empty:
                    inv_vol = pd.Series(1.0 / len(selected_list), index=selected_list)
                else:
                    inv_vol = 1.0 / vol
                    # Compléter les titres sans historique de vol avec la médiane
                    missing = [t for t in selected_list if t not in inv_vol.index]
                    if missing:
                        inv_vol = pd.concat([inv_vol,
                            pd.Series(inv_vol.median(), index=missing)])
                    inv_vol = inv_vol.reindex(selected_list).fillna(inv_vol.median())
                raw_w = inv_vol / inv_vol.sum()
                # Appliquer le plafond max_position et renormaliser
                cap = self.config.max_position if self.config.max_position < 1.0 else 0.20
                capped = raw_w.clip(upper=cap)
                w_vals = capped / capped.sum()
            else:
                w_vals = pd.Series(1.0 / len(selected_list), index=selected_list)

            w = pd.Series(0.0, index=alpha_scores.index)
            w[selected_list] = w_vals.reindex(selected_list).fillna(0)
            return w

        if self.config.use_bl:
            from src.optimization.black_litterman import black_litterman_weights
            mkt_w = self.mcap_weights if self.mcap_weights is not None else self.benchmark_weights
            return black_litterman_weights(
                returns_hist=self.monthly_returns,
                alpha_scores=alpha_scores,
                market_weights=mkt_w,
                tau=self.config.bl_tau,
                risk_aversion=2.5,
                view_confidence=self.config.bl_confidence,
                max_position=self.config.max_position,
                max_short=self.config.max_short,
                max_turnover=self.config.max_turnover,
                prev_weights=prev_weights,
            )

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
