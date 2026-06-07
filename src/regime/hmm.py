"""
Détection de régime de marché par HMM gaussien.

Architecture :
  - 2 états cachés (bull/expansion, bear/contraction)
  - 3 features observables calculées depuis les prix : rendement EW mensuel,
    volatilité réalisée 12 mois, dispersion cross-sectionnelle
  - Intégration dans le moteur de backtest via Ridge pondéré par régime :
    au lieu d'estimer les factor weights sur toutes les périodes passées de façon
    égale, chaque mois est pondéré par la similarité de son régime avec le régime
    actuel. Cela permet au modèle d'utiliser principalement les données historiques
    dont le contexte macro ressemble au contexte actuel.

Pourquoi ça marche :
  Le leadership des facteurs est cyclique — momentum domine en bull market,
  low_volatility et reversal dominent en bear. Un Ridge entraîné sur l'ensemble
  des régimes mélange ces dynamiques opposées et obtient un coefficient moyen
  qui sous-performe dans les deux régimes. Le Ridge pondéré par régime concentre
  l'apprentissage sur les périodes similaires à aujourd'hui.

Contrainte walk-forward :
  À chaque date t, le HMM est entraîné uniquement sur les données [t-window, t-1].
  La probabilité de régime de la période t-1 sert de proxy pour le régime actuel.
  Aucun lookahead possible.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(monthly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les 3 features de régime depuis les rendements mensuels bruts.

    Features :
      ew_return     — rendement mensuel equal-weight (direction du marché)
      ew_vol_12m    — volatilité réalisée 12 mois de l'EW (amplitude du risque)
      cs_dispersion — écart-type cross-sectionnel des rendements (turbulence)

    Toutes standardisées (z-score rolling sur la fenêtre disponible).
    """
    ew_ret  = monthly_returns.mean(axis=1)
    ew_vol  = ew_ret.rolling(12, min_periods=6).std() * np.sqrt(12)
    cs_disp = monthly_returns.std(axis=1)

    features = pd.DataFrame({
        "ew_return":      ew_ret,
        "ew_vol_12m":     ew_vol,
        "cs_dispersion":  cs_disp,
    }).dropna()

    return features


class RegimeDetector:
    """
    HMM gaussien 2 états pour la détection de régime de marché.

    Usage walk-forward :
      detector = RegimeDetector(n_states=2)
      detector.fit(features_window)           # entraîne sur données passées
      proba = detector.predict_proba(features) # P(état | observations)
      current_proba = proba.iloc[-1]          # régime le plus récent
    """

    def __init__(self, n_states: int = 2, random_state: int = 42):
        self.n_states     = n_states
        self.random_state = random_state
        self._model       = None
        self._scaler      = StandardScaler()
        self._bull_state  = 0   # indice de l'état "bull" (identifié après fit)
        self.is_fitted    = False

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        """
        Entraîne le HMM sur la fenêtre de features fournie.
        Identifie automatiquement quel état correspond au bull/bear
        d'après le rendement EW moyen par état.
        """
        from hmmlearn.hmm import GaussianHMM

        X = self._scaler.fit_transform(features.values)

        model = GaussianHMM(
            n_components    = self.n_states,
            covariance_type = "full",
            n_iter          = 200,
            tol             = 1e-4,
            random_state    = self.random_state,
            init_params     = "stmc",
        )
        model.fit(X)
        self._model   = model
        self.is_fitted = True

        # Labéliser : état avec rendement EW moyen le plus élevé = bull
        states       = model.predict(X)
        ew_col       = features.columns.get_loc("ew_return")
        state_means  = [features.values[states == s, ew_col].mean()
                        for s in range(self.n_states)]
        self._bull_state = int(np.argmax(state_means))

        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne P(état | observations) pour chaque mois.
        Colonne 0 = P(bull), colonne 1 = P(bear) — indépendamment de l'ordre interne.
        """
        if not self.is_fitted:
            raise RuntimeError("Appelle fit() avant predict_proba().")

        X      = self._scaler.transform(features.values)
        proba  = self._model.predict_proba(X)   # (T, n_states)

        bear_state = 1 - self._bull_state   # valide uniquement pour n_states=2
        df = pd.DataFrame({
            "p_bull": proba[:, self._bull_state],
            "p_bear": proba[:, bear_state],
        }, index=features.index)

        return df

    def current_regime(self, features: pd.DataFrame) -> dict:
        """
        Retourne le régime du dernier mois disponible.
        {'state': 'bull'/'bear', 'p_bull': float, 'p_bear': float}
        """
        proba = self.predict_proba(features)
        last  = proba.iloc[-1]
        return {
            "state":  "bull" if last["p_bull"] >= 0.5 else "bear",
            "p_bull": float(last["p_bull"]),
            "p_bear": float(last["p_bear"]),
        }


def regime_sample_weights(
    proba_history: pd.DataFrame,
    current_proba: pd.Series,
    sharpness: float = 2.0,
) -> pd.Series:
    """
    Calcule les poids de chaque mois historique pour le Ridge pondéré par régime.

    Logique : w_t = dot(P_current, P_t)^sharpness
      - Si mois t était dans le même régime que maintenant → poids élevé
      - Si mois t était dans le régime opposé → poids faible
    Le paramètre sharpness contrôle l'accentuation (1 = linéaire, 2 = quadratique).
    Un sharpness élevé renforce la spécialisation par régime.

    Retourne une Series(index=proba_history.index, values=poids normalisés).
    """
    # dot product entre vecteur de proba courant et chaque mois historique
    current = current_proba.values  # shape (n_states,)
    hist    = proba_history.values  # shape (T, n_states)
    sim     = (hist @ current) ** sharpness
    sim     = np.clip(sim, 1e-6, None)
    weights = sim / sim.sum() * len(sim)   # normalisé autour de 1
    return pd.Series(weights, index=proba_history.index)
