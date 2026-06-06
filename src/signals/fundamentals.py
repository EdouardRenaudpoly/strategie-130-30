"""
Signaux fondamentaux via SEC EDGAR XBRL — gratuit, officiel, stable depuis 2009.

Signaux produits :
  gross_profitability : Gross Profit / Total Assets (Novy-Marx 2013)
                        IC ~0.04, orthogonal au momentum
  accruals            : -(Net Income - CFO) / Total Assets (Sloan 1996)
                        IC ~0.03-0.05, corrélation négative avec momentum

Source : https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  - Données annuelles 10-K depuis ~2009
  - Champ `filed` = date de disponibilité réelle → zéro look-ahead si on l'utilise
  - Aucune clé API requise, juste un User-Agent valide

Look-ahead bias :
  On utilise `filed` + 5 jours comme date de disponibilité du signal.
  Un ratio publié le 2024-02-26 est disponible dans le modèle à partir du 2024-03-02.
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path

_SEC_EMAIL = os.environ.get("SEC_USER_AGENT_EMAIL", "your-email@example.com")
_HEADERS  = {"User-Agent": f"Research {_SEC_EMAIL}"}
_CACHE    = Path("data/raw/xbrl")
_RATE     = 0.12   # 10 req/s max (SEC policy)
_CACHE_TTL_DAYS = 30   # recharger les facts tous les 30 jours


def _get(url: str) -> requests.Response | None:
    time.sleep(_RATE)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def _fetch_facts(cik: str) -> dict:
    """
    Retourne les facts XBRL d'une entreprise.
    Cache local de 30 jours — recharge automatiquement si périmé.
    """
    cache_file = _CACHE / f"{cik}.json"
    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < _CACHE_TTL_DAYS:
            return json.loads(cache_file.read_text())

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    r = _get(url)
    if r is None:
        return {}

    data = r.json()
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def _extract_annual(facts: dict, concept: str) -> pd.DataFrame:
    """
    Extrait les valeurs annuelles (10-K) d'un concept XBRL.
    Retourne DataFrame (end_date, val, filed_date).
    Pour chaque période, garde uniquement le premier dépôt (avant tout amendement).
    """
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return pd.DataFrame(columns=["end", "val", "filed"])

    df = pd.DataFrame(entries)
    df = df[df["form"] == "10-K"].copy()
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])

    df["end"]   = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["val"]   = pd.to_numeric(df["val"], errors="coerce")

    # Pour chaque période, garder le premier filing (pas les amendements)
    df = df.sort_values("filed").drop_duplicates(subset="end", keep="first")

    return df[["end", "val", "filed"]].sort_values("end").reset_index(drop=True)


def _build_monthly_signal(
    series: pd.DataFrame,
    all_months: pd.DatetimeIndex,
    lag_days: int = 5,
) -> pd.Series:
    """
    Convertit une série de valeurs annuelles en signal mensuel.
    Chaque valeur devient disponible à `filed + lag_days`.
    Forward-fill jusqu'au prochain rapport annuel.
    """
    if series.empty:
        return pd.Series(index=all_months, dtype=float)

    # Date de disponibilité = filed + buffer anti-look-ahead
    series = series.copy()
    series["available"] = series["filed"] + pd.Timedelta(days=lag_days)

    # Placer chaque valeur à sa date de disponibilité dans le calendrier mensuel
    monthly = pd.Series(index=all_months, dtype=float)
    for _, row in series.iterrows():
        try:
            val = float(row["val"])
        except (TypeError, ValueError):
            continue
        if pd.isna(val):
            continue
        avail = row["available"]
        if pd.isna(avail):
            continue
        future = all_months[all_months >= avail]
        if len(future) > 0:
            monthly.loc[future[0]] = val

    return monthly.ffill()


def compute_fundamentals(
    tickers: list[str],
    cik_map: dict[str, str],
    start_year: int = 2012,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Calcule gross_profitability et accruals pour tous les tickers.

    Retourne {signal_name: DataFrame(dates x tickers)}.
    """
    all_months = pd.date_range(
        start=f"{start_year}-01-01",
        end=pd.Timestamp.today(),
        freq="MS",
    )

    gp_series:   dict[str, pd.Series] = {}
    acc_series:  dict[str, pd.Series] = {}
    agr_series:  dict[str, pd.Series] = {}

    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker)
        if cik is None:
            continue

        if verbose and i % 50 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}...")

        facts = _fetch_facts(cik)
        if not facts:
            continue

        gross_profit = _extract_annual(facts, "GrossProfit")
        assets       = _extract_annual(facts, "Assets")
        net_income   = _extract_annual(facts, "NetIncomeLoss")
        cfo          = _extract_annual(facts, "NetCashProvidedByUsedInOperatingActivities")

        # --- Gross Profitability ---
        if not gross_profit.empty and not assets.empty:
            merged = gross_profit.merge(assets, on="end", suffixes=("_gp", "_a"))
            merged["filed"] = merged[["filed_gp", "filed_a"]].max(axis=1)
            merged["val"]   = merged["val_gp"] / merged["val_a"].replace(0, float("nan"))
            merged = merged.dropna(subset=["val"])
            gp_series[ticker] = _build_monthly_signal(
                merged[["end", "val", "filed"]], all_months
            )

        # --- Asset Growth (inversé : croissance élevée → mauvais signal) ---
        # Cooper et al. 2008 : les entreprises qui grossissent vite sous-performent
        if not assets.empty and len(assets) >= 2:
            ag = assets.copy()
            ag["val"] = -(ag["val"] / ag["val"].shift(1) - 1)  # inversé
            ag = ag.dropna(subset=["val"])
            ag["val"] = pd.to_numeric(ag["val"], errors="coerce")
            ag = ag.dropna(subset=["val"])
            agr_series[ticker] = _build_monthly_signal(ag[["end", "val", "filed"]], all_months)

        # --- Accruals (inversés : bas accruals = bon signal) ---
        if not net_income.empty and not cfo.empty and not assets.empty:
            merged = net_income.merge(cfo, on="end", suffixes=("_ni", "_cfo"))
            merged = merged.merge(assets, on="end")
            merged["filed"] = merged[["filed_ni", "filed_cfo", "filed"]].max(axis=1)
            accruals = (merged["val_ni"] - merged["val_cfo"]) / merged["val"].replace(0, float("nan"))
            merged["val"] = pd.to_numeric(-accruals, errors="coerce")
            merged = merged.dropna(subset=["val"])
            acc_series[ticker] = _build_monthly_signal(
                merged[["end", "val", "filed"]], all_months
            )

    return {
        "gross_profitability": pd.DataFrame(gp_series,  index=all_months),
        "accruals":            pd.DataFrame(acc_series, index=all_months),
        "asset_growth":        pd.DataFrame(agr_series, index=all_months),
    }
