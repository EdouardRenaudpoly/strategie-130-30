"""
Signal SUE — Standardized Unexpected Earnings (Bernard & Thomas 1989).

Logique :
  - EPS trimestriel = entrées XBRL sur ~90 jours (10-Q + 10-K)
  - Surprise YoY    = EPS_q  −  EPS_{même trimestre, 1 an avant}
  - SUE             = surprise / std(8 dernières surprises)
  - Disponibilité   = date de dépôt + 5 jours (zéro lookahead)
  - Forward-fill    : le signal reste actif jusqu'au trimestre suivant

Pourquoi SUE fonctionne :
  Le Post-Earnings Announcement Drift (PEAD) est l'une des anomalies les plus
  robustes en finance empirique. Les marchés sous-réagissent aux surprises de
  résultats : un titre avec un SUE élevé continue de surperformer 60 jours
  après l'annonce. Le signal est orthogonal au momentum prix car il capture
  l'information fondamentale comptable, pas la tendance de cours.

Source : SEC EDGAR XBRL (même infrastructure que gross_profitability/accruals).
  Concept : EarningsPerShareBasic ou EarningsPerShareDiluted
  Unité : USD/shares  — filtre sur périodes ~90 jours (trimestrielles pures)
"""

import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

_SEC_EMAIL = __import__("os").environ.get("SEC_USER_AGENT_EMAIL", "your-email@example.com")
_HEADERS   = {"User-Agent": f"Research {_SEC_EMAIL}"}
_CACHE     = Path("data/raw/xbrl")
_RATE      = 0.12
_CACHE_TTL_DAYS = 30


def _get(url: str) -> requests.Response | None:
    time.sleep(_RATE)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def _fetch_facts(cik: str) -> dict:
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


def _extract_quarterly_eps(facts: dict) -> pd.DataFrame:
    """
    Extrait les EPS trimestriels purs depuis XBRL.

    Stratégie de sélection :
      1. Préférer EarningsPerShareBasic ; fallback sur EarningsPerShareDiluted.
      2. Garder uniquement les entrées dont la période est ~90 jours (±30 j)
         → élimine les cumuls YTD et les données annuelles.
      3. Pour chaque trimestre, garder uniquement le premier filing
         (pas les amendements) pour utiliser la date de publication originale.

    Retourne DataFrame(end_date, val, filed_date) trié par end_date.
    """
    for concept in ("EarningsPerShareBasic", "EarningsPerShareDiluted"):
        try:
            entries = facts["facts"]["us-gaap"][concept]["units"]["USD/shares"]
            break
        except KeyError:
            continue
    else:
        return pd.DataFrame(columns=["end", "val", "filed"])

    rows = []
    for e in entries:
        if e.get("form") not in ("10-Q", "10-K"):
            continue
        try:
            start  = pd.Timestamp(e["start"])
            end    = pd.Timestamp(e["end"])
            filed  = pd.Timestamp(e["filed"])
            val    = float(e["val"])
        except (KeyError, ValueError, TypeError):
            continue

        period_days = (end - start).days
        # Période trimestrielle : entre 60 et 120 jours
        if not (60 <= period_days <= 120):
            continue

        rows.append({"end": end, "val": val, "filed": filed})

    if not rows:
        return pd.DataFrame(columns=["end", "val", "filed"])

    df = pd.DataFrame(rows)
    df["end"]   = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    # Pour chaque fin de trimestre, garder le premier filing
    df = df.sort_values("filed").drop_duplicates(subset="end", keep="first")
    return df.sort_values("end").reset_index(drop=True)


def _compute_sue_series(quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les surprises YoY et le SUE pour une série de résultats trimestriels.

    YoY surprise : comparer chaque trimestre au même trimestre de l'année précédente
    (end_date à ±45 jours de end_date − 365 jours).
    SUE = surprise / rolling_std(8 dernières surprises), min_periods=4.

    Retourne DataFrame(end, surprise, sue, filed) trié par end.
    """
    if len(quarterly) < 5:
        return pd.DataFrame(columns=["end", "surprise", "sue", "filed"])

    quarterly = quarterly.sort_values("end").reset_index(drop=True)
    surprises = []

    for i, row in quarterly.iterrows():
        target_date = row["end"] - pd.DateOffset(years=1)
        # Trouver le trimestre de l'année précédente (±45 jours)
        mask = (quarterly["end"] - target_date).abs() <= pd.Timedelta(days=45)
        prior = quarterly[mask & (quarterly.index < i)]
        if prior.empty:
            continue
        prior_row = prior.iloc[-1]
        surprise  = row["val"] - prior_row["val"]
        surprises.append({
            "end":      row["end"],
            "val":      row["val"],
            "surprise": surprise,
            "filed":    row["filed"],
        })

    if not surprises:
        return pd.DataFrame(columns=["end", "surprise", "sue", "filed"])

    df = pd.DataFrame(surprises)
    rolling_std = df["surprise"].rolling(window=8, min_periods=4).std()
    df["sue"]   = df["surprise"] / rolling_std.replace(0, np.nan)
    return df[["end", "surprise", "sue", "filed"]].dropna(subset=["sue"])


def _build_monthly_sue(
    sue_series: pd.DataFrame,
    all_months: pd.DatetimeIndex,
    lag_days: int = 5,
) -> pd.Series:
    """
    Convertit une série de SUE trimestriels en signal mensuel.
    Chaque valeur devient disponible à filed + lag_days, puis forward-fill.
    """
    if sue_series.empty:
        return pd.Series(index=all_months, dtype=float)

    monthly = pd.Series(index=all_months, dtype=float)
    for _, row in sue_series.iterrows():
        avail  = row["filed"] + pd.Timedelta(days=lag_days)
        future = all_months[all_months >= avail]
        if len(future) > 0:
            monthly.loc[future[0]] = float(row["sue"])

    return monthly.ffill()


def compute_sue(
    tickers: list[str],
    cik_map: dict[str, str],
    start_year: int = 2013,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Calcule le signal SUE mensuel pour tous les tickers.

    Retourne DataFrame(dates × tickers), fréquence mensuelle début de mois.
    Les valeurs sont le SUE le plus récent disponible (forward-fillé).
    Signal positif → surprise haussière → alpha positif attendu.
    """
    all_months = pd.date_range(
        start=f"{start_year}-01-01",
        end=pd.Timestamp.today(),
        freq="MS",
    )

    sue_dict: dict[str, pd.Series] = {}

    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker)
        if cik is None:
            continue

        if verbose and i % 50 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}...")

        facts     = _fetch_facts(cik)
        quarterly = _extract_quarterly_eps(facts)
        if quarterly.empty:
            continue

        sue_series = _compute_sue_series(quarterly)
        if sue_series.empty:
            continue

        monthly = _build_monthly_sue(sue_series, all_months)
        if monthly.notna().sum() >= 12:
            sue_dict[ticker] = monthly

    return pd.DataFrame(sue_dict, index=all_months)
