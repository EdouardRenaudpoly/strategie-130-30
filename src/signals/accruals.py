"""
Signal accruals — Sloan (1996)

Logique :
  Profit comptable = Cash Flow Opérationnel + Accruals
  Les accruals = ajustements non-cash (provisions, amortissements inversés, etc.)
  Une entreprise avec des accruals élevés a des profits moins durables → sous-performance future.

  Accruals = Net Income - Cash Flow from Operations
  Signal   = -Accruals / Total Assets  (inversé : bas accruals = bon signe)

  IC documenté : ~0.03-0.05, corrélation négative avec momentum → bonne diversification.

Source : SimFin API (gratuit avec inscription, 1 jour de délai sur données historiques)
         https://app.simfin.com/api/v3/  → créer compte → copier la clé dans SIMFIN_API_KEY

Lag de publication : on décale de 45 jours (Q) et 90 jours (annuel) pour éviter look-ahead.
"""

import os
import requests
import pandas as pd
from pathlib import Path

_BASE_URL = "https://backend.simfin.com/api/v3"
_CACHE_DIR = Path("data/raw/simfin")


def _get_api_key() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("SIMFIN_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Clé SimFin manquante.\n"
            "1. Créer un compte gratuit sur https://app.simfin.com\n"
            "2. Copier ta clé API\n"
            "3. Ajouter dans .env à la racine du projet : SIMFIN_API_KEY=ta_cle"
        )
    return key


def _simfin_get(endpoint: str, params: dict) -> list[dict]:
    """Appel API SimFin v3 avec cache par paramètres."""
    import json
    cache_key = endpoint.replace("/", "_") + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    headers = {"Authorization": f"Bearer {_get_api_key()}"}
    r = requests.get(f"{_BASE_URL}{endpoint}", params=params, headers=headers, timeout=60)
    r.raise_for_status()

    data = r.json()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def _fetch_bulk_statement(statement: str, period: str, year: int) -> pd.DataFrame:
    """
    Télécharge un état financier en bulk pour toutes les entreprises US.
    statement : 'pl' (P&L), 'cf' (Cash Flow), 'bs' (Balance Sheet)
    period    : 'quarterly', 'annual'
    """
    data = _simfin_get(
        f"/companies/statements/bulk",
        {"statement": statement, "type": period, "fyear": year, "market": "us"}
    )

    if not data or not isinstance(data, list):
        return pd.DataFrame()

    # SimFin renvoie [{ticker, statements: [{fiscalPeriod, fiscalYear, data: {col: val}}]}]
    rows = []
    for company in data:
        ticker = company.get("ticker", "")
        for stmt in company.get("statements", []):
            row = {"ticker": ticker}
            row["fiscal_period"] = stmt.get("fiscalPeriod")
            row["fiscal_year"]   = stmt.get("fiscalYear")
            row["publish_date"]  = stmt.get("publishDate")
            row.update(stmt.get("data", {}))
            rows.append(row)

    return pd.DataFrame(rows)


def compute_accruals_signal(
    tickers: list[str],
    start_year: int = 2012,
    end_year: int | None = None,
) -> pd.DataFrame:
    """
    Calcule le signal accruals pour chaque ticker, fréquence mensuelle.

    Accruals = Net Income - Operating Cash Flow  (normalisé par Total Assets)
    Signal   = -accruals  (bas accruals → score élevé → bon signal)

    Le signal est daté à la date de publication + 5 jours de sécurité pour
    éviter tout look-ahead bias.

    Retourne DataFrame (dates, tickers) de fréquence mensuelle.
    """
    import datetime
    end_year = end_year or datetime.date.today().year

    pl_frames, cf_frames, bs_frames = [], [], []

    for year in range(start_year, end_year + 1):
        for period in ("quarterly",):
            pl = _fetch_bulk_statement("pl", period, year)
            cf = _fetch_bulk_statement("cf", period, year)
            bs = _fetch_bulk_statement("bs", period, year)
            if not pl.empty:
                pl_frames.append(pl)
            if not cf.empty:
                cf_frames.append(cf)
            if not bs.empty:
                bs_frames.append(bs)

    if not pl_frames:
        return pd.DataFrame()

    pl_all = pd.concat(pl_frames, ignore_index=True)
    cf_all = pd.concat(cf_frames, ignore_index=True)
    bs_all = pd.concat(bs_frames, ignore_index=True)

    # Colonnes clés SimFin (noms standardisés)
    # Net Income : "Net Income"
    # Operating Cash Flow : "Net Cash from Operating Activities"
    # Total Assets : "Total Assets"

    merge_keys = ["ticker", "fiscal_period", "fiscal_year", "publish_date"]

    merged = (
        pl_all[merge_keys + ["Net Income"]]
        .merge(
            cf_all[merge_keys + ["Net Cash from Operating Activities"]],
            on=merge_keys, how="inner"
        )
        .merge(
            bs_all[merge_keys + ["Total Assets"]],
            on=merge_keys, how="inner"
        )
    )

    merged = merged[merged["ticker"].isin(tickers)].copy()
    merged["publish_date"] = pd.to_datetime(merged["publish_date"], errors="coerce")
    merged = merged.dropna(subset=["publish_date"])

    # Délai de sécurité : +5 jours après publication pour être sûr d'avoir les données
    merged["signal_date"] = merged["publish_date"] + pd.Timedelta(days=5)

    # Accruals normalisés
    merged["net_income"] = pd.to_numeric(merged["Net Income"], errors="coerce")
    merged["cfo"]        = pd.to_numeric(merged["Net Cash from Operating Activities"], errors="coerce")
    merged["assets"]     = pd.to_numeric(merged["Total Assets"], errors="coerce")

    merged = merged.dropna(subset=["net_income", "cfo", "assets"])
    merged = merged[merged["assets"] > 0]

    merged["accruals_normalized"] = (merged["net_income"] - merged["cfo"]) / merged["assets"]
    # Signal inversé : bas accruals = bon signal
    merged["signal"] = -merged["accruals_normalized"]

    # Pivot vers format (dates mensuelles, tickers)
    # Pour chaque mois, utiliser la valeur la plus récente disponible (forward-fill)
    all_months = pd.date_range(
        start=f"{start_year}-01-01",
        end=pd.Timestamp.today(),
        freq="MS"
    )

    result = pd.DataFrame(index=all_months, columns=tickers, dtype=float)

    for ticker in tickers:
        t_data = merged[merged["ticker"] == ticker].sort_values("signal_date")
        if t_data.empty:
            continue
        t_series = t_data.set_index("signal_date")["signal"]
        t_series = t_series[~t_series.index.duplicated(keep="last")]
        t_series = t_series.reindex(all_months, method="ffill")
        result[ticker] = t_series.values

    return result
