"""
Signal insider trading — Form 4 SEC EDGAR

Logique :
  - Code P (open market purchase) = signal positif fort
  - Code S (open market sale)     = signal négatif faible (ventes souvent pour diversification)
  - Codes A/M/F/G                 = exclus (grants, options, taxes — non informatifs)

Score mensuel par ticker :
  score = (Σ dollar_purchases - 0.5 * Σ dollar_sales) / avg_price_est
  Normalisé cross-sectionnellement avant utilisation dans le modèle.

Sources : SEC EDGAR submissions API + individual Form 4 XML
Limite de débit : 10 req/s (SEC policy) → throttle à 0.12s
"""

import os
import time
import json
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path

_SEC_EMAIL = os.environ.get("SEC_USER_AGENT_EMAIL", "your-email@example.com")
_HEADERS = {"User-Agent": f"Research {_SEC_EMAIL}"}
_CACHE_DIR = Path("data/raw/edgar_form4")
_CIK_CACHE = Path("data/raw/sec_cik_map.json")
_RATE_LIMIT = 0.12   # secondes entre requêtes SEC


def _get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        time.sleep(_RATE_LIMIT)
        try:
            r = requests.get(url, headers=_HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Ticker → CIK
# ---------------------------------------------------------------------------

def get_cik_map(force: bool = False) -> dict[str, str]:
    """Retourne {ticker: '0001045810'} — CIK paddé à 10 chiffres."""
    if _CIK_CACHE.exists() and not force:
        return json.loads(_CIK_CACHE.read_text())

    r = _get("https://www.sec.gov/files/company_tickers.json")
    if r is None:
        raise RuntimeError("Impossible de télécharger le CIK map SEC")

    cik_map = {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in r.json().values()
    }
    _CIK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _CIK_CACHE.write_text(json.dumps(cik_map))
    return cik_map


# ---------------------------------------------------------------------------
# Liste des filings Form 4 pour un ticker
# ---------------------------------------------------------------------------

def _get_form4_filings(cik: str, start_date: str = "2013-01-01") -> list[dict]:
    """
    Retourne la liste des Form 4 d'une entreprise depuis start_date.
    Utilise l'API submissions de EDGAR (couvre ~1000 filings récents).
    """
    cache_file = _CACHE_DIR / f"submissions_{cik}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
    else:
        r = _get(f"https://data.sec.gov/submissions/CIK{cik}.json")
        if r is None:
            return []
        data = r.json()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return []

    filings = []
    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accs    = recent.get("accessionNumber", [])
    docs    = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == "4" and dates[i] >= start_date:
            filings.append({
                "date":      dates[i],
                "accession": accs[i],
                "doc":       docs[i],
            })

    return filings


# ---------------------------------------------------------------------------
# Parsing d'un Form 4 XML
# ---------------------------------------------------------------------------

def _xml_val(element, path: str) -> str | None:
    el = element.find(path)
    return el.text.strip() if el is not None and el.text else None


def _parse_form4(cik: str, accession: str, doc: str) -> list[dict]:
    """
    Télécharge et parse un Form 4.
    Retourne une liste de transactions {code, shares, price, is_officer, is_director}.
    Codes retenus : P (achat marché), S (vente marché).
    """
    acc_clean = accession.replace("-", "")
    # Le primaryDocument peut avoir un préfixe stylesheet : "xslF345X06/filename.xml"
    filename = doc.split("/")[-1]
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{filename}"

    cache_file = _CACHE_DIR / f"{acc_clean}.xml"
    if cache_file.exists():
        content = cache_file.read_bytes()
    else:
        r = _get(url)
        if r is None:
            return []
        content = r.content
        cache_file.write_bytes(content)

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    # Relation de l'insider avec l'entreprise
    rel = root.find(".//reportingOwnerRelationship")
    is_officer  = rel is not None and _xml_val(rel, "isOfficer")  == "1"
    is_director = rel is not None and _xml_val(rel, "isDirector") == "1"

    transactions = []
    for txn in root.findall(".//nonDerivativeTransaction"):
        code = _xml_val(txn, ".//transactionCode")
        if code not in ("P", "S"):
            # Exclure grants (A), exercices (M), taxes (F), dons (G)
            continue

        shares_str = _xml_val(txn, ".//transactionShares/value")
        price_str  = _xml_val(txn, ".//transactionPricePerShare/value")

        try:
            shares = float(shares_str or 0)
            price  = float(price_str  or 0)
        except ValueError:
            continue

        if shares <= 0:
            continue

        transactions.append({
            "code":        code,
            "shares":      shares,
            "price":       price,
            "dollar_value": shares * price,
            "is_officer":  is_officer,
            "is_director": is_director,
        })

    return transactions


# ---------------------------------------------------------------------------
# Score mensuel par ticker
# ---------------------------------------------------------------------------

def compute_insider_signal(
    tickers: list[str],
    start_date: str = "2013-01-01",
    verbose: bool = True,
    max_filings_per_ticker: int | None = None,
) -> pd.DataFrame:
    """
    Pour chaque ticker, calcule un score mensuel d'activité insider.

    Score = net_dollar_purchases normalisé :
      Achats en marché ouvert (P) → positif
      Ventes en marché ouvert (S) → négatif (pondéré 0.5 — ventes souvent forcées)

    Retourne DataFrame de shape (dates, tickers), fréquence mensuelle.
    """
    cik_map = get_cik_map()
    records = []

    for i, ticker in enumerate(tickers):
        cik = cik_map.get(ticker)
        if cik is None:
            continue

        if verbose and i % 50 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}...")

        filings = _get_form4_filings(cik, start_date)
        if max_filings_per_ticker:
            filings = filings[:max_filings_per_ticker]
        for filing in filings:
            txns = _parse_form4(cik, filing["accession"], filing["doc"])
            for txn in txns:
                records.append({
                    "ticker": ticker,
                    "date":   filing["date"],
                    **txn,
                })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Score mensuel par ticker
    # Achats = valeur dollar positive, ventes = négative (pondérée 0.5)
    df["net_dollars"] = df.apply(
        lambda r: r["dollar_value"] if r["code"] == "P" else -0.5 * r["dollar_value"],
        axis=1,
    )

    monthly = (
        df.groupby([pd.Grouper(key="date", freq="MS"), "ticker"])["net_dollars"]
        .sum()
        .unstack(level="ticker")
        .fillna(0)
    )

    # Rolling 3 mois pour lisser (un seul mois d'activité insider = bruit)
    signal = monthly.rolling(3, min_periods=1).sum()

    return signal
