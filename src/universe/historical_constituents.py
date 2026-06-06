import io
import requests
import pandas as pd
from pathlib import Path

_GITHUB_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes.csv"
_WIKI_URL   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_RAW_GITHUB = Path("data/raw/sp500_historical_constituents.csv")
_CACHE_PATH = Path("data/processed/sp500_universe_snapshots.pkl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_ticker(raw: str) -> str:
    """
    AAMRQ-201312 → AAMRQ  (suffixe = date de retrait de l'indice, métadonnée)
    BF.B → BF-B            (format yfinance)
    """
    parts = raw.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 6:
        raw = parts[0]
    return raw.replace(".", "-")


def _ticker(val) -> str | None:
    """Retourne un ticker clean ou None si NaN."""
    if pd.isna(val) or str(val).strip() == "":
        return None
    return _clean_ticker(str(val).strip())


# ---------------------------------------------------------------------------
# Source 1 : GitHub (1996 → 2019-01-11)
# ---------------------------------------------------------------------------

def _download_github(force: bool = False) -> None:
    if _RAW_GITHUB.exists() and not force:
        return
    _RAW_GITHUB.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(_GITHUB_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    _RAW_GITHUB.write_text(r.text, encoding="utf-8")


def _load_github_snapshots() -> pd.DataFrame:
    """
    Retourne DataFrame (date, tickers) depuis le fichier GitHub.
    Chaque ligne = liste complète des membres à cette date.
    """
    _download_github()
    df = pd.read_csv(_RAW_GITHUB, parse_dates=["date"])
    df["tickers"] = df["tickers"].apply(
        lambda s: set(_clean_ticker(t) for t in s.split(","))
    )
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 2 : Wikipedia (changements jusqu'à aujourd'hui)
# ---------------------------------------------------------------------------

def _fetch_wikipedia_changes() -> pd.DataFrame:
    """
    Scrape le tableau des changements de composition du S&P 500 sur Wikipedia.
    Retourne DataFrame (date, added, removed) trié chronologiquement.
    """
    r = requests.get(_WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))

    # Deuxième tableau = historique des changements
    changes = tables[1].copy()
    changes.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]

    # Retirer la ligne d'en-tête dupliquée
    changes = changes[changes["date"] != "Effective Date"].reset_index(drop=True)

    changes["date"]    = pd.to_datetime(changes["date"], format="mixed", errors="coerce")
    changes["added"]   = changes["added"].apply(_ticker)
    changes["removed"] = changes["removed"].apply(_ticker)
    changes = changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return changes[["date", "added", "removed"]]


# ---------------------------------------------------------------------------
# Fusion : GitHub snapshots + deltas Wikipedia
# ---------------------------------------------------------------------------

def _build_extended_snapshots(force: bool = False) -> pd.DataFrame:
    """
    Construit une série complète de snapshots (date, tickers) en combinant :
    - snapshots GitHub  (1996 → 2019-01-11)
    - deltas Wikipedia  (2019-01-12 → aujourd'hui)

    Le résultat est mis en cache dans un .pkl.
    """
    if _CACHE_PATH.exists() and not force:
        return pd.read_pickle(_CACHE_PATH)

    print("  Construction des snapshots historiques complets...")

    # --- GitHub ---
    github_snaps = _load_github_snapshots()
    cutoff = github_snaps["date"].max()

    # --- Wikipedia ---
    changes = _fetch_wikipedia_changes()
    new_changes = changes[changes["date"] > cutoff].copy()
    print(f"  GitHub jusqu'au {cutoff.date()} + {len(new_changes)} changements Wikipedia après")

    # Partir du dernier snapshot GitHub
    current_set = set(github_snaps.iloc[-1]["tickers"])

    extra_rows = []
    for _, row in new_changes.iterrows():
        if row["added"] and isinstance(row["added"], str):
            current_set.add(row["added"])
        if row["removed"] and isinstance(row["removed"], str) and row["removed"] in current_set:
            current_set.discard(row["removed"])
        extra_rows.append({"date": row["date"], "tickers": set(current_set)})

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        github_snaps["tickers"] = github_snaps["tickers"].apply(set)
        combined = pd.concat([github_snaps, extra_df], ignore_index=True)
    else:
        combined = github_snaps

    combined = combined.sort_values("date").reset_index(drop=True)

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_pickle(_CACHE_PATH)
    print(f"  Cache sauvegardé : {_CACHE_PATH}")

    return combined


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def load_snapshots(min_date: str = "2013-01-01", force: bool = False) -> pd.DataFrame:
    """
    Retourne DataFrame (date, tickers) avec tickers comme liste Python.
    Filtré à partir de min_date.
    """
    df = _build_extended_snapshots(force=force)
    df = df[df["date"] >= min_date].reset_index(drop=True)
    # Convertir set → list pour la compatibilité avec le reste du code
    df["tickers"] = df["tickers"].apply(list)
    return df


def get_universe_at_date(snapshots: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    """
    Retourne la liste des membres du S&P 500 au snapshot le plus récent ≤ date.
    """
    past = snapshots[snapshots["date"] <= date]
    if past.empty:
        return []
    tickers = past.iloc[-1]["tickers"]
    return list(tickers) if not isinstance(tickers, list) else tickers


def all_unique_tickers(snapshots: pd.DataFrame) -> list[str]:
    """Tous les tickers distincts sur la période couverte."""
    seen: set[str] = set()
    for tickers in snapshots["tickers"]:
        seen.update(t for t in tickers if isinstance(t, str))
    return sorted(seen)
