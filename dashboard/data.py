import pandas as pd
import streamlit as st
from pathlib import Path


def list_backtest_configs() -> list[str]:
    base = Path("data/processed/backtest")
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "metrics.csv").exists()
    )


@st.cache_data
def load_backtest(config_name: str = "sn1_bl0") -> dict | None:
    base = Path(f"data/processed/backtest/{config_name}")
    if not (base / "portfolio_returns.csv").exists():
        return None
    return {
        "returns":        pd.read_csv(base / "portfolio_returns.csv", index_col=0, parse_dates=True).squeeze(),
        "benchmark":      pd.read_csv(base / "benchmark_returns.csv", index_col=0, parse_dates=True).squeeze(),
        "weights":        pd.read_csv(base / "weights.csv",           index_col=0, parse_dates=True),
        "factor_weights": pd.read_csv(base / "factor_weights.csv",    index_col=0, parse_dates=True),
        "turnover":       pd.read_csv(base / "turnover.csv",          index_col=0, parse_dates=True).squeeze(),
        "costs":          pd.read_csv(base / "costs.csv",             index_col=0, parse_dates=True).squeeze(),
        "metrics":        pd.read_csv(base / "metrics.csv",           index_col=0, header=None).squeeze(),
    }


@st.cache_data
def load_ic() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    s = Path("data/processed/analysis/ic_series.csv")
    m = Path("data/processed/analysis/ic_summary.csv")
    if not s.exists():
        return None, None
    return (pd.read_csv(s, index_col=0, parse_dates=True),
            pd.read_csv(m, index_col=0))


@st.cache_data
def load_sectors() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    e = Path("data/processed/analysis/sector_exposures.csv")
    s = Path("data/processed/analysis/sector_summary.csv")
    if not e.exists():
        return None, None
    return (pd.read_csv(e, index_col=0, parse_dates=True),
            pd.read_csv(s, index_col=0))


@st.cache_data
def load_signals() -> dict[str, pd.DataFrame]:
    sig = {}
    for name in ["momentum", "low_volatility", "reversal", "illiquidity",
                 "high_52w", "gross_profitability", "accruals", "asset_growth", "insider", "sue"]:
        p = Path(f"data/processed/signals/{name}.csv")
        if p.exists():
            sig[name] = pd.read_csv(p, index_col=0, parse_dates=True)
    return sig


@st.cache_data
def load_picks() -> pd.DataFrame:
    picks_dir = Path("data/processed/live_picks")
    if not picks_dir.exists():
        return pd.DataFrame()
    files = sorted(picks_dir.glob("picks_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


@st.cache_data
def load_returns() -> pd.DataFrame | None:
    p = Path("data/processed/sp500_returns.csv")
    if not p.exists():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True)


@st.cache_data
def load_monthly_returns() -> pd.DataFrame | None:
    p = Path("data/processed/sp500_returns.csv")
    if not p.exists():
        return None
    daily = pd.read_csv(p, index_col=0, parse_dates=True)
    return (1 + daily).resample("MS").prod() - 1
