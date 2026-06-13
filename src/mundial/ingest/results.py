"""Ingesta del histórico de partidos internacionales (martj42)."""
from pathlib import Path

import pandas as pd
import requests

from mundial.config import ROOT, canonical, settings


def download(force: bool = False) -> Path:
    dest = ROOT / settings()["paths"]["raw"] / "results.csv"
    if force or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(settings()["sources"]["results_csv"], timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return dest


def load() -> pd.DataFrame:
    df = pd.read_csv(download(), na_values=["NA"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    df["neutral"] = df["neutral"].astype(bool)
    df["home_score"] = df["home_score"].astype("Float64")
    df["away_score"] = df["away_score"].astype("Float64")
    return df.sort_values("date").reset_index(drop=True)


def played(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["home_score"].notna() & df["away_score"].notna()]


def before(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Partidos jugados estrictamente antes de `date` (anti-fuga temporal)."""
    cutoff = pd.Timestamp(date, tz="UTC")
    p = played(df)
    return p[p["date"] < cutoff]
