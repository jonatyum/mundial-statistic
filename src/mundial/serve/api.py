"""API REST de solo lectura sobre los pronósticos.

Uso: .venv/bin/uvicorn mundial.serve.api:app --port 8026
Toda respuesta incluye last_updated y model_version (frescura, sección 9 del plan).
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from mundial.config import ROOT, settings

app = FastAPI(title="Mundial 2026 — Predicciones", version="0.2.0")
PROC = ROOT / settings()["paths"]["processed"]


def _load(name: str) -> tuple[pd.DataFrame, str]:
    path = PROC / f"{name}.csv"
    if not path.exists():
        raise HTTPException(404, f"{name} no generado aún: corré scripts/run_pipeline.py")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return df, updated


def _payload(name: str, df: pd.DataFrame, updated: str) -> dict:
    version = (str(df["model_version"].iloc[0])
               if "model_version" in df and len(df) else None)
    # NaN no es JSON válido -> None
    records = (df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
               if len(df) else [])
    return {"source": name, "last_updated": updated, "model_version": version,
            "n": len(df), "data": records}


@app.get("/health")
def health():
    files = {p.stem: datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
             for p in sorted(PROC.glob("*.csv"))}
    return {"status": "ok", "files": files}


@app.get("/matches")
def matches(group: str | None = None, status: str | None = None):
    df, updated = _load("match_forecasts")
    if group:
        df = df[df["group"] == group.upper()]
    if status:
        df = df[df["status"] == status]
    return _payload("match_forecasts", df, updated)


@app.get("/groups")
def groups():
    df, updated = _load("group_tables")
    return _payload("group_tables", df, updated)


@app.get("/knockout")
def knockout(round: str | None = None):
    df, updated = _load("knockout_forecasts")
    if round:
        df = df[df["round"].str.lower() == round.lower()]
    return _payload("knockout_forecasts", df, updated)


@app.get("/champion")
def champion():
    df, updated = _load("tournament_probabilities")
    df = df.rename(columns={df.columns[0]: "team"})
    return _payload("tournament_probabilities", df, updated)


@app.get("/calibration")
def calibration():
    df, updated = _load("calibration")
    return _payload("calibration", df, updated)
