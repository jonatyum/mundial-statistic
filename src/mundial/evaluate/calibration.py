"""Monitoreo de calibración en vivo: predicciones congeladas vs. resultados reales."""
import numpy as np
import pandas as pd

from mundial.config import ROOT, settings
from mundial.evaluate import metrics
from mundial.ingest import results


def calibration_table(df: pd.DataFrame | None = None,
                      log: pd.DataFrame | None = None) -> pd.DataFrame:
    """Une el prediction log con los resultados reales del Mundial 2026.

    Devuelve un partido por fila con log-loss/Brier individual y acumulado.
    Vacío mientras no haya partidos terminados. `df`/`log` inyectables para tests."""
    if log is None:
        log_path = ROOT / settings()["paths"]["predictions"] / "prediction_log.csv"
        if not log_path.exists():
            return pd.DataFrame()
        log = pd.read_csv(log_path)

    if df is None:
        df = results.load()
    real = results.played(df)
    real = real[(real["tournament"] == "FIFA World Cup")
                & (real["date"] >= pd.Timestamp("2026-06-01", tz="UTC"))]
    if real.empty:
        return pd.DataFrame()

    merged = log.merge(
        real[["home_team", "away_team", "home_score", "away_score"]],
        left_on=["home", "away"], right_on=["home_team", "away_team"], how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    probs = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    outcomes = np.array([metrics.outcome_index(h, a)
                         for h, a in zip(merged["home_score"], merged["away_score"])])
    picked = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, 1)

    out = merged[["date", "home", "away", "p_home", "p_draw", "p_away", "model_version"]].copy()
    out["result"] = [f"{int(h)}-{int(a)}" for h, a in
                     zip(merged["home_score"], merged["away_score"])]
    out["outcome"] = np.array(["local", "empate", "visita"])[outcomes]
    out["log_loss"] = -np.log(picked)
    out["brier"] = np.sum((probs - np.eye(3)[outcomes]) ** 2, axis=1)
    out["hit"] = probs.argmax(axis=1) == outcomes
    out = out.sort_values("date").reset_index(drop=True)
    out["log_loss_acum"] = out["log_loss"].expanding().mean()
    out["brier_acum"] = out["brier"].expanding().mean()
    out["log_loss_uniforme"] = np.log(3)
    return out
