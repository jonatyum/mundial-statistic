"""Backtest temporal sobre mundiales pasados.

Entrena SOLO con partidos anteriores al inicio del torneo (anti-fuga) y evalúa
las predicciones 1X2 sobre la fase de grupos (resultados limpios a 90').
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mundial.evaluate import metrics
from mundial.features import elo
from mundial.ingest import results
from mundial.models.dixon_coles import DixonColes
from mundial.models.elo_model import EloDavidson
from mundial.models.ensemble import log_pool

# (inicio del torneo, fin de fase de grupos)
WORLD_CUPS = {
    2014: ("2014-06-12", "2014-06-26"),
    2018: ("2018-06-14", "2018-06-28"),
    2022: ("2022-11-20", "2022-12-02"),
}


@dataclass
class BacktestResult:
    year: int
    n_matches: int
    table: pd.DataFrame          # métricas por modelo
    probs: dict[str, np.ndarray]  # por modelo: (n, 3)
    outcomes: np.ndarray


def run(year: int, df: pd.DataFrame | None = None, weights=(0.0, 0.25, 0.5, 0.75, 1.0)) -> BacktestResult:
    start, group_end = WORLD_CUPS[year]
    if df is None:
        df = results.load()
    train = results.before(df, start)

    test = results.played(df)
    test = test[(test["tournament"] == "FIFA World Cup")
                & (test["date"] >= pd.Timestamp(start, tz="UTC"))
                & (test["date"] <= pd.Timestamp(group_end, tz="UTC"))]

    ratings = elo.compute(train)
    elo_model = EloDavidson(ratings).fit_nu(train)
    dc = DixonColes().fit(train, ref_date=start)

    rows = []
    p_elo, p_dc = [], []
    for _, m in test.iterrows():
        p_elo.append(elo_model.predict(m.home_team, m.away_team, m.neutral))
        p_dc.append(dc.predict(m.home_team, m.away_team, m.neutral))
        rows.append(metrics.outcome_index(m.home_score, m.away_score))
    p_elo = np.array(p_elo)
    p_dc = np.array(p_dc)
    outcomes = np.array(rows)

    probs = {"uniform": np.full((len(outcomes), 3), 1 / 3), "elo": p_elo, "dixon_coles": p_dc}
    for w in weights:
        if 0 < w < 1:
            probs[f"ensemble_w{w}"] = log_pool(p_elo, p_dc, w)

    table = pd.DataFrame([
        {"model": name,
         "log_loss": metrics.log_loss(p, outcomes),
         "brier": metrics.brier(p, outcomes),
         "rps": metrics.rps(p, outcomes)}
        for name, p in probs.items()
    ]).set_index("model").sort_values("log_loss")

    return BacktestResult(year, len(outcomes), table, probs, outcomes)


def live_model_comparison(df: pd.DataFrame | None = None, w: float = 0.5,
                          start: str = "2026-06-11",
                          season_start: str = "2026-06-01") -> pd.DataFrame:
    """Compara cada modelo sobre los partidos YA JUGADOS del Mundial 2026.

    Entrena SOLO con datos anteriores al inicio del torneo (anti-fuga) y evalúa
    Elo-Davidson, Dixon-Coles, el ensamble de producción (peso `w`) y el baseline
    uniforme. Una fila por modelo, ordenadas por log-loss (mejor primero).
    Vacío mientras no haya partidos terminados. `df` inyectable para tests."""
    if df is None:
        df = results.load()
    test = results.played(df)
    test = test[(test["tournament"] == "FIFA World Cup")
                & (test["date"] >= pd.Timestamp(season_start, tz="UTC"))]
    if test.empty:
        return pd.DataFrame()

    train = results.before(df, start)
    ratings = elo.compute(train)
    elo_model = EloDavidson(ratings).fit_nu(train)
    dc = DixonColes().fit(train, ref_date=start)

    p_elo, p_dc, outcomes = [], [], []
    for _, m in test.iterrows():
        p_elo.append(elo_model.predict(m.home_team, m.away_team, m.neutral))
        p_dc.append(dc.predict(m.home_team, m.away_team, m.neutral))
        outcomes.append(metrics.outcome_index(m.home_score, m.away_score))
    p_elo, p_dc = np.array(p_elo), np.array(p_dc)
    outcomes = np.array(outcomes)

    models = {
        "dixon_coles": p_dc,
        "elo": p_elo,
        "ensemble": log_pool(p_elo, p_dc, w),
        "uniform": np.full((len(outcomes), 3), 1 / 3),
    }
    rows = [{
        "model": name,
        "w": round(w, 2) if name == "ensemble" else None,
        "n": int(len(outcomes)),
        "log_loss": metrics.log_loss(p, outcomes),
        "brier": metrics.brier(p, outcomes),
        "rps": metrics.rps(p, outcomes),
        "accuracy": float((p.argmax(axis=1) == outcomes).mean()),
    } for name, p in models.items()]
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def best_ensemble_weight(years=(2014, 2018, 2022), df: pd.DataFrame | None = None) -> float:
    """Peso w del log-pool que minimiza el log-loss promedio entre mundiales."""
    if df is None:
        df = results.load()
    results_by_year = [run(y, df) for y in years]
    grid = np.linspace(0, 1, 21)
    losses = []
    for w in grid:
        per_year = [
            metrics.log_loss(log_pool(r.probs["elo"], r.probs["dixon_coles"], w), r.outcomes)
            for r in results_by_year
        ]
        losses.append(np.mean(per_year))
    return float(grid[int(np.argmin(losses))])
