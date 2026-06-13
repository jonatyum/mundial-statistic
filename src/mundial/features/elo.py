"""Cálculo de ratings Elo propios desde el histórico (convención eloratings.net)."""
import numpy as np
import pandas as pd

from mundial.config import settings

_CONTINENTAL = (
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "Gold Cup", "CONCACAF Championship", "Oceania Nations Cup",
    "Confederations Cup", "African Nations Championship",
)


def k_factor(tournament: str) -> float:
    cfg = settings()["elo"]
    t = tournament or ""
    if t == "FIFA World Cup":
        return cfg["k_world_cup"]
    if any(t.startswith(c) for c in _CONTINENTAL) and "qualification" not in t:
        return cfg["k_continental"]
    if t == "Friendly":
        return cfg["k_friendly"]
    return cfg["k_qualifier"]


def _margin_mult(goal_diff: int) -> float:
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return 1.75 + (d - 3) / 8.0 if d > 3 else 1.75


def compute(df_played: pd.DataFrame) -> dict[str, float]:
    """Replay cronológico de partidos jugados -> rating final por selección."""
    cfg = settings()["elo"]
    init, home_adv = cfg["initial"], cfg["home_advantage"]
    ratings: dict[str, float] = {}
    cols = df_played[["home_team", "away_team", "home_score", "away_score",
                      "tournament", "neutral"]].to_numpy()
    for home, away, hs, as_, tourn, neutral in cols:
        rh = ratings.get(home, init)
        ra = ratings.get(away, init)
        adv = 0.0 if neutral else home_adv
        expected = 1.0 / (1.0 + 10 ** ((ra - rh - adv) / 400.0))
        diff = int(hs) - int(as_)
        actual = 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5)
        delta = k_factor(tourn) * _margin_mult(diff) * (actual - expected)
        ratings[home] = rh + delta
        ratings[away] = ra - delta
    return ratings


def expected_home(rh: float, ra: float, neutral: bool) -> float:
    adv = 0.0 if neutral else settings()["elo"]["home_advantage"]
    return 1.0 / (1.0 + 10 ** ((ra - rh - adv) / 400.0))
