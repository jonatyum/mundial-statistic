import numpy as np
import pandas as pd

from mundial.models.dixon_coles import DixonColes
from mundial.models.elo_model import EloDavidson
from mundial.models.ensemble import condition_matrix, log_pool


def _synthetic_matches(n=600, seed=7):
    """Liga sintética: Fuerte > Medio > Débil, con localía."""
    rng = np.random.default_rng(seed)
    strength = {"Fuerte": 2.0, "Medio": 1.3, "Débil": 0.7}
    teams = list(strength)
    rows = []
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    for k in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        lh = strength[h] / strength[a] * 1.4 * 1.2  # localía
        la = strength[a] / strength[h] * 1.4
        rows.append({"date": dates[k], "home_team": h, "away_team": a,
                     "home_score": float(rng.poisson(lh)),
                     "away_score": float(rng.poisson(la)),
                     "tournament": "Friendly", "neutral": False})
    return pd.DataFrame(rows)


def test_dixon_coles_recovers_ordering():
    df = _synthetic_matches()
    dc = DixonColes().fit(df, ref_date="2024-09-01")
    assert dc.attack["Fuerte"] > dc.attack["Medio"] > dc.attack["Débil"]
    assert dc.gamma > 1.0  # detecta la localía
    p = dc.predict("Fuerte", "Débil", neutral=True)
    assert p[0] > 0.5 and p[0] > p[2]
    m = dc.score_matrix("Fuerte", "Débil")
    assert abs(m.sum() - 1.0) < 1e-9


def test_elo_davidson_probs_sum_and_order():
    model = EloDavidson({"A": 2000.0, "B": 1600.0}, nu=0.7)
    p = model.predict("A", "B", neutral=True)
    assert abs(p.sum() - 1.0) < 1e-9
    assert p[0] > p[2]
    # localía mejora al local
    p_home = model.predict("B", "A", neutral=False)
    p_neutral = model.predict("B", "A", neutral=True)
    assert p_home[0] > p_neutral[0]


def test_log_pool_and_condition_matrix():
    p1 = np.array([0.5, 0.3, 0.2])
    p2 = np.array([0.2, 0.3, 0.5])
    pooled = log_pool(p1, p2, 0.5)
    assert abs(pooled.sum() - 1.0) < 1e-9
    m = np.full((3, 3), 1 / 9)
    target = np.array([0.6, 0.3, 0.1])
    out = condition_matrix(m, target)
    assert abs(np.tril(out, -1).sum() - 0.6) < 1e-9
    assert abs(np.trace(out) - 0.3) < 1e-9
