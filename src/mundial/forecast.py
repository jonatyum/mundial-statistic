"""Pronóstico completo de un partido: todas las estadísticas derivables del
ensamble (Elo-Davidson + Dixon-Coles condicionado)."""
import numpy as np

from mundial.models.ensemble import blend_market, condition_matrix, log_pool


def match_forecast(dc, elo_model, w: float, home: str, away: str,
                   neutral: bool = True, market=None, market_weight: float = 0.0) -> dict:
    p = log_pool(elo_model.predict(home, away, neutral),
                 dc.predict(home, away, neutral), w)
    p = blend_market(p, market, market_weight)  # mezcla el mercado si hay cuota
    m = condition_matrix(dc.score_matrix(home, away, neutral), p)
    g = np.arange(m.shape[0])
    xg_h = float((m.sum(axis=1) * g).sum())
    xg_a = float((m.sum(axis=0) * g).sum())

    flat = m.ravel()
    top = np.argsort(-flat)[:3]
    scores = [(int(i // m.shape[0]), int(i % m.shape[0]), float(flat[i])) for i in top]

    total = g[:, None] + g[None, :]
    over25 = float(m[total >= 3].sum())
    btts = float(m[1:, 1:].sum())

    return {
        "p_home": p[0], "p_draw": p[1], "p_away": p[2],
        "xg_home": xg_h, "xg_away": xg_a, "xg_total": xg_h + xg_a,
        "score_1": f"{scores[0][0]}-{scores[0][1]}", "p_score_1": scores[0][2],
        "score_2": f"{scores[1][0]}-{scores[1][1]}", "p_score_2": scores[1][2],
        "score_3": f"{scores[2][0]}-{scores[2][1]}", "p_score_3": scores[2][2],
        "p_over_2.5": over25, "p_under_2.5": 1 - over25, "p_btts": btts,
        "elo_home": elo_model.ratings.get(home, elo_model.default),
        "elo_away": elo_model.ratings.get(away, elo_model.default),
        "attack_home": dc.attack.get(home, 1.0), "defense_home": dc.defense.get(home, 1.0),
        "attack_away": dc.attack.get(away, 1.0), "defense_away": dc.defense.get(away, 1.0),
    }
