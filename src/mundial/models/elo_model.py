"""Modelo Elo probabilístico con extensión Davidson para el empate.

P(local)  = pi_h / Z;  P(empate) = nu*sqrt(pi_h*pi_a) / Z;  P(visita) = pi_a / Z
con pi = 10^(rating/400) (al local se le suma la ventaja de localía si no es neutral)
y Z el normalizador. nu se ajusta por máxima verosimilitud en datos recientes.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from mundial.config import settings


class EloDavidson:
    def __init__(self, ratings: dict[str, float], nu: float = 0.6):
        self.ratings = ratings
        self.nu = nu
        self.default = settings()["elo"]["initial"]

    def _pis(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        adv = 0.0 if neutral else settings()["elo"]["home_advantage"]
        rh = self.ratings.get(home, self.default) + adv
        ra = self.ratings.get(away, self.default)
        return 10 ** (rh / 400.0), 10 ** (ra / 400.0)

    def predict(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        ph, pa = self._pis(home, away, neutral)
        d = self.nu * np.sqrt(ph * pa)
        z = ph + pa + d
        return np.array([ph / z, d / z, pa / z])

    def fit_nu(self, df_played: pd.DataFrame, years: int = 8) -> "EloDavidson":
        cutoff = df_played["date"].max() - pd.Timedelta(days=365 * years)
        recent = df_played[df_played["date"] >= cutoff]
        homes = recent["home_team"].to_numpy()
        aways = recent["away_team"].to_numpy()
        neutrals = recent["neutral"].to_numpy()
        diffs = (recent["home_score"] - recent["away_score"]).to_numpy(dtype=float)
        outcomes = np.where(diffs > 0, 0, np.where(diffs == 0, 1, 2))

        adv = settings()["elo"]["home_advantage"]
        rh = np.array([self.ratings.get(h, self.default) for h in homes])
        ra = np.array([self.ratings.get(a, self.default) for a in aways])
        rh = rh + np.where(neutrals, 0.0, adv)
        pi_h, pi_a = 10 ** (rh / 400.0), 10 ** (ra / 400.0)
        sq = np.sqrt(pi_h * pi_a)

        def nll(nu: float) -> float:
            z = pi_h + pi_a + nu * sq
            probs = np.stack([pi_h / z, nu * sq / z, pi_a / z], axis=1)
            p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-12, 1)
            return -np.sum(np.log(p))

        res = minimize_scalar(nll, bounds=(0.01, 3.0), method="bounded")
        self.nu = float(res.x)
        return self
