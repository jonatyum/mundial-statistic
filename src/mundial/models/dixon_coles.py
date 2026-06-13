"""Poisson Dixon-Coles con ponderación temporal exponencial.

Ajuste por punto fijo (Maher) de ataque/defensa + ventaja de localía, con
shrinkage hacia la media para selecciones con pocos partidos; luego rho
(corrección de marcadores bajos) por MLE unidimensional.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from mundial.config import settings

PRIOR_WEIGHT = 5.0  # pseudo-partidos hacia ataque/defensa = media (shrinkage)


class DixonColes:
    def __init__(self):
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.gamma = 1.25   # ventaja de localía (multiplicador de goles)
        self.mu = 1.3       # goles base por equipo
        self.rho = 0.0
        cfg = settings()["dixon_coles"]
        self.xi = cfg["xi"]
        self.max_goals = cfg["max_goals"]

    # ------------------------------------------------------------------ fit
    def fit(self, df_played: pd.DataFrame, ref_date: str, n_iter: int = 60) -> "DixonColes":
        ref = pd.Timestamp(ref_date, tz="UTC")
        years = settings()["dixon_coles"]["train_years"]
        df = df_played[(df_played["date"] < ref)
                       & (df_played["date"] >= ref - pd.Timedelta(days=365 * years))]
        if len(df) < 100:
            raise ValueError(f"datos insuficientes para entrenar DC: {len(df)} partidos")

        w = np.exp(-self.xi * (ref - df["date"]).dt.days.to_numpy(dtype=float))
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = df["home_team"].map(idx).to_numpy()
        ai = df["away_team"].map(idx).to_numpy()
        hg = df["home_score"].to_numpy(dtype=float)
        ag = df["away_score"].to_numpy(dtype=float)
        home_field = (~df["neutral"]).to_numpy()  # localía solo si no es neutral

        att = np.ones(n)
        dfn = np.ones(n)
        self.mu = float(np.average((hg + ag) / 2.0, weights=w))

        for _ in range(n_iter):
            gam = np.where(home_field, self.gamma, 1.0)
            # ataque: goles anotados / exposición (mu * defensa rival * localía)
            num = np.bincount(hi, w * hg, n) + np.bincount(ai, w * ag, n)
            den = (np.bincount(hi, w * self.mu * dfn[ai] * gam, n)
                   + np.bincount(ai, w * self.mu * dfn[hi], n))
            att = (num + PRIOR_WEIGHT * self.mu) / (den + PRIOR_WEIGHT * self.mu)
            # defensa: goles concedidos / exposición
            num = np.bincount(hi, w * ag, n) + np.bincount(ai, w * hg, n)
            den = (np.bincount(hi, w * self.mu * att[ai], n)
                   + np.bincount(ai, w * self.mu * att[hi] * gam, n))
            dfn = (num + PRIOR_WEIGHT * self.mu) / (den + PRIOR_WEIGHT * self.mu)
            # identifiabilidad: media de ataque = media de defensa = 1;
            # mu (nivel de goles) se re-estima para conservar la escala
            att /= att.mean()
            dfn /= dfn.mean()
            gam = np.where(home_field, self.gamma, 1.0)
            unit_h = att[hi] * dfn[ai] * gam
            unit_a = att[ai] * dfn[hi]
            self.mu = float(np.sum(w * (hg + ag)) / np.sum(w * (unit_h + unit_a)))
            # localía: goles de local observados / esperados sin gamma
            mask = home_field
            exp_h = self.mu * att[hi[mask]] * dfn[ai[mask]]
            self.gamma = float(np.sum(w[mask] * hg[mask]) / np.sum(w[mask] * exp_h))

        self.attack = {t: float(att[i]) for t, i in idx.items()}
        self.defense = {t: float(dfn[i]) for t, i in idx.items()}
        self._fit_rho(hi, ai, hg, ag, w, att, dfn, home_field)
        return self

    def _fit_rho(self, hi, ai, hg, ag, w, att, dfn, home_field):
        gam = np.where(home_field, self.gamma, 1.0)
        lh = self.mu * att[hi] * dfn[ai] * gam
        la = self.mu * att[ai] * dfn[hi]
        low = (hg <= 1) & (ag <= 1)
        lh_l, la_l, hg_l, ag_l, w_l = lh[low], la[low], hg[low], ag[low], w[low]

        def tau(rho: float) -> np.ndarray:
            t = np.ones(len(hg_l))
            t = np.where((hg_l == 0) & (ag_l == 0), 1 - lh_l * la_l * rho, t)
            t = np.where((hg_l == 0) & (ag_l == 1), 1 + lh_l * rho, t)
            t = np.where((hg_l == 1) & (ag_l == 0), 1 + la_l * rho, t)
            t = np.where((hg_l == 1) & (ag_l == 1), 1 - rho, t)
            return t

        def nll(rho: float) -> float:
            return -np.sum(w_l * np.log(np.clip(tau(rho), 1e-10, None)))

        res = minimize_scalar(nll, bounds=(-0.15, 0.15), method="bounded")
        self.rho = float(res.x)

    # ------------------------------------------------------------- predict
    def lambdas(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        ah = self.attack.get(home, 1.0)
        dh = self.defense.get(home, 1.0)
        aa = self.attack.get(away, 1.0)
        da = self.defense.get(away, 1.0)
        gam = 1.0 if neutral else self.gamma
        return self.mu * ah * da * gam, self.mu * aa * dh

    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        lh, la = self.lambdas(home, away, neutral)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        # corrección DC en marcadores bajos
        m[0, 0] *= 1 - lh * la * self.rho
        m[0, 1] *= 1 + lh * self.rho
        m[1, 0] *= 1 + la * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum()

    @staticmethod
    def outcome_probs(matrix: np.ndarray) -> np.ndarray:
        home = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away = np.triu(matrix, 1).sum()
        return np.array([home, draw, away])

    def predict(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        return self.outcome_probs(self.score_matrix(home, away, neutral))
