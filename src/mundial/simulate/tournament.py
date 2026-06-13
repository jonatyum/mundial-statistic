"""Simulación Monte Carlo vectorizada del Mundial 2026 (104 partidos).

- Fase de grupos: 72 partidos; los ya jugados usan el marcador real.
- Desempates: puntos > dif. de gol > goles a favor > aleatorio
  (aprox. del reglamento FIFA: omite head-to-head y fair play, que casi
  siempre coinciden tras GD/GF; sin datos de tarjetas aún).
- Terceros: ranking de los 12, clasifican 8, asignación a llaves del R32 por
  matching bipartito que respeta los grupos candidatos de cada llave
  (aprox. del Anexo C del reglamento).
- Eliminatorias: 90' con matriz Dixon-Coles condicionada al ensamble; prórroga
  con lambda * extra_time_factor; penales ~50/50 con leve ajuste por Elo.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from mundial.config import settings, tournament
from mundial.models.ensemble import condition_matrix, log_pool

# país anfitrión de la sede por partido de R32 (resto en EE. UU.)
# Aproximación: R16 en adelante se tratan como sede EE. UU.
R32_VENUE_HOST = {75: "Mexico", 79: "Mexico", 83: "Canada", 85: "Canada"}
THIRD_SLOT_ORDER = [74, 77, 79, 80, 81, 82, 85, 87]


class TournamentSimulator:
    def __init__(self, dc, elo_model, ensemble_w: float, fixtures: pd.DataFrame,
                 n_sims: int | None = None, seed: int | None = None):
        cfg = settings()
        self.dc = dc
        self.elo = elo_model
        self.w = ensemble_w
        self.n = n_sims or cfg["simulation"]["n_sims"]
        self.et_factor = cfg["simulation"]["extra_time_factor"]
        self.rng = np.random.default_rng(cfg["seed"] if seed is None else seed)

        t = tournament()
        self.groups: dict[str, list[str]] = t["groups"]
        self.group_letters = list(self.groups)
        self.teams = [team for g in self.group_letters for team in self.groups[g]]
        self.tidx = {team: i for i, team in enumerate(self.teams)}
        self.t26 = t
        self.fixtures = fixtures  # 72 partidos de grupos con marcador real si ya se jugó
        self._precompute_pairs()
        self._precompute_third_assignments()

    # ---------------------------------------------------------- precomputos
    def _match_probs(self, home: str, away: str, neutral: bool):
        """Matriz de marcadores del ensamble (DC condicionada) y lambdas."""
        p = log_pool(self.elo.predict(home, away, neutral),
                     self.dc.predict(home, away, neutral), self.w)
        m = condition_matrix(self.dc.score_matrix(home, away, neutral), p)
        return m, self.dc.lambdas(home, away, neutral)

    def _precompute_pairs(self):
        """Matrices acumuladas para todo par ordenado en sede neutral (para KO)."""
        n_t = len(self.teams)
        g = self.dc.max_goals + 1
        self.pair_cum = np.empty((n_t * n_t, g * g), dtype=np.float64)
        self.pair_lh = np.empty(n_t * n_t)
        self.pair_la = np.empty(n_t * n_t)
        for i, home in enumerate(self.teams):
            for j, away in enumerate(self.teams):
                if i == j:
                    self.pair_cum[i * n_t + j] = 1.0
                    self.pair_lh[i * n_t + j] = self.pair_la[i * n_t + j] = 1.0
                    continue
                m, (lh, la) = self._match_probs(home, away, neutral=True)
                self.pair_cum[i * n_t + j] = np.cumsum(m.ravel())
                self.pair_lh[i * n_t + j] = lh
                self.pair_la[i * n_t + j] = la
        ratings = self.elo.ratings
        default = self.elo.default
        self.team_elo = np.array([ratings.get(t, default) for t in self.teams])

    def _precompute_third_assignments(self):
        """Matching llave->grupo para cada combinación de 8 terceros (<=495)."""
        slots = {m: set(self.t26["round_of_32"][m]["away"].split(":")[1].split(","))
                 for m in THIRD_SLOT_ORDER}
        self.third_assign: dict[int, np.ndarray] = {}
        letters = self.group_letters
        for combo in combinations(range(12), 8):
            qualified = [letters[i] for i in combo]
            assignment = self._match_slots(slots, qualified)
            mask = sum(1 << i for i in combo)
            self.third_assign[mask] = np.array(
                [letters.index(assignment[m]) for m in THIRD_SLOT_ORDER])

    @staticmethod
    def _match_slots(slots: dict[int, set], qualified: list[str]) -> dict[int, str]:
        """Backtracking: asigna cada llave a un grupo clasificado permitido."""
        order = sorted(slots, key=lambda m: len(slots[m] & set(qualified)))
        used: set[str] = set()
        result: dict[int, str] = {}

        def bt(k: int) -> bool:
            if k == len(order):
                return True
            m = order[k]
            for g in sorted(slots[m] & set(qualified)):
                if g not in used:
                    used.add(g)
                    result[m] = g
                    if bt(k + 1):
                        return True
                    used.remove(g)
                    del result[m]
            return False

        if not bt(0):  # no debería ocurrir (Anexo C garantiza matching)
            raise RuntimeError(f"sin matching para terceros {qualified}")
        return result

    # ---------------------------------------------------------- fase grupos
    def _simulate_groups(self):
        """Devuelve winners, runners, thirds: arrays (n_sims, 12) con idx global,
        y standings dict para reportes."""
        n, n_teams = self.n, len(self.teams)
        pts = np.zeros((n, n_teams))
        gf = np.zeros((n, n_teams))
        ga = np.zeros((n, n_teams))
        self.match_samples = {}  # (home, away) -> (hg, ag) por sim

        for _, m in self.fixtures.iterrows():
            hi, ai = self.tidx[m.home_team], self.tidx[m.away_team]
            if pd.notna(m.home_score):  # partido ya jugado: marcador real
                hg = np.full(n, int(m.home_score))
                ag = np.full(n, int(m.away_score))
            else:
                mat, _ = self._match_probs(m.home_team, m.away_team, bool(m.neutral))
                cum = np.cumsum(mat.ravel())
                idx = np.searchsorted(cum, self.rng.random(n))
                g = self.dc.max_goals + 1
                hg, ag = idx // g, idx % g
            self.match_samples[(m.home_team, m.away_team)] = (hg, ag)
            pts[:, hi] += np.where(hg > ag, 3, np.where(hg == ag, 1, 0))
            pts[:, ai] += np.where(ag > hg, 3, np.where(hg == ag, 1, 0))
            gf[:, hi] += hg
            gf[:, ai] += ag
            ga[:, hi] += ag
            ga[:, ai] += hg

        gd = gf - ga
        self._pts, self._gd, self._gf = pts, gd, gf
        # composite: puntos > GD > GF > ruido aleatorio (desempate por sorteo)
        comp = pts * 1e8 + (gd + 60) * 1e5 + gf * 1e2 + self.rng.random((n, n_teams)) * 99

        winners = np.empty((n, 12), dtype=np.int32)
        runners = np.empty((n, 12), dtype=np.int32)
        thirds = np.empty((n, 12), dtype=np.int32)
        for gi in range(12):
            cols = np.arange(gi * 4, gi * 4 + 4)
            order = np.argsort(-comp[:, cols], axis=1)  # (n, 4) posiciones
            winners[:, gi] = cols[order[:, 0]]
            runners[:, gi] = cols[order[:, 1]]
            thirds[:, gi] = cols[order[:, 2]]
        self._group_comp = comp
        return winners, runners, thirds

    def _rank_thirds(self, thirds: np.ndarray) -> np.ndarray:
        """(n,12) idx de terceros -> (n,8) índice de grupo de los clasificados
        + asignación por llave. Devuelve (n, 8) idx global de equipo por llave."""
        n = self.n
        comp_thirds = np.take_along_axis(self._group_comp, thirds, axis=1)
        order = np.argsort(-comp_thirds, axis=1)
        qual_groups = np.sort(order[:, :8], axis=1)            # (n, 8) idx de grupo
        masks = (1 << qual_groups).sum(axis=1)                  # (n,)
        self._third_qual_masks = masks

        slot_group = np.empty((n, 8), dtype=np.int32)
        for mask in np.unique(masks):
            sel = masks == mask
            slot_group[sel] = self.third_assign[int(mask)]
        # equipo tercero del grupo asignado a cada llave
        return np.take_along_axis(thirds, slot_group, axis=1)

    # ---------------------------------------------------------- eliminatorias
    def _ko_match(self, home: np.ndarray, away: np.ndarray, venue_host: str | None):
        """Simula un cruce KO por sim. Devuelve idx del ganador."""
        n_t = len(self.teams)
        # ventaja de localía si uno de los dos es el anfitrión de la sede
        if venue_host is not None and venue_host in self.tidx:
            host = self.tidx[venue_host]
            swap = away == host
            home, away = np.where(swap, away, home), np.where(swap, home, away)
            home_field = (home == host)
        else:
            home_field = np.zeros(len(home), dtype=bool)

        pair = home * n_t + away
        g = self.dc.max_goals + 1
        u = self.rng.random(self.n)
        cum = self.pair_cum[pair]
        idx = (cum < u[:, None]).sum(axis=1)
        hg, ag = idx // g, idx % g

        # localía del anfitrión en KO: sesgo vía Elo en vez de re-simular matriz
        # (aprox: el efecto entra en prórroga/penales y al desempatar 90')
        lh = self.pair_lh[pair] * (settings()["elo"]["home_advantage"] / 400 * 0.5 + 1.0) ** home_field
        la = self.pair_la[pair]

        draw = hg == ag
        # prórroga con lambdas reducidas
        et_h = self.rng.poisson(lh * self.et_factor)
        et_a = self.rng.poisson(la * self.et_factor)
        hg2 = hg + np.where(draw, et_h, 0)
        ag2 = ag + np.where(draw, et_a, 0)
        still = hg2 == ag2
        # penales: base 50/50 con leve ajuste Elo (+ localía ya incluida en lh)
        d_elo = self.team_elo[home] - self.team_elo[away] + np.where(home_field, 40, 0)
        p_home_pens = np.clip(0.5 + d_elo * 0.0003, 0.35, 0.65)
        pens_home = self.rng.random(self.n) < p_home_pens
        home_wins = np.where(still, pens_home, hg2 > ag2)
        return np.where(home_wins, home, away)

    # ---------------------------------------------------------------- run
    def run(self) -> dict:
        winners, runners, thirds = self._simulate_groups()
        third_by_slot = self._rank_thirds(thirds)   # (n, 8) por THIRD_SLOT_ORDER
        slot_of = {m: k for k, m in enumerate(THIRD_SLOT_ORDER)}
        gl = {letter: i for i, letter in enumerate(self.group_letters)}

        def resolve(token: str, match_no: int) -> np.ndarray:
            kind, arg = token.split("_", 1) if "_" in token else (token, "")
            if token.startswith("W_"):
                return winners[:, gl[token[2:]]]
            if token.startswith("R_"):
                return runners[:, gl[token[2:]]]
            if token.startswith("T:"):
                return third_by_slot[:, slot_of[match_no]]
            raise ValueError(token)

        self.ko_record: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        ko_winner: dict[int, np.ndarray] = {}
        for mno, spec in self.t26["round_of_32"].items():
            home = resolve(spec["home"], mno)
            away = resolve(spec["away"], mno)
            ko_winner[mno] = self._ko_match(home, away, R32_VENUE_HOST.get(mno))
            self.ko_record[mno] = (home * len(self.teams) + away, ko_winner[mno])

        rounds = [("round_of_16", "United States"), ("quarter_finals", "United States"),
                  ("semi_finals", "United States"), ("final", "United States")]
        reach = {"r32": np.zeros(len(self.teams)), "r16": np.zeros(len(self.teams)),
                 "qf": np.zeros(len(self.teams)), "sf": np.zeros(len(self.teams)),
                 "final": np.zeros(len(self.teams)), "champion": np.zeros(len(self.teams))}

        # llegada a R32
        for arr in (winners, runners):
            for gi in range(12):
                reach["r32"] += np.bincount(arr[:, gi], minlength=len(self.teams))
        for k in range(8):
            reach["r32"] += np.bincount(third_by_slot[:, k], minlength=len(self.teams))

        stage_keys = ["r16", "qf", "sf", "final"]
        current = ko_winner
        for (stage, venue), key in zip(rounds, stage_keys + ["champion"]):
            nxt: dict[int, np.ndarray] = {}
            for mno, (m1, m2) in self.t26[stage].items():
                home, away = current[m1], current[m2]
                for arr in (home, away):
                    reach[key] += np.bincount(arr, minlength=len(self.teams))
                nxt[mno] = self._ko_match(home, away, venue)
                self.ko_record[mno] = (home * len(self.teams) + away, nxt[mno])
            current = nxt
        final_winner = current[104]
        reach["champion"] = np.bincount(final_winner, minlength=len(self.teams))

        probs = pd.DataFrame(
            {k: v / self.n for k, v in reach.items()}, index=self.teams
        ).sort_values("champion", ascending=False)
        probs.insert(0, "group", [self.group_letters[self.teams.index(t) // 4]
                                  for t in probs.index])
        return {
            "probs": probs,
            "group_tables": self._group_tables(winners, runners, thirds),
            "ko_forecasts": self._ko_forecasts(),
            "n_sims": self.n,
        }

    # ----------------------------------------------------------- reportes
    def _group_tables(self, winners, runners, thirds) -> pd.DataFrame:
        """Por equipo: P(1º/2º/3º), P(3º clasifica), P(avanza), stats esperadas."""
        n, n_teams = self.n, len(self.teams)
        p1 = np.zeros(n_teams)
        p2 = np.zeros(n_teams)
        p3 = np.zeros(n_teams)
        p3q = np.zeros(n_teams)
        qual_bool = ((self._third_qual_masks[:, None] >> np.arange(12)) & 1).astype(bool)
        for gi in range(12):
            p1 += np.bincount(winners[:, gi], minlength=n_teams)
            p2 += np.bincount(runners[:, gi], minlength=n_teams)
            p3 += np.bincount(thirds[:, gi], minlength=n_teams)
            q = qual_bool[:, gi]
            p3q += np.bincount(thirds[q, gi], minlength=n_teams)
        df = pd.DataFrame({
            "group": [self.group_letters[i // 4] for i in range(n_teams)],
            "p_1st": p1 / n, "p_2nd": p2 / n, "p_3rd": p3 / n,
            "p_3rd_qualifies": p3q / n,
            "p_advance": (p1 + p2 + p3q) / n,
            "exp_pts": self._pts.mean(axis=0),
            "exp_gf": self._gf.mean(axis=0),
            "exp_gd": self._gd.mean(axis=0),
        }, index=self.teams)
        return df.sort_values(["group", "exp_pts"], ascending=[True, False])

    def _ko_forecasts(self, top_k: int = 4) -> pd.DataFrame:
        """Por partido KO: los cruces más probables y P(avanza) condicional."""
        n_t = len(self.teams)
        dates = self.t26.get("ko_dates", {})
        round_of = {}
        for stage, name in [("round_of_32", "R32"), ("round_of_16", "Octavos"),
                            ("quarter_finals", "Cuartos"), ("semi_finals", "Semifinal"),
                            ("final", "Final")]:
            for mno in self.t26[stage]:
                round_of[mno] = name
        rows = []
        for mno in sorted(self.ko_record):
            codes, winner = self.ko_record[mno]
            home_idx = codes // n_t
            uniq, counts = np.unique(codes, return_counts=True)
            order = np.argsort(-counts)[:top_k]
            home_wins = np.bincount(codes[winner == home_idx],
                                    minlength=int(uniq.max()) + 1)
            for rank, k in enumerate(order, start=1):
                code, cnt = int(uniq[k]), int(counts[k])
                h, a = self.teams[code // n_t], self.teams[code % n_t]
                rows.append({
                    "match": mno, "round": round_of[mno],
                    "date": str(dates.get(mno, "")),
                    "rank": rank, "p_pairing": cnt / self.n,
                    "home": h, "away": a,
                    "p_home_advances": home_wins[code] / cnt,
                })
        return pd.DataFrame(rows)
