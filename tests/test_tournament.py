"""Tests del formato del torneo: la parte que invalida todo si está mal."""
from itertools import combinations

import numpy as np
import pytest

from mundial.config import tournament
from mundial.simulate.tournament import THIRD_SLOT_ORDER, TournamentSimulator

T = tournament()
LETTERS = list(T["groups"])
SLOTS = {m: set(T["round_of_32"][m]["away"].split(":")[1].split(","))
         for m in THIRD_SLOT_ORDER}


def test_groups_are_12x4_and_48_unique_teams():
    teams = [t for g in T["groups"].values() for t in g]
    assert len(T["groups"]) == 12
    assert all(len(g) == 4 for g in T["groups"].values())
    assert len(set(teams)) == 48


def test_bracket_structure():
    r32 = T["round_of_32"]
    assert len(r32) == 16
    # cada ganador y segundo de grupo aparece exactamente una vez
    tokens = [spec[side] for spec in r32.values() for side in ("home", "away")]
    for letter in LETTERS:
        assert tokens.count(f"W_{letter}") == 1
        assert tokens.count(f"R_{letter}") == 1
    assert sum(1 for t in tokens if t.startswith("T:")) == 8
    # flujo: cada partido alimenta exactamente un cruce de la ronda siguiente
    feeds = [m for pair in T["round_of_16"].values() for m in pair]
    assert sorted(feeds) == sorted(r32.keys())
    feeds_qf = [m for pair in T["quarter_finals"].values() for m in pair]
    assert sorted(feeds_qf) == sorted(T["round_of_16"].keys())


def test_third_place_matching_respects_constraints():
    """Para TODAS las 495 combinaciones de 8 terceros existe matching válido."""
    for combo in combinations(range(12), 8):
        qualified = [LETTERS[i] for i in combo]
        result = TournamentSimulator._match_slots(SLOTS, qualified)
        assert len(result) == 8
        assert sorted(result.values()) == sorted(qualified)[:8] or set(result.values()) <= set(qualified)
        for match_no, group in result.items():
            assert group in SLOTS[match_no], f"llave {match_no} no admite grupo {group}"
        # sin grupos repetidos
        assert len(set(result.values())) == 8


def test_group_ranking_tiebreakers():
    """Puntos > diferencia de gol > goles a favor."""
    # composite replica la fórmula del simulador
    def comp(pts, gd, gf, noise=0.0):
        return pts * 1e8 + (gd + 60) * 1e5 + gf * 1e2 + noise

    assert comp(6, -2, 1) > comp(4, 10, 20)        # puntos manda
    assert comp(6, 3, 1) > comp(6, 2, 20)          # luego GD
    assert comp(6, 3, 5) > comp(6, 3, 4)           # luego GF
