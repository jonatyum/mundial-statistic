"""Tests del benchmark/mezcla de mercado y de la comparación por-modelo."""
import numpy as np
import pandas as pd

from mundial.evaluate import backtest
from mundial.ingest import odds
from mundial.models.ensemble import blend_market


def test_blend_market_geometric_pool():
    p = np.array([0.5, 0.3, 0.2])
    mkt = [0.2, 0.3, 0.5]
    # peso 0 -> no toca el modelo
    assert np.allclose(blend_market(p, mkt, 0.0), p)
    # sin cuota -> no toca
    assert np.allclose(blend_market(p, None, 0.5), p)
    # peso 1 -> queda el mercado
    assert np.allclose(blend_market(p, mkt, 1.0), mkt)
    # peso intermedio -> mezcla normalizada que se acerca al mercado en la visita
    blended = blend_market(p, mkt, 0.5)
    assert abs(blended.sum() - 1.0) < 1e-12
    assert blended[2] > p[2]  # P(visita) sube hacia el mercado


def _played_df(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["neutral"] = True
    df["tournament"] = "FIFA World Cup"
    return df


def _snapshot(ts, rows):
    return pd.Timestamp(ts, tz="UTC"), pd.DataFrame(rows)


def test_market_accuracy_scores_pre_match_odds(monkeypatch):
    df = _played_df([
        {"date": "2026-06-13", "home_team": "Brazil", "away_team": "Morocco",
         "home_score": 2.0, "away_score": 0.0},
    ])
    # snapshot anterior al partido, con el mercado favoreciendo a Brasil (acierta)
    snaps = [_snapshot("2026-06-12T12:00", [
        {"home": "Brazil", "away": "Morocco",
         "mkt_home": 0.6, "mkt_draw": 0.25, "mkt_away": 0.15}])]
    monkeypatch.setattr(odds, "_odds_snapshots", lambda: snaps)

    row = odds.market_accuracy(df)
    assert row is not None
    assert row["model"] == "market" and row["n"] == 1
    assert row["accuracy"] == 1.0  # el favorito (local) ganó
    assert row["log_loss"] < np.log(3)  # mejor que el uniforme


def test_market_accuracy_ignores_post_match_snapshots(monkeypatch):
    df = _played_df([
        {"date": "2026-06-13", "home_team": "Brazil", "away_team": "Morocco",
         "home_score": 2.0, "away_score": 0.0},
    ])
    # snapshot POSTERIOR al inicio del partido: no se debe usar (anti-fuga)
    snaps = [_snapshot("2026-06-14T00:00", [
        {"home": "Brazil", "away": "Morocco",
         "mkt_home": 0.6, "mkt_draw": 0.25, "mkt_away": 0.15}])]
    monkeypatch.setattr(odds, "_odds_snapshots", lambda: snaps)
    assert odds.market_accuracy(df) is None


def test_market_accuracy_without_snapshots(monkeypatch):
    monkeypatch.setattr(odds, "_odds_snapshots", lambda: [])
    df = _played_df([
        {"date": "2026-06-13", "home_team": "Brazil", "away_team": "Morocco",
         "home_score": 2.0, "away_score": 0.0},
    ])
    assert odds.market_accuracy(df) is None


def test_live_model_comparison_empty_without_played_matches():
    # solo un partido futuro sin resultado -> nada que evaluar
    df = _played_df([
        {"date": "2026-06-30", "home_team": "Brazil", "away_team": "Morocco",
         "home_score": np.nan, "away_score": np.nan},
    ])
    assert backtest.live_model_comparison(df).empty
