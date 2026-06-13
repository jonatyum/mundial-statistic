"""Test del ciclo diario: predicción congelada -> resultado real -> evaluación."""
import numpy as np
import pandas as pd

from mundial.evaluate.calibration import calibration_table


def _fake_world(results_rows, log_rows):
    df = pd.DataFrame(results_rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["neutral"] = True
    df["tournament"] = "FIFA World Cup"
    return df, pd.DataFrame(log_rows)


def test_calibration_joins_and_scores():
    df, log = _fake_world(
        results_rows=[
            {"date": "2026-06-11", "home_team": "Mexico", "away_team": "South Africa",
             "home_score": 2.0, "away_score": 0.0, "city": "x", "country": "x"},
            # partido futuro sin resultado: no debe aparecer
            {"date": "2026-06-18", "home_team": "Mexico", "away_team": "South Korea",
             "home_score": np.nan, "away_score": np.nan, "city": "x", "country": "x"},
        ],
        log_rows=[
            {"date": "2026-06-11", "home": "Mexico", "away": "South Africa",
             "p_home": 0.67, "p_draw": 0.22, "p_away": 0.11, "model_version": "test"},
            {"date": "2026-06-18", "home": "Mexico", "away": "South Korea",
             "p_home": 0.53, "p_draw": 0.26, "p_away": 0.22, "model_version": "test"},
        ],
    )
    out = calibration_table(df, log)
    assert len(out) == 1  # solo el partido terminado
    row = out.iloc[0]
    assert row["result"] == "2-0"
    assert row["outcome"] == "local"
    assert row["hit"]  # el más probable (local 67%) salió
    assert abs(row["log_loss"] - (-np.log(0.67))) < 1e-9
    assert row["log_loss"] < np.log(3)  # mejor que el uniforme


def test_calibration_empty_when_no_results():
    df, log = _fake_world(
        results_rows=[{"date": "2026-06-18", "home_team": "A", "away_team": "B",
                       "home_score": np.nan, "away_score": np.nan,
                       "city": "x", "country": "x"}],
        log_rows=[{"date": "2026-06-18", "home": "A", "away": "B",
                   "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3, "model_version": "t"}],
    )
    assert calibration_table(df, log).empty
