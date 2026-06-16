"""Cuotas de mercado (The Odds API) como benchmark vivo del modelo.

Requiere la variable de entorno ODDS_API_KEY (cuenta gratuita en
https://the-odds-api.com — 500 requests/mes). Sin key, degrada con un aviso:
el resto del sistema funciona igual.
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from mundial.config import ROOT, canonical, settings

SPORT = "soccer_fifa_world_cup"
URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"


def _odds_dir():
    """Carpeta de snapshots de cuotas. En data/predictions/ (versionada) para que
    se acumulen entre corridas del GitHub Action; data/raw/ está gitignored."""
    return ROOT / settings()["paths"]["predictions"] / "odds"


def fetch() -> pd.DataFrame | None:
    """Descarga cuotas h2h, las cachea en data/raw/ y devuelve probabilidades
    implícitas sin margen (un partido por fila). None si no hay API key."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY no configurada: se omite el benchmark de mercado. "
              "Registrate en https://the-odds-api.com y exportá la variable.")
        return None
    resp = requests.get(URL, params={
        "apiKey": key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal",
    }, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    cache = _odds_dir() / f"odds_{stamp}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw))
    return implied_probs(raw)


def implied_probs(raw: list[dict]) -> pd.DataFrame:
    """Promedia cuotas entre casas y quita el margen (normalización 1/odds)."""
    rows = []
    for event in raw:
        home = canonical(event["home_team"])
        away = canonical(event["away_team"])
        inv = {"home": [], "draw": [], "away": []}
        for bk in event.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in market["outcomes"]}
                if len(prices) < 3:
                    continue
                inv["home"].append(1 / prices.get(event["home_team"], float("inf")))
                inv["away"].append(1 / prices.get(event["away_team"], float("inf")))
                draw_price = prices.get("Draw")
                inv["draw"].append(1 / draw_price if draw_price else 0.0)
        if not inv["home"]:
            continue
        h = sum(inv["home"]) / len(inv["home"])
        d = sum(inv["draw"]) / len(inv["draw"])
        a = sum(inv["away"]) / len(inv["away"])
        z = h + d + a  # quita el margen del mercado
        rows.append({"commence": event["commence_time"], "home": home, "away": away,
                     "mkt_home": h / z, "mkt_draw": d / z, "mkt_away": a / z,
                     "n_bookmakers": len(inv["home"])})
    return pd.DataFrame(rows)


def _odds_snapshots() -> list[tuple]:
    """Snapshots de cuotas cacheados (data/raw/odds_*.json) ordenados ascendente
    por fecha, como (timestamp UTC, DataFrame de probabilidades implícitas)."""
    odds_dir = _odds_dir()
    snaps = []
    for p in sorted(odds_dir.glob("odds_*.json")):
        stamp = p.stem.replace("odds_", "")
        try:
            ts = pd.Timestamp(datetime.strptime(stamp, "%Y-%m-%dT%H%M"), tz="UTC")
            snaps.append((ts, implied_probs(json.loads(p.read_text()))))
        except (ValueError, json.JSONDecodeError):
            continue
    return snaps


def latest_market_probs() -> dict:
    """Probabilidades 1X2 del mercado del snapshot más reciente, para mezclarlas en
    el pronóstico de los próximos partidos. {(home, away): [p_home, p_draw, p_away]}."""
    snaps = _odds_snapshots()
    if not snaps:
        return {}
    _, ip = snaps[-1]
    return {(r.home, r.away): [float(r.mkt_home), float(r.mkt_draw), float(r.mkt_away)]
            for r in ip.itertuples()}


def market_accuracy(df: pd.DataFrame, season_start: str = "2026-06-01") -> dict | None:
    """Acierto del mercado sobre los partidos YA JUGADOS del Mundial 2026.

    Para cada partido toma el snapshot de cuotas más reciente ANTERIOR a su día
    (anti-fuga: nunca usa cuotas posteriores al inicio). Devuelve una fila estilo
    `live_model_comparison` (model='market') o None si no hay cobertura todavía."""
    import numpy as np

    from mundial.evaluate import metrics
    from mundial.ingest import results

    snaps = _odds_snapshots()
    if not snaps:
        return None
    played = results.played(df)
    played = played[(played["tournament"] == "FIFA World Cup")
                    & (played["date"] >= pd.Timestamp(season_start, tz="UTC"))]

    probs, outcomes = [], []
    for _, m in played.iterrows():
        best = None  # último snapshot previo al partido que tenga este cruce
        for ts, ip in snaps:
            if ts >= m.date:
                break  # snaps ascendentes: los siguientes son aún más tarde
            row = ip[(ip["home"] == m.home_team) & (ip["away"] == m.away_team)]
            if len(row):
                best = row.iloc[0]
        if best is None:
            continue
        probs.append([best.mkt_home, best.mkt_draw, best.mkt_away])
        outcomes.append(metrics.outcome_index(m.home_score, m.away_score))
    if not probs:
        return None
    probs, outcomes = np.array(probs), np.array(outcomes)
    return {"model": "market", "w": None, "n": int(len(outcomes)),
            "log_loss": metrics.log_loss(probs, outcomes),
            "brier": metrics.brier(probs, outcomes),
            "rps": metrics.rps(probs, outcomes),
            "accuracy": float((probs.argmax(axis=1) == outcomes).mean())}


def compare_with_model() -> pd.DataFrame | None:
    """Une cuotas con el pronóstico del modelo para ver discrepancias."""
    market = fetch()
    if market is None or market.empty:
        return None
    fc = pd.read_csv(ROOT / settings()["paths"]["processed"] / "match_forecasts.csv")
    merged = fc.merge(market, on=["home", "away"], how="inner")
    merged["delta_home"] = merged["p_home"] - merged["mkt_home"]
    out = merged[["date", "home", "away", "p_home", "mkt_home", "p_draw", "mkt_draw",
                  "p_away", "mkt_away", "delta_home", "n_bookmakers"]]
    out.to_csv(ROOT / settings()["paths"]["processed"] / "model_vs_market.csv", index=False)
    return out
