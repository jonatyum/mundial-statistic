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
    cache = ROOT / settings()["paths"]["raw"] / f"odds_{stamp}.json"
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
