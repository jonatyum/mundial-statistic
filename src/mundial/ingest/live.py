"""Marcadores en vivo del Mundial 2026 vía el scoreboard público de ESPN.

Sin API key. Estados: pre (no empezó) / in (en juego) / post (terminado).
Los días ya terminados se cachean en disco; el día actual siempre se consulta.
"""
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from mundial.config import ROOT, canonical, settings

URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
TOURNAMENT_START = date(2026, 6, 11)


def _all_finished(events: list) -> bool:
    return bool(events) and all(
        ev["competitions"][0]["status"]["type"]["state"] == "post" for ev in events)


def fetch_day(d: date, use_cache: bool = True) -> list[dict]:
    """Partidos de un día (estado + marcador).

    OJO: ESPN agrupa por fecha de EE.UU. (Eastern), no UTC — un día "pasado"
    en UTC puede tener partidos en juego. Solo se cachea/sirve de caché un día
    cuando TODOS sus partidos están terminados."""
    cache = ROOT / settings()["paths"]["raw"] / "espn" / f"{d:%Y%m%d}.json"
    events = None
    if use_cache and cache.exists():
        cached = json.loads(cache.read_text())
        if _all_finished(cached):
            events = cached
    if events is None:
        resp = requests.get(URL, params={"dates": f"{d:%Y%m%d}"}, timeout=10)
        resp.raise_for_status()
        events = resp.json().get("events", [])
        if _all_finished(events):  # día realmente cerrado: cachear
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(events))
    rows = []
    for ev in events:
        comp = ev["competitions"][0]
        status = comp["status"]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        if "home" not in sides or "away" not in sides:
            continue
        rows.append({
            "home": canonical(sides["home"]["team"]["displayName"]),
            "away": canonical(sides["away"]["team"]["displayName"]),
            "home_score": int(sides["home"].get("score") or 0),
            "away_score": int(sides["away"].get("score") or 0),
            "state": status["type"]["state"],          # pre | in | post
            "detail": status["type"].get("shortDetail", ""),  # FT, 65', etc.
        })
    return rows


def scoreboard(days_back: int = 1) -> dict[str, dict]:
    """Estado actual de los partidos recientes: {clave 'A|B' ordenada: registro}."""
    today = datetime.now(timezone.utc).date()
    board = {}
    for delta in range(days_back, -1, -1):
        try:
            for r in fetch_day(today - timedelta(days=delta)):
                board["|".join(sorted([r["home"], r["away"]]))] = r
        except requests.RequestException:
            continue  # sin red: el dashboard degrada a los datos del pipeline
    return board


def patch_results(df: pd.DataFrame) -> int:
    """Completa en memoria los marcadores FINALES de ESPN que el dataset
    histórico aún no publicó. Devuelve cuántos partidos completó."""
    today = datetime.now(timezone.utc).date()
    finals = {}
    d = TOURNAMENT_START
    while d <= today:
        try:
            for r in fetch_day(d):
                if r["state"] == "post":
                    finals["|".join(sorted([r["home"], r["away"]]))] = r
        except requests.RequestException:
            return 0
        d += timedelta(days=1)
    if not finals:
        return 0

    wc = (df["tournament"] == "FIFA World Cup") & df["home_score"].isna() \
         & (df["date"] >= pd.Timestamp("2026-06-01", tz="UTC")) \
         & (df["date"] <= pd.Timestamp(today, tz="UTC"))
    patched = 0
    for idx in df[wc].index:
        key = "|".join(sorted([df.at[idx, "home_team"], df.at[idx, "away_team"]]))
        r = finals.get(key)
        if r is None:
            continue
        if r["home"] == df.at[idx, "home_team"]:
            df.at[idx, "home_score"], df.at[idx, "away_score"] = r["home_score"], r["away_score"]
        else:  # ESPN listó el local al revés
            df.at[idx, "home_score"], df.at[idx, "away_score"] = r["away_score"], r["home_score"]
        patched += 1
    return patched
