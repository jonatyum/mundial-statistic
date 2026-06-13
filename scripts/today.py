"""Vista diaria: partidos del día con pronóstico congelado y, si ya terminaron,
resultado real + evaluación.

Uso: .venv/bin/python scripts/today.py [YYYY-MM-DD]   (default: hoy UTC)
"""
import sys
from datetime import datetime, timezone

import pandas as pd

from mundial.config import ROOT, settings
from mundial.evaluate.calibration import calibration_table

day = sys.argv[1] if len(sys.argv) > 1 else str(datetime.now(timezone.utc).date())
proc = ROOT / settings()["paths"]["processed"]

fc = pd.read_csv(proc / "match_forecasts.csv")
day_matches = fc[fc["date"] == day]
if day_matches.empty:
    print(f"No hay partidos del Mundial el {day}.")
    sys.exit(0)

if "kickoff_utc" in day_matches.columns:
    day_matches = day_matches.sort_values("kickoff_utc", na_position="last")

print(f"=== Partidos del {day} ===\n")
for _, r in day_matches.iterrows():
    hora = ""
    if "kickoff_utc" in r and isinstance(r.kickoff_utc, str) and r.kickoff_utc:
        t = pd.Timestamp(r.kickoff_utc).tz_convert(datetime.now().astimezone().tzinfo)
        hora = f" · {t:%H:%M} hora local"
    print(f"{r.home} vs {r.away}  (Grupo {r.group}, {r.city}{hora})")
    print(f"  1X2: {r.p_home:.0%} / {r.p_draw:.0%} / {r.p_away:.0%}"
          f"  | goles esperados: {r.xg_home:.2f} - {r.xg_away:.2f}")
    print(f"  marcadores: {r.score_1} ({r.p_score_1:.0%}), {r.score_2} ({r.p_score_2:.0%}),"
          f" {r.score_3} ({r.p_score_3:.0%}) | +2.5: {r['p_over_2.5']:.0%}"
          f" | ambos anotan: {r.p_btts:.0%}")
    if r.status == "jugado":
        print(f"  >>> RESULTADO REAL: {r.result}")
    print()

calib = calibration_table()
if len(calib):
    day_calib = calib[calib["date"] == day]
    if len(day_calib):
        print(f"=== Evaluación del día ({len(day_calib)} partidos terminados) ===")
        for _, r in day_calib.iterrows():
            ok = "ACIERTO" if r.hit else "fallo"
            print(f"{r.home} {r.result} {r.away}: salió '{r.outcome}'"
                  f" | log-loss {r.log_loss:.3f} | {ok}")
        print(f"\nAcumulado torneo: log-loss {calib['log_loss_acum'].iloc[-1]:.4f}"
              f" (uniforme 1.0986) | aciertos {calib['hit'].mean():.0%}"
              f" ({int(calib['hit'].sum())}/{len(calib)})")
else:
    print("Aún sin partidos terminados en el dataset: la evaluación aparecerá "
          "cuando la fuente publique los resultados (corridas de 07:00/23:30).")
