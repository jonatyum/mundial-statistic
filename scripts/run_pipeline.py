"""Pipeline completo: ingesta -> modelos -> pronóstico partido a partido ->
simulación -> reporte.

Uso: .venv/bin/python scripts/run_pipeline.py [--sims N] [--fresh]
Re-ejecutable tras cada jornada: incorpora resultados nuevos automáticamente.
"""
import argparse
import subprocess
from datetime import datetime, timezone

import pandas as pd
import yaml

from mundial.config import ROOT, settings, tournament
from mundial.evaluate import backtest
from mundial.features import elo
from mundial.forecast import match_forecast
from mundial.ingest import results
from mundial.models.dixon_coles import DixonColes
from mundial.models.elo_model import EloDavidson
from mundial.simulate.tournament import TournamentSimulator


def model_version() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = ""
    return f"0.2.0+{rev or 'nogit'}"


def fmt_pct(x: float) -> str:
    return f"{x:.0%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="re-descargar datos")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    version = model_version()
    print(f"[1/6] Ingesta (versión {version})...")
    if args.fresh:
        results.download(force=True)
    df = results.load()
    try:
        from mundial.ingest.live import patch_results
        n_live = patch_results(df)
        if n_live:
            print(f"      +{n_live} resultados finales vía ESPN (aún no en el dataset)")
    except Exception as e:  # sin red u otro fallo: seguir con el dataset
        print(f"      marcadores en vivo no disponibles ({e})")
    train = results.before(df, now.isoformat())
    print(f"      {len(train)} partidos de entrenamiento hasta {now.date()}")

    print("[2/6] Peso del ensamble por backtest 2018+2022...")
    w = backtest.best_ensemble_weight(years=(2018, 2022), df=df)
    print(f"      w(elo)={w:.2f}, w(dixon_coles)={1 - w:.2f}")

    print("[3/6] Entrenando modelos con todo el histórico...")
    ratings = elo.compute(train)
    elo_model = EloDavidson(ratings).fit_nu(train)
    dc = DixonColes().fit(train, ref_date=now.isoformat())
    print(f"      nu={elo_model.nu:.3f}, gamma={dc.gamma:.3f}, rho={dc.rho:+.4f}, mu={dc.mu:.3f}")

    wc = df[(df["tournament"] == "FIFA World Cup")
            & (df["date"] >= pd.Timestamp("2026-06-01", tz="UTC"))].copy()
    print(f"[4/6] Pronóstico de los {len(wc)} partidos de fase de grupos...")
    team_group = {t: g for g, teams in tournament()["groups"].items() for t in teams}
    kickoffs = yaml.safe_load(
        (ROOT / "config" / "kickoff_times.yaml").read_text())["kickoffs_utc"]
    rows = []
    for _, m in wc.iterrows():
        fc = match_forecast(dc, elo_model, w, m.home_team, m.away_team, bool(m.neutral))
        played = pd.notna(m.home_score)
        key = "|".join(sorted([m.home_team, m.away_team]))
        rows.append({
            "date": m.date.date(), "kickoff_utc": kickoffs.get(key, ""),
            "group": team_group[m.home_team],
            "home": m.home_team, "away": m.away_team, "city": m.city,
            "status": "jugado" if played else "pendiente",
            "result": f"{int(m.home_score)}-{int(m.away_score)}" if played else "",
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in fc.items()},
            "model_version": version, "frozen_at": now.isoformat(),
        })
    preds = pd.DataFrame(rows).sort_values("kickoff_utc")
    proc = ROOT / settings()["paths"]["processed"]
    proc.mkdir(parents=True, exist_ok=True)
    preds.to_csv(proc / "match_forecasts.csv", index=False)

    # prediction log append-only: congela solo partidos pendientes aún no registrados
    log_path = ROOT / settings()["paths"]["predictions"] / "prediction_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pend = preds[preds["status"] == "pendiente"].drop(columns=["status", "result"])
    if log_path.exists():
        log = pd.read_csv(log_path)
        seen = set(zip(log["home"], log["away"]))
        new = pend[[(h, a) not in seen for h, a in zip(pend["home"], pend["away"])]]
        if len(new):
            pd.concat([log, new], ignore_index=True).to_csv(log_path, index=False)
    else:
        pend.to_csv(log_path, index=False)

    print("[5/6] Simulación Monte Carlo del torneo...")
    sim = TournamentSimulator(dc, elo_model, w, wc, n_sims=args.sims)
    out = sim.run()
    probs, groups_df, ko = out["probs"], out["group_tables"], out["ko_forecasts"]
    # pronóstico de goles para cada cruce KO posible (sede neutral)
    ko_fc = [match_forecast(dc, elo_model, w, r.home, r.away, neutral=True)
             for _, r in ko.iterrows()]
    ko["xg_home"] = [round(f["xg_home"], 2) for f in ko_fc]
    ko["xg_away"] = [round(f["xg_away"], 2) for f in ko_fc]
    ko["score_mode"] = [f["score_1"] for f in ko_fc]
    ko["p_score_mode"] = [round(f["p_score_1"], 4) for f in ko_fc]
    probs.round(4).to_csv(proc / "tournament_probabilities.csv")
    groups_df.round(4).to_csv(proc / "group_tables.csv")
    ko.round(4).to_csv(proc / "knockout_forecasts.csv", index=False)

    # historial de P(campeón) para la "carrera al título" (reemplaza el día actual)
    hist_path = proc / "champion_history.csv"
    today_hist = probs.reset_index(names="team")[["team", "group", "champion", "final", "sf"]]
    today_hist.insert(0, "date", str(now.date()))
    today_hist["model_version"] = version
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist = hist[hist["date"] != str(now.date())]
        today_hist = pd.concat([hist, today_hist], ignore_index=True)
    today_hist.to_csv(hist_path, index=False)

    # calibración en vivo: predicciones congeladas vs resultados reales
    from mundial.evaluate.calibration import calibration_table
    calib = calibration_table(df)
    calib.round(4).to_csv(proc / "calibration.csv", index=False)
    if len(calib):
        print(f"      calibración: {len(calib)} partidos | "
              f"log-loss acum {calib['log_loss_acum'].iloc[-1]:.4f} (uniforme 1.0986)")

    # snapshot fechado del estado completo (sección 9.5 del plan)
    snap = ROOT / settings()["paths"]["snapshots"] / str(now.date())
    snap.mkdir(parents=True, exist_ok=True)
    for f in proc.glob("*.csv"):
        (snap / f.name).write_bytes(f.read_bytes())

    print("[6/6] Reporte partido a partido...")
    L = [
        "# Mundial 2026 — Pronóstico partido a partido",
        f"\nGenerado: {now:%Y-%m-%d %H:%M UTC} · modelo `{version}` · "
        f"{out['n_sims']:,} simulaciones · ensamble w(elo)={w:.2f}\n",
        "## Fase de grupos (72 partidos)\n",
    ]
    for g in sorted(team_group.values() if False else set(preds["group"])):
        L.append(f"### Grupo {g}\n")
        L.append("| Fecha (UTC) | Partido | 1 | X | 2 | Goles esperados | Marcadores probables | +2.5 | Ambos anotan |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        sub = preds[preds["group"] == g].sort_values("kickoff_utc")
        for _, r in sub.iterrows():
            res = f" ({r.result})" if r.status == "jugado" else ""
            hora = (f"{r.date} {pd.Timestamp(r.kickoff_utc):%H:%M}"
                    if r.kickoff_utc else str(r.date))
            scorelines = (f"{r.score_1} ({r.p_score_1:.0%}), {r.score_2} ({r.p_score_2:.0%}), "
                          f"{r.score_3} ({r.p_score_3:.0%})")
            L.append(f"| {hora} | **{r.home}** vs **{r.away}**{res} | {fmt_pct(r.p_home)} "
                     f"| {fmt_pct(r.p_draw)} | {fmt_pct(r.p_away)} | {r.xg_home:.2f} - {r.xg_away:.2f} "
                     f"| {scorelines} | {fmt_pct(r['p_over_2.5'])} | {fmt_pct(r.p_btts)} |")
        # mini tabla del grupo
        L.append("\n| Posición esperada | Pts esp. | P(1º) | P(2º) | P(3º clasifica) | P(avanza) |")
        L.append("|---|---|---|---|---|---|")
        for team, t in groups_df[groups_df["group"] == g].iterrows():
            L.append(f"| {team} | {t.exp_pts:.2f} | {fmt_pct(t.p_1st)} | {fmt_pct(t.p_2nd)} "
                     f"| {fmt_pct(t.p_3rd_qualifies)} | {fmt_pct(t.p_advance)} |")
        L.append("")

    L.append("## Eliminatorias — cruces más probables y pronóstico\n")
    for rnd in ["R32", "Octavos", "Cuartos", "Semifinal", "Final"]:
        sub = ko[ko["round"] == rnd]
        if not len(sub):
            continue
        L.append(f"### {rnd}\n")
        L.append("| Partido | Fecha | Cruce más probable | P(cruce) | Avanza | Goles esperados | Marcador 90' | Alternativas |")
        L.append("|---|---|---|---|---|---|---|---|")
        for mno in sorted(sub["match"].unique()):
            mm = sub[sub["match"] == mno].sort_values("rank")
            top = mm.iloc[0]
            fav = top.home if top.p_home_advances >= 0.5 else top.away
            p_fav = max(top.p_home_advances, 1 - top.p_home_advances)
            alts = "; ".join(f"{r.home}–{r.away} ({r.p_pairing:.0%})"
                             for _, r in mm.iloc[1:3].iterrows())
            L.append(f"| {mno} | {top.date} | **{top.home}** vs **{top.away}** "
                     f"| {top.p_pairing:.0%} | {fav} ({p_fav:.0%}) "
                     f"| {top.xg_home:.2f} - {top.xg_away:.2f} "
                     f"| {top.score_mode} ({top.p_score_mode:.0%}) | {alts} |")
        L.append("")

    L.append("## Probabilidad de campeón (referencia)\n")
    L.append("| Selección | Grupo | Campeón | Final | Semis |")
    L.append("|---|---|---|---|---|")
    for team, r in probs.head(10).iterrows():
        L.append(f"| {team} | {r.group} | {r.champion:.1%} | {r['final']:.1%} | {r.sf:.1%} |")
    (ROOT / "REPORT.md").write_text("\n".join(L) + "\n")
    print("      match_forecasts.csv, group_tables.csv, knockout_forecasts.csv,")
    print("      tournament_probabilities.csv, REPORT.md, prediction_log.csv")


if __name__ == "__main__":
    main()
