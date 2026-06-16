"""Dashboard Streamlit (solo lectura, sección 12 del plan).

Uso: .venv/bin/streamlit run src/mundial/serve/dashboard.py
Diseño: design system "Mundial 2026" (design/ y claude.ai/design).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
# que `import mundial.*` funcione en Streamlit Cloud sin editable-install
sys.path.insert(0, str(ROOT / "src"))

C = {"bg": "#0d1117", "surface": "#161b27", "s2": "#1e2433", "border": "#2a3242",
     "text": "#e6edf3", "muted": "#8b949e", "home": "#22c55e", "draw": "#64748b",
     "away": "#3b82f6", "gold": "#eab308", "danger": "#ef4444"}

FLAG = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Turkey": "🇹🇷", "Australia": "🇦🇺",
    "Germany": "🇩🇪", "Ecuador": "🇪🇨", "Ivory Coast": "🇨🇮", "Curaçao": "🇨🇼",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Iran": "🇮🇷", "Egypt": "🇪🇬", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Uruguay": "🇺🇾", "Saudi Arabia": "🇸🇦", "Cape Verde": "🇨🇻",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Norway": "🇳🇴", "Iraq": "🇮🇶",
    "Argentina": "🇦🇷", "Austria": "🇦🇹", "Algeria": "🇩🇿", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "Colombia": "🇨🇴", "Uzbekistan": "🇺🇿", "DR Congo": "🇨🇩",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Panama": "🇵🇦", "Ghana": "🇬🇭",
}


# ---- i18n ----
from mundial.config import model_info, team_names_es  # noqa: E402

NAMES_ES = team_names_es()
TR = {
    "es": {
        "title": "Mundial 2026 — Predicciones",
        "updated": "Actualizado", "model": "modelo", "language": "Idioma",
        "tab_matches": "⚽ Partidos", "tab_groups": "📊 Grupos", "tab_ko": "🏆 Eliminatorias",
        "tab_title": "🥇 Carrera al título", "tab_health": "📈 Salud del modelo", "tab_info": "ℹ️ Información",
        "group": "Grupo", "vs": "vs", "xg": "goles esperados", "draw": "empate",
        "over25_chip": "+2.5 goles", "btts_chip": "ambos anotan",
        "live": "EN VIVO", "final": "FINAL", "inplay": "en juego", "finalresult": "resultado final",
        "seedetail": "Ver detalle del partido",
        "comparison": "Comparativa", "attack": "Ataque", "defense": "Solidez def.", "xg_short": "Goles esp.",
        "markets": "Mercados", "win": "Gana", "draw_cap": "Empate", "over25": "Más de 2.5 goles",
        "under25": "Menos de 2.5 goles", "btts_cap": "Ambos anotan", "likely": "Marcadores más probables",
        "f_group": "Grupo", "f_status": "Estado", "f_date": "Fecha",
        "all_m": "Todos", "all_f": "Todas", "st_pending": "pendiente", "st_live": "en vivo", "st_played": "jugado",
        "nofilter": "No hay partidos con esos filtros.",
        "livebanner": "🔴 {n} partido(s) en juego — marcador actualizado cada 60 s",
        "matches_n": "{n} partido{s}", "plural": "s",
        "th_team": "Equipo", "th_exp_pts": "Pts esp.", "th_gf": "GF esp.", "th_gd": "DG esp.",
        "th_1st": "P(1º)", "th_2nd": "P(2º)", "th_3rd": "P(3º clasif.)", "th_advance": "P(avanza)",
        "g_paths": "Grupo {g} — vías de clasificación", "probability": "probabilidad",
        "round": "Ronda", "ko_match": "Partido", "ko_pairing": "P(cruce)", "ko_advances": "Avanza", "ko_alts": "Cruces alternativos por llave",
        "race_byday": "P(campeón) jornada a jornada — top 10", "race_now": "P(campeón) hoy", "pchamp": "P(campeón)",
        "health_wait": "⏳ Aún no hay partidos terminados: la calibración en vivo aparecerá tras la primera jornada (corridas automáticas de 07:00 y 23:30).",
        "evaluated": "Partidos evaluados", "logloss_cum": "Log-loss acumulado", "vs_uniform": "vs uniforme",
        "hits": "Aciertos (resultado modal)", "cal_title": "Log-loss acumulado vs baseline uniforme",
        "cal_xlabel": "partido nº", "cal_ylabel": "log-loss",
        "cmp_title": "¿Qué modelo/fuente acierta más?", "th_model": "Modelo / fuente", "th_logloss2": "Log-loss",
        "th_n": "N", "th_brier": "Brier", "th_rps": "RPS", "th_acc": "Acierto",
        "cmp_caption": "Modelos sobre {n} partidos jugados · entrenamiento anti-fuga (datos previos al torneo). Menor log-loss = mejores probabilidades; muy ruidoso con pocos partidos.",
        "cmp_market_note": "El mercado (casas de apuestas) se mide solo sobre los partidos con cuota capturada antes del inicio (ver columna N).",
        "mlabels": {"dixon_coles": "Dixon-Coles", "elo": "Elo-Davidson", "ensemble": "Ensamble", "uniform": "Uniforme (baseline)", "market": "Mercado (casas)"},
        "footer": "Predicciones congeladas al inicio de cada partido · prediction log público · ensamble Elo-Davidson + Dixon-Coles · 100k simulaciones Monte Carlo",
        "info_intro": "Resumen", "info_models": "Modelos", "info_formula": "Fórmula", "info_params": "Parámetros",
        "info_sources": "Fuentes de datos", "info_pipeline": "Cómo se actualiza", "info_optional": "opcional",
        "no_data": "No hay pronósticos generados: corré `scripts/run_pipeline.py` primero.",
        "days": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
        "months": ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
        "rounds": {"R32": "R32", "Octavos": "Octavos", "Cuartos": "Cuartos", "Semifinal": "Semifinal", "Final": "Final"},
    },
    "en": {
        "title": "World Cup 2026 — Predictions",
        "updated": "Updated", "model": "model", "language": "Language",
        "tab_matches": "⚽ Matches", "tab_groups": "📊 Groups", "tab_ko": "🏆 Knockouts",
        "tab_title": "🥇 Title race", "tab_health": "📈 Model health", "tab_info": "ℹ️ Info",
        "group": "Group", "vs": "vs", "xg": "expected goals", "draw": "draw",
        "over25_chip": "over 2.5", "btts_chip": "both score",
        "live": "LIVE", "final": "FINAL", "inplay": "in play", "finalresult": "final score",
        "seedetail": "See match detail",
        "comparison": "Comparison", "attack": "Attack", "defense": "Defence", "xg_short": "xG",
        "markets": "Markets", "win": "Win", "draw_cap": "Draw", "over25": "Over 2.5 goals",
        "under25": "Under 2.5 goals", "btts_cap": "Both teams score", "likely": "Most likely scores",
        "f_group": "Group", "f_status": "Status", "f_date": "Date",
        "all_m": "All", "all_f": "All", "st_pending": "pending", "st_live": "live", "st_played": "played",
        "nofilter": "No matches for those filters.",
        "livebanner": "🔴 {n} match(es) in play — score refreshes every 60 s",
        "matches_n": "{n} match{s}", "plural": "es",
        "th_team": "Team", "th_exp_pts": "Exp. pts", "th_gf": "Exp. GF", "th_gd": "Exp. GD",
        "th_1st": "P(1st)", "th_2nd": "P(2nd)", "th_3rd": "P(3rd qual.)", "th_advance": "P(advance)",
        "g_paths": "Group {g} — qualification paths", "probability": "probability",
        "round": "Round", "ko_match": "Match", "ko_pairing": "P(pairing)", "ko_advances": "Advances", "ko_alts": "Alternative pairings per bracket",
        "race_byday": "P(champion) matchday by matchday — top 10", "race_now": "P(champion) today", "pchamp": "P(champion)",
        "health_wait": "⏳ No finished matches yet: live calibration will appear after the first matchday (automatic runs at 07:00 and 23:30).",
        "evaluated": "Matches evaluated", "logloss_cum": "Cumulative log-loss", "vs_uniform": "vs uniform",
        "hits": "Modal-result hits", "cal_title": "Cumulative log-loss vs uniform baseline",
        "cal_xlabel": "match no.", "cal_ylabel": "log-loss",
        "cmp_title": "Which model/source is most accurate?", "th_model": "Model / source", "th_logloss2": "Log-loss",
        "th_n": "N", "th_brier": "Brier", "th_rps": "RPS", "th_acc": "Hit rate",
        "cmp_caption": "Models over {n} played matches · leak-free training (pre-tournament data only). Lower log-loss = better probabilities; very noisy with few matches.",
        "cmp_market_note": "The market (bookmakers) is scored only on matches with odds captured before kickoff (see column N).",
        "mlabels": {"dixon_coles": "Dixon-Coles", "elo": "Elo-Davidson", "ensemble": "Ensemble", "uniform": "Uniform (baseline)", "market": "Market (bookmakers)"},
        "footer": "Predictions frozen at kickoff · public prediction log · Elo-Davidson + Dixon-Coles ensemble · 100k Monte Carlo simulations",
        "info_intro": "Summary", "info_models": "Models", "info_formula": "Formula", "info_params": "Parameters",
        "info_sources": "Data sources", "info_pipeline": "How it updates", "info_optional": "optional",
        "no_data": "No forecasts generated yet: run `scripts/run_pipeline.py` first.",
        "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "months": ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        "rounds": {"R32": "R32", "Octavos": "Round of 16", "Cuartos": "Quarter-finals", "Semifinal": "Semi-finals", "Final": "Final"},
    },
}


def T(key: str):
    return TR.get(LANG, TR["es"]).get(key, TR["es"].get(key, key))


def localize(value):
    """Cadena directa o dict {es,en} -> texto en el idioma activo."""
    if isinstance(value, dict):
        return value.get(LANG) or value.get("es") or value.get("en") or ""
    return value or ""


def team_disp(name: str) -> str:
    """Nombre traducido según idioma (canónico en inglés)."""
    return NAMES_ES.get(name, name) if LANG == "es" else name


def team(name: str) -> str:
    return f"{FLAG.get(name, '')} {team_disp(name)}".strip()


st.set_page_config(page_title="Mundial 2026 — Predicciones", page_icon="⚽", layout="wide")

if "lang" not in st.session_state:
    st.session_state.lang = "es"
LANG = st.session_state.lang

st.markdown(f"""<style>
.block-container {{padding-top: 2.2rem; max-width: 1250px}}
.strip {{height:4px;border-radius:2px;margin-bottom:14px;
  background:linear-gradient(90deg,#006847 0 33%,#3c3b6e 33% 66%,#d80621 66% 100%)}}
.mcard {{background:{C['surface']};border:1px solid {C['border']};border-radius:16px;
  padding:16px 18px;margin-bottom:14px;font-variant-numeric:tabular-nums;
  box-shadow:0 2px 10px rgba(0,0,0,.25);transition:transform .12s, border-color .12s}}
.mcard:hover {{transform:translateY(-2px);border-color:#3d4a61}}
.mlegend {{display:flex;justify-content:space-between;font-size:11px;color:{C['muted']};
  margin-top:5px;padding:0 2px}}
.mlegend b {{font-weight:600}}
.datehdr {{font-size:14px;font-weight:700;color:{C['text']};margin:6px 0 10px;
  display:flex;align-items:center;gap:10px}}
.datehdr::after {{content:"";flex:1;height:1px;background:{C['border']}}}
.datehdr small {{color:{C['muted']};font-weight:400}}
.mlabel {{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:{C['muted']};
  display:flex;justify-content:space-between}}
.mteams {{display:flex;align-items:center;justify-content:space-between;margin:8px 0 2px}}
.mteam {{font-size:16px;font-weight:700}}
.mvs {{color:{C['muted']};font-size:11px}}
.mxg {{text-align:center;color:{C['muted']};font-size:11px;margin-bottom:8px}}
.mxg b {{color:{C['text']};font-size:14px}}
.mbar {{display:flex;height:24px;border-radius:7px;overflow:hidden;font-size:12px;font-weight:700}}
.mbar div {{display:flex;align-items:center;justify-content:center;color:#0d1117;min-width:34px}}
.chips {{display:flex;gap:5px;flex-wrap:wrap;margin-top:9px}}
.chip {{font-size:12px;padding:2px 10px;border-radius:999px;background:{C['s2']};
  border:1px solid {C['border']};color:{C['text']}}}
.chip.hot {{border-color:{C['gold']};color:{C['gold']}}}
.chip.ok {{border-color:{C['home']};color:{C['home']}}}
.chip.bad {{border-color:{C['danger']};color:{C['danger']}}}
.chip.live {{border-color:{C['danger']};color:{C['danger']};font-weight:700;
  animation:pulse 1.6s ease-in-out infinite}}
@keyframes pulse {{50% {{opacity:.55}}}}
.result {{font-size:26px;font-weight:800;text-align:center;margin:4px 0 6px;letter-spacing:.04em}}
.result small {{font-size:12px;color:{C['muted']};font-weight:400;display:block}}
/* desplegable de detalle */
.dsec {{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:{C['muted']};
  margin:6px 0 8px}}
.cmp {{display:grid;grid-template-columns:46px 1fr 46px;gap:8px;align-items:center;
  margin-bottom:9px;font-variant-numeric:tabular-nums}}
.cmp .vl {{text-align:right;font-weight:700;font-size:13px}}
.cmp .vr {{text-align:left;font-weight:700;font-size:13px}}
.cmp .mid {{text-align:center}}
.cmp .mid small {{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:{C['muted']};display:block;margin-bottom:3px}}
.cmpbar {{display:flex;height:8px;border-radius:4px;overflow:hidden;background:{C['s2']}}}
.cmpbar i {{display:block;height:100%}}
.mkt {{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {C['s2']};font-size:13px}}
.mkt:last-child {{border:none}}
.mkt b {{font-variant-numeric:tabular-nums}}
.slist {{display:flex;flex-direction:column;gap:5px;margin-top:4px}}
.slrow {{display:flex;align-items:center;gap:8px;font-size:13px;font-variant-numeric:tabular-nums}}
.slrow .sb {{flex:1;height:7px;border-radius:4px;background:{C['s2']};overflow:hidden}}
.slrow .sb i {{display:block;height:100%;background:{C['gold']}}}
.slrow .sn {{width:42px;font-weight:700}}
.slrow .sp {{width:38px;text-align:right;color:{C['muted']}}}
/* desplegable nativo dentro del box */
details.din {{margin-top:12px;border-top:1px solid {C['s2']};padding-top:8px}}
details.din summary {{cursor:pointer;list-style:none;font-size:12px;font-weight:600;
  color:{C['away']};display:flex;align-items:center;gap:6px;user-select:none}}
details.din summary::-webkit-details-marker {{display:none}}
details.din summary::before {{content:"▸";transition:transform .15s}}
details.din[open] summary::before {{transform:rotate(90deg)}}
details.din summary:hover {{color:{C['text']}}}
details.din .dbody {{margin-top:12px}}
.kpair {{display:flex;align-items:center;justify-content:space-between;background:{C['s2']};
  border-radius:10px;padding:9px 12px;margin-bottom:7px;font-weight:700}}
.kpair span:last-child {{font-weight:400;font-size:11px;color:{C['muted']}}}
h1, h2, h3 {{letter-spacing:-.01em}}
[data-testid="stMetricValue"] {{font-variant-numeric:tabular-nums}}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load(name: str):
    path = PROC / f"{name}.csv"
    if not path.exists():
        return None, None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()
    updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return df, updated


@st.cache_data(ttl=60, show_spinner=False)
def load_live() -> dict:
    """Marcadores en vivo (ESPN, sin key). Cache 60 s."""
    try:
        from mundial.ingest.live import scoreboard
        return scoreboard(days_back=1)
    except Exception:
        return {}


def html(s: str) -> str:
    """Colapsa el HTML a una línea: Streamlit trata las líneas indentadas
    como bloque de código markdown y mostraría el HTML crudo."""
    return " ".join(s.split())


def hora_local(kickoff_utc) -> str:
    """Hora de inicio convertida a la zona horaria de esta máquina."""
    if not kickoff_utc or pd.isna(kickoff_utc):
        return ""
    t = pd.Timestamp(kickoff_utc).tz_convert(datetime.now().astimezone().tzinfo)
    return f"{t:%H:%M}"


def match_card(r, live: dict | None = None, sig: str = "") -> str:
    """live: registro del scoreboard ESPN para este partido (o None).
    sig: firma de los filtros activos; al cambiar fuerza re-render del DOM
    para que los <details> abiertos se contraigan."""
    hora = hora_local(r.get("kickoff_utc", ""))
    reloj = f" · 🕐 {hora}" if hora else ""

    # estado y marcador: en vivo > final (ESPN o dataset) > programado
    marcador = ""
    if live is not None and live["home"] == r["home"]:
        hs, as_ = live["home_score"], live["away_score"]
    elif live is not None:
        hs, as_ = live["away_score"], live["home_score"]
    if live is not None and live["state"] == "in":
        badge = f'<span class="chip live">🔴 {T("live")} · {live["detail"]}</span>'
        marcador = f'<div class="result">{hs} – {as_}<small>{T("inplay")}</small></div>'
    elif (live is not None and live["state"] == "post") or r["status"] == "jugado":
        badge = f'<span class="chip ok">✅ {T("final")}</span>'
        res = f"{hs} – {as_}" if live is not None else r["result"].replace("-", " – ")
        marcador = f'<div class="result">{res}<small>{T("finalresult")}</small></div>'
    else:
        badge = f'📍 {r["city"]}'
    result = marcador
    p1, px_, p2 = r["p_home"], r["p_draw"], r["p_away"]
    chips = (f'<span class="chip hot">{r["score_1"]} · {r["p_score_1"]:.0%}</span>'
             f'<span class="chip">{r["score_2"]} · {r["p_score_2"]:.0%}</span>'
             f'<span class="chip">{r["score_3"]} · {r["p_score_3"]:.0%}</span>'
             f'<span class="chip">{T("over25_chip")} · {r["p_over_2.5"]:.0%}</span>'
             f'<span class="chip">{T("btts_chip")} · {r["p_btts"]:.0%}</span>')
    return html(f"""<div class="mcard" data-f="{sig}">
      <div class="mlabel"><span>{T("group")} {r['group']}{reloj}</span><span>{badge}</span></div>
      <div class="mteams"><span class="mteam">{team(r['home'])}</span><span class="mvs">{T("vs")}</span>
        <span class="mteam">{team(r['away'])}</span></div>
      {result}
      <div class="mxg">{T("xg")} <b>{r['xg_home']:.2f} – {r['xg_away']:.2f}</b></div>
      <div class="mbar"><div style="background:{C['home']};width:{p1:.0%}">{p1:.0%}</div>
        <div style="background:{C['draw']};width:{px_:.0%}">{px_:.0%}</div>
        <div style="background:{C['away']};width:{p2:.0%}">{p2:.0%}</div></div>
      <div class="mlegend"><b>{team_disp(r['home'])}</b><span>{T("draw")}</span><b>{team_disp(r['away'])}</b></div>
      <div class="chips">{chips}</div>
      <details class="din"><summary>{T("seedetail")}</summary>
        <div class="dbody">{match_detail(r)}</div></details>
    </div>""")


def _cmp_row(label: str, lval, rval, lshare: float, fmt="{:.0f}") -> str:
    """Fila comparativa: valor izq · barra dividida · valor der."""
    lpct = max(2, min(98, lshare * 100))
    return (f'<div class="cmp"><span class="vl">{fmt.format(lval)}</span>'
            f'<span class="mid"><small>{label}</small>'
            f'<span class="cmpbar"><i style="width:{lpct:.0f}%;background:{C["home"]}"></i>'
            f'<i style="width:{100 - lpct:.0f}%;background:{C["away"]}"></i></span></span>'
            f'<span class="vr">{fmt.format(rval)}</span></div>')


def match_detail(r) -> str:
    """HTML del desplegable: comparativa de equipos, mercados y marcadores."""
    # Elo: cuota relativa por la fórmula logística
    eh, ea = r["elo_home"], r["elo_away"]
    elo_share = 1 / (1 + 10 ** ((ea - eh) / 400))
    # ataque (mayor = mejor) y solidez defensiva (menor índice = mejor → invierto)
    ah, aa = r["attack_home"], r["attack_away"]
    atk_share = ah / (ah + aa) if (ah + aa) else 0.5
    dh, da = r["defense_home"], r["defense_away"]
    def_share = (1 / dh) / ((1 / dh) + (1 / da)) if dh and da else 0.5

    cmp = (f'<div class="dsec">{T("comparison")}</div>'
           + _cmp_row("Elo", eh, ea, elo_share)
           + _cmp_row(T("attack"), ah, aa, atk_share, fmt="{:.2f}")
           + _cmp_row(T("defense"), dh, da, def_share, fmt="{:.2f}")
           + _cmp_row(T("xg_short"), r["xg_home"], r["xg_away"],
                      r["xg_home"] / (r["xg_home"] + r["xg_away"]), fmt="{:.2f}"))

    mkt = (f'<div class="dsec">{T("markets")}</div>'
           f'<div class="mkt"><span>{T("win")} {team_disp(r["home"])}</span><b>{r["p_home"]:.0%}</b></div>'
           f'<div class="mkt"><span>{T("draw_cap")}</span><b>{r["p_draw"]:.0%}</b></div>'
           f'<div class="mkt"><span>{T("win")} {team_disp(r["away"])}</span><b>{r["p_away"]:.0%}</b></div>'
           f'<div class="mkt"><span>{T("over25")}</span><b>{r["p_over_2.5"]:.0%}</b></div>'
           f'<div class="mkt"><span>{T("under25")}</span><b>{r["p_under_2.5"]:.0%}</b></div>'
           f'<div class="mkt"><span>{T("btts_cap")}</span><b>{r["p_btts"]:.0%}</b></div>')

    pmax = max(r["p_score_1"], r["p_score_2"], r["p_score_3"]) or 1
    rows = "".join(
        f'<div class="slrow"><span class="sn">{r[f"score_{k}"]}</span>'
        f'<span class="sb"><i style="width:{r[f"p_score_{k}"] / pmax:.0%}"></i></span>'
        f'<span class="sp">{r[f"p_score_{k}"]:.0%}</span></div>' for k in (1, 2, 3))
    scores = f'<div class="dsec">{T("likely")}</div><div class="slist">{rows}</div>'
    return html(f'<div>{cmp}{mkt}{scores}</div>')


def fecha_linda(iso: str) -> str:
    d = datetime.fromisoformat(iso)
    day, month = T("days")[d.weekday()].capitalize(), T("months")[d.month]
    return f"{day} {d.day} de {month}" if LANG == "es" else f"{day} {month} {d.day}"


def plotly_theme(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color=C["text"], font_family="system-ui",
                      colorway=[C["home"], C["gold"], C["away"], "#e6e6e6", "#e59866",
                                "#af7ac5", "#76d7c4", "#f1948a", "#85c1e9", "#f7dc6f"])
    fig.update_xaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    fig.update_yaxes(gridcolor=C["border"], zerolinecolor=C["border"])
    return fig


matches, updated = load("match_forecasts")
if matches is None:
    st.error(T("no_data"))
    st.stop()

st.markdown('<div class="strip"></div>', unsafe_allow_html=True)
head_l, head_c, head_r = st.columns([3, 1, 1])
with head_l:
    st.markdown(f"## ⚽ {T('title')}")
with head_c:
    st.radio(T("language"), ["es", "en"], key="lang", horizontal=True,
             format_func=lambda x: "Español" if x == "es" else "English")
with head_r:
    version = matches["model_version"].iloc[0]
    st.markdown(
        f"<div style='text-align:right;color:{C['muted']};font-size:13px;padding-top:18px'>"
        f"<span style='color:{C['home']}'>●</span> {T('updated')} {updated:%d-%b %H:%M UTC} · "
        f"{T('model')} <code>{version}</code></div>", unsafe_allow_html=True)

tab_m, tab_g, tab_ko, tab_race, tab_health, tab_info = st.tabs(
    [T("tab_matches"), T("tab_groups"), T("tab_ko"), T("tab_title"), T("tab_health"), T("tab_info")])

@st.fragment(run_every=60)
def render_partidos():
    """Se re-ejecuta solo cada 60 s: marcadores en vivo siempre frescos."""
    live_board = load_live()
    en_vivo = sum(1 for v in live_board.values() if v["state"] == "in")
    c1, c2, c3 = st.columns(3)
    sel_group = c1.selectbox(T("f_group"), ["all"] + sorted(matches["group"].unique()),
                             format_func=lambda g: T("all_m") if g == "all" else f'{T("group")} {g}')
    status_opts = ["all", "pending", "live", "played"]
    status_lbl = {"all": T("all_m"), "pending": T("st_pending"), "live": T("st_live"), "played": T("st_played")}
    sel_status = c2.selectbox(T("f_status"), status_opts, format_func=lambda s: status_lbl[s])
    hoy = str(datetime.now().astimezone().date())
    dates = ["all"] + sorted(matches["date"].unique())
    sel_date = c3.selectbox(T("f_date"), dates, index=dates.index(hoy) if hoy in dates else 0,
                            format_func=lambda d: T("all_f") if d == "all" else fecha_linda(d))
    if en_vivo:
        st.markdown(html(f'<span class="chip live">{T("livebanner").format(n=en_vivo)}</span>'),
                    unsafe_allow_html=True)
    fsig = f"{sel_group}|{sel_status}|{sel_date}"  # cambia con cualquier filtro

    view = matches.copy()
    view["_live"] = [live_board.get("|".join(sorted([h, a])))
                     for h, a in zip(view["home"], view["away"])]
    if sel_group != "all":
        view = view[view["group"] == sel_group]
    if sel_status == "live":
        view = view[[v is not None and v["state"] == "in" for v in view["_live"]]]
    elif sel_status == "played":
        view = view[[(v is not None and v["state"] == "post") or s == "jugado"
                     for v, s in zip(view["_live"], view["status"])]]
    elif sel_status == "pending":
        view = view[[(v is None or v["state"] == "pre") and s == "pendiente"
                     for v, s in zip(view["_live"], view["status"])]]
    if sel_date != "all":
        view = view[view["date"] == sel_date]
    if "kickoff_utc" in view.columns:
        view = view.sort_values(["date", "kickoff_utc", "group"], na_position="last")
    else:
        view = view.sort_values(["date", "group"])
    if view.empty:
        st.info(T("nofilter"))
    for day in view["date"].unique():
        day_matches = view[view["date"] == day]
        n = len(day_matches)
        st.markdown(html(
            f'<div class="datehdr">📅 {fecha_linda(day)} '
            f'<small>{T("matches_n").format(n=n, s=T("plural") if n > 1 else "")}</small></div>'
        ), unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (_, r) in enumerate(day_matches.iterrows()):
            cols[i % 2].markdown(match_card(r, r["_live"], fsig), unsafe_allow_html=True)


with tab_m:
    render_partidos()

with tab_g:
    groups_df, _ = load("group_tables")
    if groups_df is not None and len(groups_df):
        groups_df = groups_df.rename(columns={groups_df.columns[0]: "team"})
        sel = st.selectbox("Grupo ", sorted(groups_df["group"].unique()),
                           format_func=lambda g: f'{T("group")} {g}')
        g = groups_df[groups_df["group"] == sel].copy()
        g["team"] = g["team"].map(team)
        show = g[["team", "exp_pts", "exp_gf", "exp_gd",
                  "p_1st", "p_2nd", "p_3rd_qualifies", "p_advance"]]
        show.columns = [T("th_team"), T("th_exp_pts"), T("th_gf"), T("th_gd"),
                        T("th_1st"), T("th_2nd"), T("th_3rd"), T("th_advance")]
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={c: st.column_config.ProgressColumn(
                         c, format="percent", min_value=0, max_value=1)
                         for c in (T("th_1st"), T("th_2nd"), T("th_3rd"), T("th_advance"))})
        fig = px.bar(g, x="team", y=["p_1st", "p_2nd", "p_3rd_qualifies"],
                     title=T("g_paths").format(g=sel), barmode="stack",
                     labels={"value": T("probability"), "team": "", "variable": ""})
        st.plotly_chart(plotly_theme(fig), width="stretch")

with tab_ko:
    ko, _ = load("knockout_forecasts")
    if ko is not None and len(ko):
        rnd = st.radio(T("round"), ["R32", "Octavos", "Cuartos", "Semifinal", "Final"],
                       horizontal=True, format_func=lambda r: T("rounds")[r])
        sub = ko[(ko["round"] == rnd) & (ko["rank"] == 1)].sort_values("match")
        cols = st.columns(2)
        for i, (_, r) in enumerate(sub.iterrows()):
            fav = r["home"] if r["p_home_advances"] >= 0.5 else r["away"]
            p_fav = max(r["p_home_advances"], 1 - r["p_home_advances"])
            gold = ' style="color:#eab308"' if rnd == "Final" else ""
            card = html(f"""<div class="mcard">
              <div class="mlabel"><span{gold}>{'★ ' if rnd == 'Final' else ''}{T("rounds")[rnd]} · {T("ko_match")} {r['match']} · {fecha_linda(r['date'])}</span>
                <span>{T("ko_pairing")} {r['p_pairing']:.0%}</span></div>
              <div class="kpair"><span>{team(r['home'])}</span><span></span></div>
              <div class="kpair"><span>{team(r['away'])}</span><span></span></div>
              <div style="margin:8px 0 2px">{T("ko_advances")} <b>{team(fav)}</b>
                <span style="color:{C['home']};font-size:19px;font-weight:800"> {p_fav:.0%}</span></div>
              <div class="chips"><span class="chip">xG {r['xg_home']:.2f} – {r['xg_away']:.2f}</span>
                <span class="chip hot">90': {r['score_mode']} ({r['p_score_mode']:.0%})</span></div>
            </div>""")
            cols[i % 2].markdown(card, unsafe_allow_html=True)
        with st.expander(T("ko_alts")):
            st.dataframe(ko[(ko["round"] == rnd) & (ko["rank"] > 1)],
                         width="stretch", hide_index=True)

with tab_race:
    hist, _ = load("champion_history")
    if hist is not None and len(hist):
        latest_date = hist["date"].max()
        latest = hist[hist["date"] == latest_date].nlargest(15, "champion").copy()
        top_teams = latest.nlargest(10, "champion")["team"]
        c1, c2 = st.columns([3, 2])
        with c1:
            hist_top = hist[hist["team"].isin(top_teams)].copy()
            hist_top["team"] = hist_top["team"].map(team_disp)
            fig = px.line(hist_top, x="date", y="champion",
                          color="team", markers=True,
                          title=T("race_byday"),
                          labels={"champion": T("pchamp"), "date": "", "team": ""})
            st.plotly_chart(plotly_theme(fig), width="stretch")
        with c2:
            latest["team"] = latest["team"].map(team)
            fig2 = px.bar(latest, x="champion", y="team", orientation="h",
                          title=T("race_now"),
                          labels={"champion": "", "team": ""})
            fig2.update_traces(marker_color=C["gold"])
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(plotly_theme(fig2), width="stretch")

with tab_health:
    calib, _ = load("calibration")
    if calib is None or calib.empty:
        st.info(T("health_wait"))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(T("evaluated"), len(calib))
        ll = calib["log_loss_acum"].iloc[-1]
        c2.metric(T("logloss_cum"), f"{ll:.4f}", delta=f"{1.0986 - ll:+.4f} {T('vs_uniform')}",
                  delta_color="normal")
        c3.metric(T("hits"), f"{calib['hit'].mean():.0%}")
        fig = px.line(calib.reset_index(), x="index",
                      y=["log_loss_acum", "log_loss_uniforme"],
                      title=T("cal_title"),
                      labels={"index": T("cal_xlabel"), "value": T("cal_ylabel"), "variable": ""})
        st.plotly_chart(plotly_theme(fig), width="stretch")
        show = calib[["date", "home", "away", "result", "outcome",
                      "p_home", "p_draw", "p_away", "log_loss", "hit"]].copy()
        show["home"] = show["home"].map(team)
        show["away"] = show["away"].map(team)
        st.dataframe(show, width="stretch", hide_index=True)

        mc, _ = load("model_comparison")
        if mc is not None and len(mc):
            st.markdown(f"#### 🏅 {T('cmp_title')}")
            real = mc[mc["model"] != "uniform"]
            best_id = real.iloc[0]["model"] if len(real) else None

            def _mlabel(row):
                name = T("mlabels").get(row["model"], row["model"])
                if row["model"] == "ensemble" and pd.notna(row.get("w")):
                    name += f" (w={row['w']:.2f})"
                return name + (" ⭐" if row["model"] == best_id else "")

            disp = mc.copy()
            disp[T("th_model")] = disp.apply(_mlabel, axis=1)
            cmp_show = disp[[T("th_model"), "n", "log_loss", "brier", "rps", "accuracy"]].copy()
            cmp_show.columns = [T("th_model"), T("th_n"), T("th_logloss2"), T("th_brier"),
                                T("th_rps"), T("th_acc")]
            st.dataframe(cmp_show, width="stretch", hide_index=True,
                         column_config={T("th_acc"): st.column_config.ProgressColumn(
                             T("th_acc"), format="percent", min_value=0, max_value=1)})
            caption = T("cmp_caption").format(n=int(mc.iloc[0]["n"]))
            if (mc["model"] == "market").any():
                caption += " " + T("cmp_market_note")
            st.caption(caption)

with tab_info:
    mi = model_info()
    if mi.get("intro"):
        st.info(localize(mi["intro"]))
    st.subheader(f"🧮 {T('info_models')}")
    for m in mi.get("models", []):
        st.markdown(f"**{localize(m['name'])}** — {localize(m.get('desc'))}")
        if m.get("formula"):
            st.markdown(f"*{T('info_formula')}:*")
            st.code(m["formula"], language=None)
        params = localize(m.get("params")) or []
        if params:
            st.markdown(f"*{T('info_params')}:*\n" + "\n".join(f"- {p}" for p in params))
    st.subheader(f"🛰️ {T('info_sources')}")
    for s in mi.get("sources", []):
        opt = f" · {T('info_optional')}" if s.get("optional") else ""
        role = localize(s.get("role"))
        role_md = f"*{role}*  \n" if role else ""
        st.markdown(f"**[{s['name']}]({s['url']})**{opt}  \n{role_md}{localize(s.get('use'))}")
    st.subheader(f"🔄 {T('info_pipeline')}")
    st.markdown(localize(mi.get("pipeline")))

st.markdown(
    f"<div style='color:{C['muted']};font-size:11px;text-align:center;margin-top:20px'>"
    f"{T('footer')}</div>",
    unsafe_allow_html=True)
