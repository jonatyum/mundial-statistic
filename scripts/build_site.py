"""Genera el sitio estático para GitHub Pages a partir de los CSV del pipeline.

Lee data/processed/*.csv, embebe los datos como JSON en un index.html
autocontenido (mismo diseño que el dashboard Streamlit) y lo escribe en docs/.
El JS del navegador hace filtros, desplegables, gráficos y marcadores en vivo (ESPN).

Uso: .venv/bin/python scripts/build_site.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
CONFIG = ROOT / "config"

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
# nombres ESPN -> canónico (para casar los marcadores en vivo en el navegador)
ALIASES = {
    "Czechia": "Czech Republic", "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Congo DR": "DR Congo", "USA": "United States", "Korea Republic": "South Korea",
    "Cabo Verde": "Cape Verde", "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
    "Curacao": "Curaçao",
}


def load(name: str) -> pd.DataFrame:
    path = PROC / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def records(df: pd.DataFrame, rename_first=None) -> list[dict]:
    if df.empty:
        return []
    if rename_first:
        df = df.rename(columns={df.columns[0]: rename_first})
    return df.where(pd.notna(df), None).to_dict(orient="records")


def main():
    matches = load("match_forecasts")
    version = matches["model_version"].iloc[0] if len(matches) else "—"
    now = datetime.now(timezone.utc)
    updated = now.strftime("%d-%b %H:%M UTC")

    teams = yaml.safe_load((CONFIG / "teams.yaml").read_text())
    model_info = yaml.safe_load((CONFIG / "model_info.yaml").read_text())

    data = {
        "generated": updated,
        "generated_iso": now.isoformat(),
        "version": str(version),
        "matches": records(matches),
        "groups": records(load("group_tables"), "team"),
        "knockout": records(load("knockout_forecasts")),
        "champion_history": records(load("champion_history")),
        "champion_now": records(load("tournament_probabilities"), "team"),
        "calibration": records(load("calibration")),
        "model_comparison": records(load("model_comparison")),
        "flags": FLAG,
        "aliases": ALIASES,
        "names_es": teams.get("names_es", {}),
        "model_info": model_info,
    }

    template = (Path(__file__).parent / "site_template.html").read_text()
    html = template.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    DOCS.mkdir(exist_ok=True)
    (DOCS / "index.html").write_text(html)
    (DOCS / ".nojekyll").write_text("")  # evita que GitHub Pages corra Jekyll
    print(f"✓ docs/index.html generado ({len(html):,} bytes) · {len(data['matches'])} partidos")


if __name__ == "__main__":
    main()
