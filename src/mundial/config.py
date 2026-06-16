from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def settings() -> dict:
    return yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())


@lru_cache
def tournament() -> dict:
    return yaml.safe_load((ROOT / "config" / "tournament_2026.yaml").read_text())


@lru_cache
def _teams() -> dict:
    return yaml.safe_load((ROOT / "config" / "teams.yaml").read_text())


def team_aliases() -> dict:
    return _teams()["aliases"]


@lru_cache
def team_names_es() -> dict:
    """Canónico (inglés) -> nombre para mostrar en español."""
    return _teams().get("names_es", {})


@lru_cache
def model_info() -> dict:
    """Modelos, fórmulas y fuentes (config/model_info.yaml). Bilingüe."""
    return yaml.safe_load((ROOT / "config" / "model_info.yaml").read_text())


def canonical(name: str) -> str:
    return team_aliases().get(name, name)


def team_es(name: str) -> str:
    """Nombre en español si existe, si no el canónico."""
    return team_names_es().get(name, name)
