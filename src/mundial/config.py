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
def team_aliases() -> dict:
    return yaml.safe_load((ROOT / "config" / "teams.yaml").read_text())["aliases"]


def canonical(name: str) -> str:
    return team_aliases().get(name, name)
