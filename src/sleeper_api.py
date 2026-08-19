"""Thin wrapper around Sleeper's public API.

Sleeper needs no auth for read-only league data, so this is just cached
GET requests. Right now it only pulls scoring_settings (Phase 2, feature
scoring). Phase 6 will add draft-pick polling here too, so all Sleeper
calls stay in one file instead of getting scattered across the pipeline.

Scoring settings get cached to disk with no refresh path, unlike weekly
stats in data_pipeline.py (which grow every week and always need a fresh
pull for a new season). A league's scoring settings are fixed once the
league is created for the season, they don't change week to week, so a
stale cache isn't a real risk here the way it would be for stats.
"""

import json
from pathlib import Path

import requests

from src.data_pipeline import load_config

SLEEPER_LEAGUE_URL = "https://api.sleeper.app/v1/league/{league_id}"


def pull_scoring_settings(league_id, cache_dir="data/raw"):
    """Fetch this league's scoring_settings, cached to a local JSON file.

    Returns a dict of Sleeper stat key -> point value, e.g. {"pass_td":
    4.0, "rec": 1.0, ...}. Sleeper folds scoring_settings into the
    general league-info response, so we pull the whole thing and keep
    just that one field.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "league_scoring_settings.json"

    if cache_path.exists():
        print(f"Loading cached scoring settings from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    url = SLEEPER_LEAGUE_URL.format(league_id=league_id)
    print(f"Pulling league info from {url}...")
    response = requests.get(url)
    response.raise_for_status()
    scoring_settings = response.json()["scoring_settings"]

    with open(cache_path, "w") as f:
        json.dump(scoring_settings, f, indent=2, sort_keys=True)
    print(f"Cached {len(scoring_settings)} scoring keys to {cache_path}")
    return scoring_settings


if __name__ == "__main__":
    config = load_config()
    league_id = config["league"]["league_id"]

    settings = pull_scoring_settings(league_id)

    print()
    print("=== Full scoring_settings ===")
    for key in sorted(settings):
        print(f"{key}: {settings[key]}")

    print()
    print("=== Sign check on penalty stats ===")
    for key in ("pass_int", "fum_lost"):
        if key in settings:
            value = settings[key]
            sign = "negative (good, no extra negation needed)" if value < 0 else "POSITIVE (mapping bug: needs negation in scoring.py)"
            print(f"{key}: {value} -> {sign}")
        else:
            print(f"{key}: not present in scoring_settings")
