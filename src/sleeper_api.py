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

import pandas as pd
import requests

from src.data_pipeline import load_config

SLEEPER_LEAGUE_URL = "https://api.sleeper.app/v1/league/{league_id}"
DRAFT_PICKS_URL = "https://api.sleeper.app/v1/draft/{draft_id}/picks"

PICK_COLUMNS = [
    "pick_no", "round", "draft_slot", "roster_id",
    "sleeper_player_id", "first_name", "last_name", "position", "team",
]


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


def _picks_to_dataframe(picks):
    """Pure parsing, separate from the HTTP call, so it can be verified
    against a realistic pick shape without needing a live, in-progress
    draft. Each pick's metadata already carries first_name/last_name/
    position/team inline (confirmed against Sleeper's own API docs), no
    separate player-reference pull needed for basic matching."""
    if not picks:
        return pd.DataFrame(columns=PICK_COLUMNS)

    rows = []
    for pick in picks:
        metadata = pick.get("metadata") or {}
        rows.append({
            "pick_no": pick["pick_no"],
            "round": pick["round"],
            "draft_slot": pick["draft_slot"],
            "roster_id": pick["roster_id"],
            "sleeper_player_id": pick["player_id"],
            "first_name": metadata.get("first_name"),
            "last_name": metadata.get("last_name"),
            "position": metadata.get("position"),
            "team": metadata.get("team"),
        })
    return pd.DataFrame(rows, columns=PICK_COLUMNS).sort_values("pick_no").reset_index(drop=True)


def pull_draft_picks(draft_id):
    """Pull every pick made so far in a draft. No caching, unlike
    pull_scoring_settings: picks change throughout the draft, every poll
    needs a genuinely fresh pull. A stale cache here would be actively
    dangerous (recommending an already-drafted player), not just
    imprecise the way a slightly-stale cache would be elsewhere.

    Pre-draft (no picks yet) returns an empty DataFrame with the right
    columns, not an error, so live_tracker.py can poll starting well
    before the draft actually begins.
    """
    url = DRAFT_PICKS_URL.format(draft_id=draft_id)
    response = requests.get(url)
    response.raise_for_status()
    return _picks_to_dataframe(response.json())


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

    print()
    print("=== Draft picks pull (real configured draft) ===")
    draft_id = config["league"]["draft_id"]
    picks = pull_draft_picks(draft_id)
    print(f"{len(picks)} picks pulled for draft_id {draft_id}")
    if len(picks) == 0:
        print("Expected right now: this draft is pre-draft (verified via "
              "the Sleeper API directly), not a bug. Real end-to-end "
              "verification against non-empty picks happens in Phase 7's "
              "mock draft.")
    print(f"Columns: {picks.columns.tolist()}")

    print()
    print("=== Parsing check: synthetic pick, Sleeper's own documented shape ===")
    synthetic_pick = {
        "player_id": "2391",
        "picked_by": "234343434",
        "roster_id": "1",
        "round": 5,
        "draft_slot": 5,
        "pick_no": 1,
        "is_keeper": None,
        "draft_id": "257270643320426496",
        "metadata": {
            "team": "ARI", "status": "Injured Reserve", "sport": "nfl",
            "position": "RB", "player_id": "2391", "number": "31",
            "news_updated": "1513007102037", "last_name": "Johnson",
            "injury_status": "Out", "first_name": "David",
        },
    }
    parsed = _picks_to_dataframe([synthetic_pick])
    row = parsed.iloc[0]
    print(
        f"Parsed: first_name={row['first_name']!r}, last_name={row['last_name']!r}, "
        f"position={row['position']!r}, team={row['team']!r} "
        f"(expected: David / Johnson / RB / ARI)"
    )
