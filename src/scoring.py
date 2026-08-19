"""Turn nflverse weekly stat lines into fantasy points under this
league's exact scoring rules (pulled live from Sleeper, see
sleeper_api.py), not a generic PPR assumption.

Vectorized over the whole dataframe rather than a per-row .apply, same
style as data_pipeline.py's zero_stat_row_check.
"""

import pandas as pd

from src.data_pipeline import load_config
from src.sleeper_api import pull_scoring_settings

# Sleeper scoring key -> nflverse stat column. Only keys with an offense
# counterpart in our data are listed here; defense/kicking-only keys in
# Sleeper's scoring_settings (pts_allow_*, fgm_*, blk_kick, def_td, ...)
# don't apply to the offense players this project tracks and are skipped,
# logged rather than silently dropped (see score_dataframe below).
STAT_MAP = {
    "pass_yd": "passing_yards",
    "pass_td": "passing_tds",
    "pass_int": "passing_interceptions",
    "pass_2pt": "passing_2pt_conversions",
    "rush_yd": "rushing_yards",
    "rush_td": "rushing_tds",
    "rush_2pt": "rushing_2pt_conversions",
    "rec": "receptions",
    "rec_yd": "receiving_yards",
    "rec_td": "receiving_tds",
    "rec_2pt": "receiving_2pt_conversions",
    "st_td": "special_teams_tds",
}

# fum_lost is one Sleeper key covering three nflverse columns (a fumble lost
# on a rush, a reception, or a sack), since Sleeper scores total fumbles
# lost regardless of how the ball came out.
FUM_LOST_COLUMNS = ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]

# 40+ yard TD bonus. Sleeper pays this per touchdown of 40+ yards, but
# nflverse only gives us a per-game count of 40+ yard plays, not which
# specific play scored. Approximated as 2 * min(tds, 40+ yard plays) per
# category per game: exact in the common one-TD case (matches the flat
# "+2" DECISIONS.md originally described), and more accurate than a flat
# flag on multi-TD games. Still imperfect: a game with one short
# (non-40+) TD and a separate, unrelated 40+ yard play that wasn't a
# score still counts as min(1, 1) = 1 here and gets over-credited, since
# the count-only columns can't tell the two plays apart. See DECISIONS.md.
BONUS_CATEGORIES = {
    "pass_td_40p": ("passing_tds", "passing_40"),
    "rush_td_40p": ("rushing_tds", "rushing_40"),
    "rec_td_40p": ("receiving_tds", "receiving_40"),
}


def load_scoring_settings(cache_dir="data/raw"):
    """Load this league's scoring settings (cached after the first pull,
    see sleeper_api.py)."""
    config = load_config()
    league_id = config["league"]["league_id"]
    return pull_scoring_settings(league_id, cache_dir=cache_dir)


def score_dataframe(df, scoring_settings):
    """Add a fantasy_points column, computed under this league's exact
    scoring rules. Returns a copy; doesn't mutate df."""
    points = pd.Series(0.0, index=df.index)
    unmapped = []

    for sleeper_key, weight in scoring_settings.items():
        if sleeper_key in BONUS_CATEGORIES:
            continue  # handled separately below, not a flat per-stat weight
        if sleeper_key == "fum_lost":
            for col in FUM_LOST_COLUMNS:
                if col in df.columns:
                    points += df[col].fillna(0) * weight
            continue
        col = STAT_MAP.get(sleeper_key)
        if col is None:
            unmapped.append(sleeper_key)
            continue
        if col in df.columns:
            points += df[col].fillna(0) * weight

    if unmapped:
        print(
            f"scoring.py: {len(unmapped)} Sleeper scoring keys have no "
            f"offense-stat mapping, skipped: {sorted(unmapped)}"
        )

    for bonus_key, (td_col, yard40_col) in BONUS_CATEGORIES.items():
        if bonus_key not in scoring_settings:
            continue
        weight = scoring_settings[bonus_key]
        tds = df[td_col].fillna(0)
        plays_40p = df[yard40_col].fillna(0)
        bonus_count = pd.concat([tds, plays_40p], axis=1).min(axis=1)
        points += bonus_count * weight

    result = df.copy()
    result["fantasy_points"] = points
    return result


if __name__ == "__main__":
    from src.data_pipeline import pull_weekly_stats

    config = load_config()
    seasons = config["data"]["seasons"]

    df = pull_weekly_stats(seasons)
    settings = load_scoring_settings()
    scored = score_dataframe(df, settings)

    print()
    print("=== Fantasy points distribution ===")
    print(scored["fantasy_points"].describe())

    print()
    print("=== Spot check: our score vs. nflverse's own PPR calc ===")
    print(
        "nflverse's fantasy_points_ppr uses generic PPR scoring (no 40+ "
        "bonus, no league-specific 2pt weight), so ours should track "
        "closely but not match exactly, especially on 40+ bonus games."
    )
    cmc = scored[
        (scored["player_display_name"] == "Christian McCaffrey")
        & (scored["season"] == 2024)
    ][["week", "fantasy_points", "fantasy_points_ppr"]]
    print("Christian McCaffrey, 2024:")
    print(cmc.sort_values("week"))

    print()
    print("=== Spot check: a game with the 40+ bonus applied ===")
    bonus_game = scored[
        (scored["rushing_tds"] > 0) & (scored["rushing_40"] > 0)
    ].head(1)
    if len(bonus_game):
        row = bonus_game.iloc[0]
        print(
            f"{row['player_display_name']}, {int(row['season'])} week "
            f"{int(row['week'])}: {row['rushing_tds']} rush TD(s), "
            f"{row['rushing_40']} 40+ yard rush play(s), our "
            f"fantasy_points={row['fantasy_points']:.2f}, "
            f"nflverse fantasy_points_ppr={row['fantasy_points_ppr']:.2f} "
            f"(diff should be roughly the 2.0 bonus, plus this league's "
            f"other scoring differences from generic PPR)"
        )
    else:
        print("No qualifying rushing 40+ bonus game found in this data")
