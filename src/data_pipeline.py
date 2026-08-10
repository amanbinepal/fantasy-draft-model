"""Pull and cache weekly NFL player stats from nflverse.

One row = one player's recorded stat line for one regular-season game.
"Games played," for this project, means "the player has a row here":
nflverse only creates a row when a player recorded some statistical
output that week, so byes, healthy scratches, and injury-outs don't
produce rows on their own. See DECISIONS.md for the reasoning and the
known edge case (a row can still exist with every stat at zero).

Pulled directly via pandas.read_parquet from nflverse-data's GitHub
releases, not through nfl_data_py: that package is stuck on nflverse's
now-deprecated `player_stats` release (no 2025 data there), while the
current `stats_player` release has every season we need under a
`stats_player_week_{year}.parquet` naming pattern.
"""

from pathlib import Path

import pandas as pd
import yaml

NFLVERSE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{year}.parquet"
)

# Our league only rosters these positions (config.yaml). The raw pull
# also includes defense, kicking, and special teams players, which we
# drop here rather than carry through the rest of the pipeline.
OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]

STAT_COLUMNS = [
    "completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "special_teams_tds",
]


def load_config(path="config.yaml"):
    """Read league and data settings from config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def pull_weekly_stats(seasons, cache_dir="data/raw"):
    """Pull weekly regular-season player stats, cached to a local parquet file.

    Returns one row per offense player per game they recorded a stat line
    for. Postseason rows are dropped: a standard redraft fantasy league
    scores the regular season only.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"weekly_stats_{min(seasons)}_{max(seasons)}.parquet"

    if cache_path.exists():
        print(f"Loading cached pull from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Pulling weekly stats for {seasons} from nflverse...")
    frames = []
    for year in seasons:
        url = NFLVERSE_URL.format(year=year)
        try:
            frames.append(pd.read_parquet(url))
        except Exception as exc:
            raise RuntimeError(
                f"Pull failed for {year} ({url}). PROJECT_PLAN.md flags a "
                "known issue where sandboxed environments can block "
                "nflverse's release-file CDN, a known infra quirk rather "
                "than a bug in this code. Also worth checking: whether "
                "nflverse has moved this data again, the same way "
                "`player_stats` got deprecated in favor of `stats_player`."
            ) from exc
    df = pd.concat(frames, ignore_index=True)

    df = df[df["season_type"] == "REG"]
    df = df[df["position"].isin(OFFENSE_POSITIONS)]
    df = df.reset_index(drop=True)

    df.to_parquet(cache_path)
    print(f"Cached {len(df)} rows to {cache_path}")
    return df


def games_played_summary(df, season):
    """Row count per player for one season: the games-played definition
    in practice. Grouped by player_id (not just name) since names can
    collide."""
    season_df = df[df["season"] == season]
    return season_df.groupby(["player_id", "player_display_name"]).size()


def zero_stat_row_check(df):
    """Rows where every stat column is zero: a player who has a row (so
    counts as "played" under stat-line presence) but recorded nothing
    meaningful. Rare, but worth checking on every pull rather than
    assuming it stays rare."""
    present_cols = [c for c in STAT_COLUMNS if c in df.columns]
    zero_mask = (df[present_cols].fillna(0) == 0).all(axis=1)
    return df.loc[zero_mask]


if __name__ == "__main__":
    config = load_config()
    seasons = config["data"]["seasons"]

    df = pull_weekly_stats(seasons)

    print()
    print("=== Shape and columns ===")
    print(f"{len(df)} rows, {len(df.columns)} columns")
    print(f"Seasons covered: {sorted(df['season'].unique().tolist())}")
    print(f"Columns: {sorted(df.columns.tolist())}")

    print()
    print("=== What one row represents ===")
    sample = df.iloc[0]
    print(
        f"One row = one player's recorded stat line for one regular-season "
        f"game. Example: {sample['player_display_name']}, "
        f"{sample['team']}, week {int(sample['week'])} of "
        f"{int(sample['season'])}."
    )

    print()
    print("=== Games-played distribution (most recent season) ===")
    latest_season = max(seasons)
    counts = games_played_summary(df, latest_season)
    print(
        f"Season {latest_season}: min {counts.min()}, max {counts.max()}, "
        f"median {counts.median()}"
    )
    print("Fewest games played (top 5, natural injury/committee check):")
    print(counts.nsmallest(5))

    print()
    print("=== Spot-check: known injury case (2024) ===")
    if 2024 in df["season"].unique():
        cmc = df[
            (df["player_display_name"] == "Christian McCaffrey")
            & (df["season"] == 2024)
        ]
        weeks = sorted(cmc["week"].tolist())
        print(
            f"Christian McCaffrey (2024): played weeks {weeks} "
            f"({len(weeks)} games) out of a 17-week season"
        )
    else:
        print("2024 not in pulled seasons, skipping this check")

    print()
    print("=== All-zero-stat-row check ===")
    zero_rows = zero_stat_row_check(df)
    pct = 100 * len(zero_rows) / len(df)
    print(
        f"{len(zero_rows)} of {len(df)} rows ({pct:.2f}%) have zero across "
        f"all stat columns"
    )
    if len(zero_rows):
        print(zero_rows["position"].value_counts())
