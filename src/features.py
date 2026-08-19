"""Build the training table: prior-season usage stats -> next-season
fantasy points, under this league's exact scoring rules.

Row granularity: one row = one (player, target_season) pair, where a
row's features come entirely from target_season - 1 (the immediately
preceding season), never from target_season or later. The leakage rule
is mechanical rather than a thing to remember: features and target are
built from two different seasons' dataframes before ever being joined.

Rookies have no prior season in our data, so they produce no row here.
Phase 3 handles them separately via FantasyPros ECR.
"""

import pandas as pd

from src.data_pipeline import load_config, pull_weekly_stats
from src.scoring import load_scoring_settings, score_dataframe

# Core counting stats aggregated to season totals, then also expressed as
# per-game rates. Kept lean for this first pass: no advanced route/target
# share metrics yet (target_share, air_yards_share, wopr, racr, pacr),
# those are pass-catcher-only and can be added later once we see backtest
# results.
COUNTING_STATS = [
    "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
]


def build_season_table(df, scoring_settings):
    """One row per player-season: games played, fantasy points, and
    season totals/per-game rates for the core counting stats. games_played
    uses the same stat-line-presence definition as data_pipeline.py, row
    count per player per season."""
    scored = score_dataframe(df, scoring_settings)

    agg_kwargs = {
        "player_display_name": ("player_display_name", "first"),
        "position": ("position", lambda s: s.mode().iat[0]),
        "games_played": ("player_id", "count"),
        "fantasy_points": ("fantasy_points", "sum"),
    }
    agg_kwargs.update({stat: (stat, "sum") for stat in COUNTING_STATS})

    season_df = scored.groupby(["player_id", "season"]).agg(**agg_kwargs).reset_index()

    season_df["fantasy_points_per_game"] = (
        season_df["fantasy_points"] / season_df["games_played"]
    )
    for stat in COUNTING_STATS:
        season_df[f"{stat}_per_game"] = season_df[stat] / season_df["games_played"]

    return season_df


def build_training_table(season_df):
    """Pair each player-season's stats (features) with that same
    player's total fantasy points the following season (target).

    Joined on player_id alone, not name or position: a label drift
    between seasons (a name-string quirk, a position reclassification)
    must not be misread as a real dropout. Left join, so a player with a
    feature-season row but no rows at all the following season gets
    target_fantasy_points = 0 (a real bust/injury/retirement outcome)
    rather than being silently dropped.

    Only pairs a feature season with a target season that actually
    exists in season_df. Without this, the most recent feature season
    (the one we don't have next-season data for yet, because that season
    hasn't been played) would get target_fantasy_points = 0 for every
    single player, indistinguishable from a real dropout when the real
    reason is "that season isn't in our data yet." Those rows become
    next season's live projection input instead (Phase 3), not a
    training row with a fabricated zero.
    """
    known_seasons = set(season_df["season"].unique())

    feature_cols = ["games_played", "fantasy_points", "fantasy_points_per_game"] + \
        COUNTING_STATS + [f"{stat}_per_game" for stat in COUNTING_STATS]

    features = season_df.rename(columns={"season": "feature_season"}).copy()
    features["target_season"] = features["feature_season"] + 1
    features = features[features["target_season"].isin(known_seasons)]

    targets = season_df[["player_id", "season", "fantasy_points"]].rename(
        columns={"season": "target_season", "fantasy_points": "target_fantasy_points"}
    )

    table = features.merge(targets, on=["player_id", "target_season"], how="left")
    table["target_fantasy_points"] = table["target_fantasy_points"].fillna(0)

    cols = ["player_id", "player_display_name", "position", "feature_season", "target_season"] \
        + feature_cols + ["target_fantasy_points"]
    return table[cols].reset_index(drop=True)


if __name__ == "__main__":
    config = load_config()
    seasons = config["data"]["seasons"]

    df = pull_weekly_stats(seasons)
    scoring_settings = load_scoring_settings()

    season_df = build_season_table(df, scoring_settings)
    table = build_training_table(season_df)

    print("=== Shape ===")
    print(f"{len(table)} rows, {len(table.columns)} columns")
    print(
        f"Feature seasons covered: {sorted(table['feature_season'].unique().tolist())} "
        f"(note: {max(seasons)} has no row here, no {max(seasons) + 1} data exists "
        f"yet to pair it with)"
    )

    print()
    print("=== Rows per position ===")
    print(table["position"].value_counts())

    print()
    print("=== Worked example: a player's feature season next to their actual next-season total ===")
    example = table[table["games_played"] >= 15].iloc[0]
    print(
        f"{example['player_display_name']} ({example['position']}): "
        f"feature season {int(example['feature_season'])} had "
        f"{int(example['games_played'])} games, {example['fantasy_points']:.1f} "
        f"fantasy points ({example['fantasy_points_per_game']:.2f}/game). "
        f"Target season {int(example['target_season'])} total: "
        f"{example['target_fantasy_points']:.1f} points."
    )

    print()
    print("=== Dropout example: no stat-line rows at all in the target season ===")
    dropout = table[table["target_fantasy_points"] == 0].sort_values(
        "feature_season", ascending=False
    )
    if len(dropout):
        row = dropout.iloc[0]
        print(
            f"{row['player_display_name']} ({row['position']}): feature season "
            f"{int(row['feature_season'])} had {int(row['games_played'])} games, "
            f"{row['fantasy_points']:.1f} points, but no rows at all in target "
            f"season {int(row['target_season'])} (which does exist in our data) "
            f"-> target_fantasy_points = 0."
        )
    else:
        print("No dropout rows found (unexpected, worth investigating)")

    print()
    print("=== Join-key sanity check: a player active in consecutive seasons ===")
    both_seasons = table[table["target_fantasy_points"] > 0].sort_values(
        "target_fantasy_points", ascending=False
    )
    if len(both_seasons):
        row = both_seasons.iloc[0]
        print(
            f"{row['player_display_name']}: feature season "
            f"{int(row['feature_season'])} -> target season "
            f"{int(row['target_season'])}, target_fantasy_points = "
            f"{row['target_fantasy_points']:.1f} (nonzero, so the player_id join "
            f"succeeded regardless of name/position formatting)"
        )
