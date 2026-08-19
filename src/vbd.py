"""Convert blended projections into an actual draft board via
Value-Based Drafting (VBD): points above replacement, replacement level
set by this league's real roster shape, not a generic assumption.

Replacement level means "worst starter" (Value Over Last Starter), the
standard VBD baseline, not "worst rostered player": bench (6) and IR (2)
slots aren't counted, there's no clean way to derive how deep each
position gets bench-stashed from the roster shape the way starter counts
and flex slots can be derived.

QB has no flex eligibility in this league (FLEX is separate from the 1
QB slot, no superflex), so QB's replacement level is just its 10th-best
player. RB/WR/TE share the 20 flex slots, so their replacement levels
come from one combined pool: whichever position has deeper top-end value
naturally claims more of those 20 slots, not a fixed split. This is the
"double FLEX matters" PROJECT_PLAN.md flags directly.

Tiers use median gap + k*MAD (median absolute deviation), not mean+std.
Mean and std are self-referential in a way that breaks on the smaller
positions (QB, TE): the single largest gap gets included when computing
the mean/std used to judge whether that same gap clears the threshold,
so one real outlier tier break can inflate the std enough that nothing
clears the bar, silently collapsing a thin position to 1-2 tiers. Median
and MAD barely move for a single extreme value, the standard substitute
for exactly this failure mode.
"""

import numpy as np
import pandas as pd

from src.data_pipeline import load_config, pull_weekly_stats
from src.ecr_blend import (
    current_player_pool,
    implied_points_by_rank,
    load_ecr,
    match_ecr_to_players,
    blend_projections,
    _predict_current_season,
)
from src.features import build_season_table, build_training_table
from src.model import train_final_models
from src.scoring import load_scoring_settings

FLEX_ELIGIBLE = ["RB", "WR", "TE"]
FIXED_POSITIONS = ["QB", "RB", "WR", "TE"]

TIER_GAP_MAD_MULTIPLIER = 3.0


def starter_counts(config):
    """Fixed per-position starter counts (roster slot count * teams),
    and total flex slots."""
    teams = config["league"]["teams"]
    roster = config["league"]["roster"]
    fixed = {pos: roster[pos] * teams for pos in FIXED_POSITIONS}
    flex_slots = roster["FLEX"] * teams
    return fixed, flex_slots


def compute_replacement_levels(blended, config):
    """Flex-aware replacement level per position. Each position's fixed
    starter count is a guaranteed floor, not a soft target: every team
    fields exactly 1 TE starter regardless of whether that TE would
    "deserve" a value-based spot against a deep WR class, so only the
    players *beyond* each position's fixed count compete openly for the
    20 flex slots. (An earlier version pooled all RB/WR/TE together
    unconstrained before taking the top 70, which let deep positions
    crowd out a shallow position's guaranteed floor, verified wrong:
    it let TE claim only 8 of the top 70 despite needing 10 fixed
    starters. Fixed here.)

    Returns (replacement dict, the combined RB/WR/TE startable pool) so
    callers can inspect the actual position breakdown of who claimed the
    flex slots.
    """
    fixed, flex_slots = starter_counts(config)
    replacement = {}

    qb = blended[blended["position"] == "QB"].sort_values("blended_points", ascending=False)
    replacement["QB"] = qb.iloc[fixed["QB"] - 1]["blended_points"]

    fixed_starters, remaining = {}, {}
    for position in FLEX_ELIGIBLE:
        pos_sorted = blended[blended["position"] == position].sort_values(
            "blended_points", ascending=False
        ).reset_index(drop=True)
        fixed_starters[position] = pos_sorted.iloc[:fixed[position]]
        remaining[position] = pos_sorted.iloc[fixed[position]:]

    remaining_combined = pd.concat(remaining.values()).sort_values(
        "blended_points", ascending=False
    )
    flex_winners = remaining_combined.head(flex_slots)

    for position in FLEX_ELIGIBLE:
        position_starters = pd.concat([
            fixed_starters[position],
            flex_winners[flex_winners["position"] == position],
        ])
        replacement[position] = position_starters["blended_points"].min()

    startable = pd.concat(list(fixed_starters.values()) + [flex_winners])
    return replacement, startable


def compute_vbd(blended, config):
    """Adds replacement_level, vbd_value, per-position position_rank, and
    overall raw_rank/vbd_rank/rank_diff (positive = moved up under VBD
    relative to raw points) to the blended table."""
    replacement, startable = compute_replacement_levels(blended, config)

    result = blended.copy()
    result["replacement_level"] = result["position"].map(replacement)
    result["vbd_value"] = result["blended_points"] - result["replacement_level"]
    result["position_rank"] = (
        result.groupby("position")["vbd_value"].rank(ascending=False, method="first").astype(int)
    )
    result["raw_rank"] = result["blended_points"].rank(ascending=False, method="first").astype(int)
    result["vbd_rank"] = result["vbd_value"].rank(ascending=False, method="first").astype(int)
    result["rank_diff"] = result["raw_rank"] - result["vbd_rank"]

    return result, replacement, startable


def assign_tiers(vbd_df, mad_multiplier=TIER_GAP_MAD_MULTIPLIER):
    """Per position, median-gap + MAD tier detection. Returns the
    dataframe with a tier column added, plus a per-position debug dict
    (gaps, median_gap, mad, threshold) so small positions can get an
    explicit second look, not just a resulting tier count."""
    result = vbd_df.copy()
    tier_col = pd.Series(index=result.index, dtype=int)
    gap_debug = {}

    for position, group in result.groupby("position"):
        sorted_group = group.sort_values("vbd_value", ascending=False)
        gaps = (-sorted_group["vbd_value"].diff()).to_numpy()  # gaps[0] is NaN, no prior player
        real_gaps = gaps[1:]
        median_gap = np.median(real_gaps) if len(real_gaps) else 0.0
        mad = np.median(np.abs(real_gaps - median_gap)) if len(real_gaps) else 0.0
        threshold = median_gap + mad_multiplier * mad

        tier = 1
        tier_labels = [tier]
        for gap in real_gaps:
            if gap > threshold:
                tier += 1
            tier_labels.append(tier)

        tier_col.loc[sorted_group.index] = tier_labels
        gap_debug[position] = {
            "gaps": real_gaps, "median_gap": median_gap, "mad": mad, "threshold": threshold,
        }

    result["tier"] = tier_col
    return result, gap_debug


def _build_blended_table():
    """Reruns the full chain (cached at every stage) to get the current
    blended projection table. Same pattern every prior file's main block
    already uses."""
    config = load_config()
    seasons = config["data"]["seasons"]
    current_season = max(seasons)

    df = pull_weekly_stats(seasons)
    scoring_settings = load_scoring_settings()
    season_df = build_season_table(df, scoring_settings)
    table = build_training_table(season_df)

    models = train_final_models(table)
    ridge_projections = _predict_current_season(season_df, models, current_season)

    player_pool = current_player_pool(df, current_season)
    ecr_df = load_ecr()
    matched = match_ecr_to_players(ecr_df, player_pool)
    implied = implied_points_by_rank(matched, ridge_projections)
    blended = blend_projections(ridge_projections, implied)

    return blended, config


if __name__ == "__main__":
    blended, config = _build_blended_table()

    vbd_df, replacement, startable = compute_vbd(blended, config)
    vbd_df, gap_debug = assign_tiers(vbd_df)

    print("=== Replacement level per position ===")
    for position, value in replacement.items():
        print(f"{position}: {value:.1f}")

    print()
    print("=== Flex-slot allocation: who actually claimed the 20 flex slots ===")
    fixed, flex_slots = starter_counts(config)
    print(f"Combined startable pool (RB+WR+TE, {len(startable)} players): "
          f"{startable['position'].value_counts().to_dict()}")
    print(f"(fixed starters alone would be RB={fixed['RB']}, WR={fixed['WR']}, TE={fixed['TE']}, "
          f"+{flex_slots} flex slots split by actual value, not evenly)")

    print()
    print("=== Top 15: raw blended_points vs. VBD, side by side ===")
    raw_top15 = blended.sort_values("blended_points", ascending=False).head(15)[
        ["player_display_name", "position", "blended_points"]
    ].reset_index(drop=True).add_prefix("raw_")
    vbd_top15 = vbd_df.sort_values("vbd_value", ascending=False).head(15)[
        ["player_display_name", "position", "vbd_value"]
    ].reset_index(drop=True).add_prefix("vbd_")
    comparison = pd.concat([raw_top15, vbd_top15], axis=1)
    comparison.index = comparison.index + 1
    print(comparison.to_string())

    print()
    print("=== Worked example: biggest raw-vs-VBD rank mover inside the top 100 ===")
    relevant = vbd_df[(vbd_df["raw_rank"] <= 100) | (vbd_df["vbd_rank"] <= 100)]
    example = relevant.loc[relevant["rank_diff"].abs().idxmax()]
    print(
        f"{example['player_display_name']} ({example['position']}): "
        f"blended_points={example['blended_points']:.1f}, "
        f"replacement_level[{example['position']}]={example['replacement_level']:.1f}, "
        f"vbd_value={example['vbd_value']:.1f}. "
        f"Raw rank {int(example['raw_rank'])} -> VBD rank {int(example['vbd_rank'])} "
        f"({'up' if example['rank_diff'] > 0 else 'down'} {abs(int(example['rank_diff']))} spots)"
    )

    print()
    print("=== Tier counts per position ===")
    for position in FIXED_POSITIONS:
        n_tiers = vbd_df.loc[vbd_df["position"] == position, "tier"].max()
        n_players = (vbd_df["position"] == position).sum()
        print(f"{position}: {n_tiers} tiers across {n_players} players")

    print()
    print("=== QB and TE: explicit second look (small-n, outlier-sensitive positions) ===")
    for position in ["QB", "TE"]:
        debug = gap_debug[position]
        print(f"{position}: median_gap={debug['median_gap']:.2f}, mad={debug['mad']:.2f}, "
              f"threshold={debug['threshold']:.2f}")
        print(f"{position} raw gaps: {np.round(debug['gaps'], 1).tolist()}")
