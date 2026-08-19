"""Blend Ridge point projections with FantasyPros' Expert Consensus
Rankings (ECR): the part of a projection that structurally can't come
from historical box scores (training camp buzz, depth-chart battles,
coordinator changes), and the only signal available at all for rookies,
who have no NFL history for Ridge to train on.

FantasyPros' free API tier only returns sample/non-production data (real
access needs an $8.99/mo HOF subscription), so this reads the manually
exported cheat-sheet CSV instead, per PROJECT_PLAN.md's stated fallback.
Real, inspected shape: RK (overall rank), TIERS, PLAYER NAME, TEAM, POS
(position + positional rank combined, e.g. "WR12"). No points column and
no rookie flag.

Blend mechanics: ECR has no points, but Phase 5's VBD math needs real
point values, not just an averaged rank. Within each position, Ridge's
projections are sorted high to low; a player ECR ranks Nth at that
position gets ecr_implied_points = whatever value sits at Ridge's own
Nth spot in that sorted list, not Ridge's own specific prediction for
that named player. That distinction matters: if it used the player's own
Ridge number instead, ecr_implied_points would just equal ridge_points
for every player, and ECR's actual rank would never factor into the
computation at all, collapsing the blend to Ridge alone regardless of
weight. The rank-based lookup is what lets ECR's disagreement with Ridge
move the blended number. For a rookie with no Ridge projection,
ecr_implied_points is their entire projection.
"""

import difflib
import re

import numpy as np
import pandas as pd

from src.data_pipeline import load_config, pull_weekly_stats
from src.features import build_season_table, build_training_table
from src.model import POSITION_FEATURES, train_final_models
from src.scoring import load_scoring_settings

# Ridge weight per position; ecr gets (1 - weight). Started as a flat 0.5
# everywhere (Phase 3, before any backtest existed). Phase 4's walk-forward
# backtest (Ridge vs. naive) gave real per-position evidence, so these are
# now tiered by how consistently Ridge beat naive there: QB lost clearly
# (1/5 folds, worse MAE, tempered by a Spearman win) so leans more on ECR;
# WR won clearly (4/5 folds, better MAE) so leans more on Ridge; RB and TE
# had contradictory or near-tied evidence, not enough signal in 5 noisy
# folds to move off neutral. See DECISIONS.md: this is a directionally
# justified adjustment from Ridge-vs-naive evidence, not a calibrated
# Ridge-vs-ECR optimum, ECR itself has no historical data to backtest.
POSITION_BLEND_WEIGHTS = {"QB": 0.40, "RB": 0.50, "WR": 0.60, "TE": 0.50}

OFFENSE_POSITIONS = ["QB", "RB", "WR", "TE"]

# ECR's team codes mostly match nflverse's; these two don't, confirmed
# by diffing both sets directly. FA (free agent) has no nflverse
# equivalent, mapped to None: no team-based tiebreak or fuzzy-candidate
# restriction for those rows.
TEAM_MAP = {"JAC": "JAX", "LAR": "LA"}

NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

FUZZY_CUTOFF = 0.85


def normalize_name(name):
    """Lowercase, strip punctuation, strip generational suffixes, so
    "Ja'Marr Chase" and "Michael Pittman Jr." compare cleanly against
    nflverse's naming."""
    if pd.isna(name):
        return ""
    name = name.lower()
    name = re.sub(r"[.']", "", name)
    name = re.sub(r"-", " ", name)
    tokens = [t for t in name.split() if t not in NAME_SUFFIXES]
    return " ".join(tokens)


def normalize_team(team, apply_mapping=True):
    if pd.isna(team) or team == "FA":
        return None
    if not apply_mapping:
        return team
    return TEAM_MAP.get(team, team)


def load_ecr(path="data/raw/fantasypros_ecr.csv"):
    """Read the exported cheat-sheet CSV, drop K/DST (config.yaml's
    roster has neither slot), split POS into position + position_rank."""
    ecr = pd.read_csv(path)
    ecr = ecr.rename(columns={
        "RK": "ecr_rank", "TIERS": "tier", "PLAYER NAME": "player_name",
        "TEAM": "team", "POS": "pos_raw",
    })
    parsed = ecr["pos_raw"].str.extract(r"^([A-Z]+)(\d+)$")
    ecr["position"] = parsed[0]
    ecr["position_rank"] = parsed[1].astype(int)
    ecr = ecr[ecr["position"].isin(OFFENSE_POSITIONS)].reset_index(drop=True)

    ecr["normalized_name"] = ecr["player_name"].apply(normalize_name)
    return ecr[[
        "ecr_rank", "tier", "player_name", "team", "position",
        "position_rank", "normalized_name",
    ]]


def current_player_pool(df, season):
    """One row per player_id active in `season`: display name, position
    (mode, in case of a rare inconsistent label), and most recent team
    (last by week, in case of a mid-season trade). This is the matching
    universe ECR gets resolved against, built straight from the raw
    scored weekly data rather than features.py, since team is purely a
    matching-key concern here, not a modeling feature."""
    season_df = df[df["season"] == season].sort_values("week")
    pool = season_df.groupby("player_id").agg(
        player_display_name=("player_display_name", "first"),
        position=("position", lambda s: s.mode().iat[0]),
        team=("team", "last"),
    ).reset_index()
    pool = pool[pool["position"].isin(OFFENSE_POSITIONS)]
    pool["normalized_name"] = pool["player_display_name"].apply(normalize_name)
    return pool


def match_ecr_to_players(ecr_df, player_pool, normalize_teams=True):
    """Resolve each ECR row to a player_id in player_pool, or mark it
    unmatched. Matches on name first, across all positions (not scoped
    to matching position: if ECR's position disagrees with our stale
    2025 label, that's a real signal, not noise, logged via
    position_changed rather than treated as a non-match). Falls back to
    fuzzy name matching, restricted to the same team when known, for
    anything that doesn't match exactly. `normalize_teams=False` is used
    for the diagnostic re-run that measures how many rows the JAC->JAX /
    LAR->LA mapping actually rescues.
    """
    by_name = player_pool.groupby("normalized_name")

    player_ids, match_types, matched_positions, position_changed = [], [], [], []

    for _, row in ecr_df.iterrows():
        ecr_team = normalize_team(row["team"], apply_mapping=normalize_teams)
        candidates = (
            by_name.get_group(row["normalized_name"])
            if row["normalized_name"] in by_name.groups
            else player_pool.iloc[0:0]
        )

        player_id, match_type, matched_position = None, "unmatched", None

        if len(candidates) == 1:
            player_id, match_type = candidates.iloc[0]["player_id"], "exact"
            matched_position = candidates.iloc[0]["position"]
        elif len(candidates) > 1:
            if ecr_team is not None:
                candidates_norm_team = candidates["team"].apply(
                    lambda t: normalize_team(t, apply_mapping=normalize_teams)
                )
                tie_broken = candidates[candidates_norm_team == ecr_team]
                if len(tie_broken) == 1:
                    player_id, match_type = tie_broken.iloc[0]["player_id"], "exact"
                    matched_position = tie_broken.iloc[0]["position"]
            # still ambiguous after team tiebreak: leave unmatched, logged
        else:
            fuzzy_pool = player_pool
            if ecr_team is not None:
                pool_norm_team = player_pool["team"].apply(
                    lambda t: normalize_team(t, apply_mapping=normalize_teams)
                )
                restricted = player_pool[pool_norm_team == ecr_team]
                if len(restricted) > 0:
                    fuzzy_pool = restricted
            close = difflib.get_close_matches(
                row["normalized_name"], fuzzy_pool["normalized_name"].tolist(),
                n=1, cutoff=FUZZY_CUTOFF,
            )
            if close:
                fuzzy_row = fuzzy_pool[fuzzy_pool["normalized_name"] == close[0]].iloc[0]
                player_id, match_type = fuzzy_row["player_id"], "fuzzy"
                matched_position = fuzzy_row["position"]

        player_ids.append(player_id)
        match_types.append(match_type)
        matched_positions.append(matched_position)
        position_changed.append(
            matched_position is not None and matched_position != row["position"]
        )

    result = ecr_df.copy()
    result["player_id"] = player_ids
    result["match_type"] = match_types
    result["matched_position"] = matched_positions
    result["position_changed"] = position_changed
    return result


def implied_points_by_rank(ecr_df, ridge_projections):
    """Assign ecr_implied_points per ECR row: the value at that row's
    position_rank in Ridge's own sorted (descending) projection list for
    that position. Ranks beyond Ridge's projected player count at a
    position clip to Ridge's lowest projected value there."""
    result = ecr_df.copy()
    implied = []
    for _, row in result.iterrows():
        position_values = (
            ridge_projections.loc[ridge_projections["position"] == row["position"], "predicted_points"]
            .sort_values(ascending=False)
            .to_numpy()
        )
        if len(position_values) == 0:
            implied.append(np.nan)
            continue
        idx = min(row["position_rank"] - 1, len(position_values) - 1)
        implied.append(position_values[idx])
    result["ecr_implied_points"] = implied
    return result


def blend_projections(ridge_projections, ecr_matched, weights=POSITION_BLEND_WEIGHTS):
    """Outer-join Ridge's projections with matched ECR-implied points on
    player_id. Weighted average where both exist, using that row's own
    position's weight (not one constant for every position), whichever
    single source exists otherwise. Unmatched ECR rows (rookies, or
    genuine misses) have no player_id to join on, they're appended
    separately with ecr_implied_points as their entire projection,
    weight is irrelevant there either way."""
    matched = ecr_matched[ecr_matched["player_id"].notna()]
    unmatched = ecr_matched[ecr_matched["player_id"].isna()]

    merged = ridge_projections.merge(
        matched[["player_id", "ecr_implied_points", "match_type", "position_changed"]],
        on="player_id", how="outer",
    )
    row_weight = merged["position"].map(weights)
    merged["blended_points"] = np.where(
        merged["ecr_implied_points"].notna(),
        row_weight * merged["predicted_points"] + (1 - row_weight) * merged["ecr_implied_points"],
        merged["predicted_points"],
    )

    rookie_rows = pd.DataFrame({
        "player_id": None,
        "player_display_name": unmatched["player_name"],
        "position": unmatched["position"],
        "predicted_points": np.nan,
        "ecr_implied_points": unmatched["ecr_implied_points"],
        "match_type": "unmatched",
        "position_changed": False,
        "blended_points": unmatched["ecr_implied_points"],
    })

    return pd.concat([merged, rookie_rows], ignore_index=True)


def _predict_current_season(season_df, models, current_season):
    """Ridge point projections for target_season = current_season + 1,
    generated from each player's current_season features."""
    current = season_df[season_df["season"] == current_season]
    predictions = []
    for position, model in models.items():
        rows = current[current["position"] == position]
        if len(rows) == 0:
            continue
        preds = model.predict(rows[POSITION_FEATURES[position]])
        out = rows[["player_id", "player_display_name", "position"]].copy()
        out["predicted_points"] = preds
        predictions.append(out)
    return pd.concat(predictions, ignore_index=True)


if __name__ == "__main__":
    config = load_config()
    seasons = config["data"]["seasons"]
    current_season = max(seasons)

    df = pull_weekly_stats(seasons)
    scoring_settings = load_scoring_settings()
    season_df = build_season_table(df, scoring_settings)
    table = build_training_table(season_df)

    print(f"Blend weights (Ridge weight per position, ecr gets the rest): "
          f"{POSITION_BLEND_WEIGHTS}")

    models = train_final_models(table)
    ridge_projections = _predict_current_season(season_df, models, current_season)
    print(f"Ridge projections generated for {len(ridge_projections)} players "
          f"(from {current_season} features -> {current_season + 1} projection)")

    player_pool = current_player_pool(df, current_season)
    ecr_df = load_ecr()
    matched = match_ecr_to_players(ecr_df, player_pool)

    print()
    print("=== Match rate per position ===")
    print(matched.groupby(["position", "match_type"]).size().unstack(fill_value=0))

    print()
    print("=== Sample unmatched ECR rows ===")
    print(matched[matched["match_type"] == "unmatched"][["ecr_rank", "player_name", "team", "position"]].head(10))

    print()
    print("=== Sample fuzzy-matched ECR rows (spot-check these) ===")
    fuzzy = matched[matched["match_type"] == "fuzzy"]
    if len(fuzzy):
        print(fuzzy[["ecr_rank", "player_name", "team", "position"]].head(10))
    else:
        print("None found")

    print()
    print("=== Position-changed matches ===")
    changed = matched[matched["position_changed"]]
    if len(changed):
        print(changed[["player_name", "position", "matched_position"]])
    else:
        print("None found")

    implied = implied_points_by_rank(matched, ridge_projections)
    blended = blend_projections(ridge_projections, implied)

    print()
    print("=== Worked veteran example ===")
    veteran = blended[
        blended["match_type"].isin(["exact", "fuzzy"]) & blended["predicted_points"].notna()
    ].dropna(subset=["ecr_implied_points"]).iloc[0]
    print(
        f"{veteran['player_display_name']} ({veteran['position']}): "
        f"ridge={veteran['predicted_points']:.1f}, "
        f"ecr_implied={veteran['ecr_implied_points']:.1f}, "
        f"blended={veteran['blended_points']:.1f}"
    )

    print()
    print("=== QB weight-change example (Aaron Rodgers, flat 0.5 gave 160.5 in Phase 3) ===")
    qb_candidates = blended[
        (blended["player_display_name"] == "Aaron Rodgers")
        & blended["ecr_implied_points"].notna()
    ]
    if len(qb_candidates):
        rodgers = qb_candidates.iloc[0]
        print(
            f"{rodgers['player_display_name']} (QB, weight={POSITION_BLEND_WEIGHTS['QB']}): "
            f"ridge={rodgers['predicted_points']:.1f}, "
            f"ecr_implied={rodgers['ecr_implied_points']:.1f}, "
            f"blended={rodgers['blended_points']:.1f} (should sit closer to "
            f"ecr_implied than the old flat-0.5 blend of 160.5 did)"
        )
    else:
        print("Aaron Rodgers not found with both signals this run, skipping")

    print()
    print("=== Worked rookie/unmatched example ===")
    rookie = blended[blended["match_type"] == "unmatched"].iloc[0]
    print(
        f"{rookie['player_display_name']} ({rookie['position']}): "
        f"ridge=n/a, ecr_implied={rookie['ecr_implied_points']:.1f}, "
        f"blended={rookie['blended_points']:.1f} (equals ecr_implied: "
        f"{rookie['blended_points'] == rookie['ecr_implied_points']})"
    )

    print()
    print("=== Team-normalization impact (JAC->JAX, LAR->LA) ===")
    matched_raw_team = match_ecr_to_players(ecr_df, player_pool, normalize_teams=False)
    # NaN != NaN is True in pandas, so a plain != would count every row
    # that's unmatched in *both* runs (most rookies, unrelated to team
    # codes at all) as "changed." Exclude the both-null case explicitly.
    both_unmatched = matched["player_id"].isna() & matched_raw_team["player_id"].isna()
    id_changed = (matched["player_id"] != matched_raw_team["player_id"]) & ~both_unmatched
    changed_rows = id_changed | (matched["match_type"] != matched_raw_team["match_type"])
    print(
        f"{changed_rows.sum()} of {len(matched)} ECR rows changed match outcome "
        f"between normalized and raw team codes (rescued by the mapping, or "
        f"would've resolved differently/not at all without it)"
    )
