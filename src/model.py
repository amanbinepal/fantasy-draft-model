"""Ridge regression projection model, fit separately per position.

Position-specific feature columns rather than one uniform list: each
position only sees the stat categories that actually apply to it (QB
gets passing + rushing, no receiving; RB/WR get rushing + receiving, no
passing; TE gets receiving only). Keeps the story simple ("why is
passing_yards a WR feature" shouldn't come up) and matters more for QB's
smaller sample specifically.

Ridge, not a more flexible model like XGBoost, because these training
sets are small (a few hundred to ~1400 rows per position) and Ridge's
L2 penalty shrinks noisy coefficients instead of letting the model chase
patterns that don't generalize. Regularization strength (alpha) is
picked per position by RidgeCV via built-in cross-validation rather than
hand-picked, since QB's ~476 rows and WR's ~1400 plausibly want different
amounts of shrinkage. Features are standardized first (Pipeline with
StandardScaler): the penalty only makes sense when features are on
comparable scales, otherwise it shrinks large-scale columns (passing_yards,
in the hundreds) unfairly hard relative to small-scale ones
(passing_interceptions, under 5).

Known limitation: RidgeCV's leave-one-out CV assumes roughly independent
rows, but the same player contributes a row per season pair (Tom Brady's
2019 and 2020 rows both exist), so folds aren't fully independent. This
only affects alpha selection, not the held-out evaluation below (which
splits by season, not row). Phase 4's walk-forward backtest is the real
generalization check regardless of how alpha got picked here.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_pipeline import load_config, pull_weekly_stats
from src.features import build_season_table, build_training_table
from src.scoring import load_scoring_settings

BASE_FEATURES = ["games_played", "fantasy_points", "fantasy_points_per_game"]

QB_STATS = [
    "attempts", "completions", "passing_yards", "passing_tds",
    "passing_interceptions", "carries", "rushing_yards", "rushing_tds",
]
RB_WR_STATS = [
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
]
TE_STATS = ["receptions", "targets", "receiving_yards", "receiving_tds"]


def _with_per_game(stats):
    """Each stat plus its _per_game counterpart, matching the columns
    features.py actually produces."""
    cols = []
    for stat in stats:
        cols.append(stat)
        cols.append(f"{stat}_per_game")
    return cols


POSITION_FEATURES = {
    "QB": BASE_FEATURES + _with_per_game(QB_STATS),
    "RB": BASE_FEATURES + _with_per_game(RB_WR_STATS),
    "WR": BASE_FEATURES + _with_per_game(RB_WR_STATS),
    "TE": BASE_FEATURES + _with_per_game(TE_STATS),
}

ALPHAS = np.logspace(-2, 4, 50)


def train_test_split_by_season(table):
    """Forward-chaining, single fold: train on everything through the
    2023->2024 pair, test on the 2024->2025 pair. Not a random split,
    a season boundary, so nothing from the test season's collection
    touches training. Previews Phase 4's fuller multi-fold walk-forward
    without building the whole thing now."""
    train = table[table["target_season"] <= 2024]
    test = table[table["target_season"] == 2025]
    return train, test


def fit_position_model(train_df, position):
    """Fit a StandardScaler + RidgeCV pipeline for one position."""
    rows = train_df[train_df["position"] == position]
    feature_cols = POSITION_FEATURES[position]
    X = rows[feature_cols]
    y = rows["target_fantasy_points"]

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])
    pipeline.fit(X, y)
    return pipeline


def train_final_models(table):
    """Fit one pipeline per position on the entire labeled table (all
    target_seasons, 2020-2025), not just the pre-2025 split used for
    evaluation below. This is the model later phases use to project the
    2026 season from 2025 features. Kept separate from the held-out
    split: that split exists only to answer "does Ridge beat naive," it
    is not the deployed model."""
    return {
        position: fit_position_model(table, position)
        for position in POSITION_FEATURES
    }


def naive_baseline(df):
    """"Assume it repeats": this season's own total as next season's
    prediction. No fitting."""
    return df["fantasy_points"]


def evaluate(y_true, y_pred):
    """MAE (fantasy points, directly interpretable) and Spearman rank
    correlation (draft usefulness is about ranking players correctly
    relative to each other, not hitting an exact point total)."""
    mae = mean_absolute_error(y_true, y_pred)
    rank_corr, _ = spearmanr(y_true, y_pred)
    return {"mae": mae, "spearman": rank_corr}


if __name__ == "__main__":
    config = load_config()
    seasons = config["data"]["seasons"]

    df = pull_weekly_stats(seasons)
    scoring_settings = load_scoring_settings()
    season_df = build_season_table(df, scoring_settings)
    table = build_training_table(season_df)

    train, test = train_test_split_by_season(table)
    print(f"Train rows: {len(train)} (target_season <= 2024)")
    print(f"Test rows: {len(test)} (target_season == 2025)")

    print()
    print("=== Ridge vs. naive baseline, per position ===")
    for position in POSITION_FEATURES:
        test_rows = test[test["position"] == position]
        if len(test_rows) == 0:
            print(f"{position}: no test rows, skipping")
            continue

        pipeline = fit_position_model(train, position)
        X_test = test_rows[POSITION_FEATURES[position]]
        y_true = test_rows["target_fantasy_points"]

        ridge_pred = pipeline.predict(X_test)
        naive_pred = naive_baseline(test_rows)

        ridge_scores = evaluate(y_true, ridge_pred)
        naive_scores = evaluate(y_true, naive_pred)
        picked_alpha = pipeline.named_steps["ridge"].alpha_

        beat_naive = ridge_scores["mae"] < naive_scores["mae"]
        print(
            f"{position} (n={len(test_rows)}, alpha={picked_alpha:.3g}): "
            f"Ridge MAE={ridge_scores['mae']:.2f}, Spearman={ridge_scores['spearman']:.3f} | "
            f"Naive MAE={naive_scores['mae']:.2f}, Spearman={naive_scores['spearman']:.3f} | "
            f"Ridge beat naive on MAE: {beat_naive}"
        )
