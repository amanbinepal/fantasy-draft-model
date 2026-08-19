"""Walk-forward backtest: Ridge vs. naive, across every season we can
honestly test.

For each test_season, train only on target_seasons strictly before it,
predict test_season, compare to what actually happened. 2020 isn't a
valid test season, there's no earlier target_season to train on (2019 is
the earliest feature season), so the first fold is 2021.

Scoped to Ridge vs. naive, not the three-way (Ridge, ECR, blend)
comparison PROJECT_PLAN.md originally asks for: FantasyPros' historical
rankings aren't available for free (checked directly, historical/bulk
access is a paid Commercial-tier API feature), so there's no honest way
to backtest ECR or the blend against past seasons. Any proxy built from
data we already have would just be re-derived from box scores, which
defeats the reason ECR is useful in the first place. See DECISIONS.md.
"""

import pandas as pd

from src.data_pipeline import load_config, pull_weekly_stats
from src.features import build_season_table, build_training_table
from src.model import POSITION_FEATURES, evaluate, fit_position_model, naive_baseline
from src.scoring import load_scoring_settings

WALK_FORWARD_TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]


def run_fold(table, test_season):
    """Fit Ridge and compute naive for every position, on one
    forward-chaining train/test split. Returns a list of per-position
    result dicts."""
    train = table[table["target_season"] < test_season]
    test = table[table["target_season"] == test_season]

    results = []
    for position in POSITION_FEATURES:
        train_rows = train[train["position"] == position]
        test_rows = test[test["position"] == position]
        if len(test_rows) == 0:
            continue

        pipeline = fit_position_model(train_rows, position)
        X_test = test_rows[POSITION_FEATURES[position]]
        y_true = test_rows["target_fantasy_points"]

        ridge_scores = evaluate(y_true, pipeline.predict(X_test))
        naive_scores = evaluate(y_true, naive_baseline(test_rows))

        results.append({
            "test_season": test_season,
            "position": position,
            "train_n": len(train_rows),
            "test_n": len(test_rows),
            "ridge_mae": ridge_scores["mae"],
            "ridge_spearman": ridge_scores["spearman"],
            "naive_mae": naive_scores["mae"],
            "naive_spearman": naive_scores["spearman"],
        })
    return results


def run_backtest(table):
    """Walk-forward across every test season, one results row per
    (test_season, position)."""
    all_results = []
    for test_season in WALK_FORWARD_TEST_SEASONS:
        all_results.extend(run_fold(table, test_season))
    return pd.DataFrame(all_results)


def summarize(results):
    """Per position: win rate (folds where Ridge MAE beat naive MAE),
    mean MAE, mean Spearman, for both Ridge and naive."""
    def _summarize_position(group):
        wins = (group["ridge_mae"] < group["naive_mae"]).sum()
        return pd.Series({
            "folds": len(group),
            "ridge_wins": wins,
            "ridge_mae_mean": group["ridge_mae"].mean(),
            "naive_mae_mean": group["naive_mae"].mean(),
            "ridge_spearman_mean": group["ridge_spearman"].mean(),
            "naive_spearman_mean": group["naive_spearman"].mean(),
        })

    return results.groupby("position").apply(_summarize_position, include_groups=False)


if __name__ == "__main__":
    config = load_config()
    seasons = config["data"]["seasons"]

    df = pull_weekly_stats(seasons)
    scoring_settings = load_scoring_settings()
    season_df = build_season_table(df, scoring_settings)
    table = build_training_table(season_df)

    results = run_backtest(table)

    print("=== Per-fold results ===")
    print(results.to_string(index=False))

    summary = summarize(results)
    print()
    print("=== Per-position summary ===")
    print(summary.to_string())

    print()
    print("=== Which model actually won, per position (Ridge vs. naive only) ===")
    for position, row in summary.iterrows():
        folds = int(row["folds"])
        wins = int(row["ridge_wins"])
        winner = "Ridge" if row["ridge_mae_mean"] < row["naive_mae_mean"] else "Naive"
        print(
            f"{position}: Ridge beat naive on MAE in {wins}/{folds} folds "
            f"({100 * wins / folds:.0f}%). Mean MAE: Ridge={row['ridge_mae_mean']:.2f}, "
            f"Naive={row['naive_mae_mean']:.2f}. Winner on aggregate: {winner}"
        )
