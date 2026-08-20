"""Poll Sleeper's live draft, diff against last-seen picks, shrink the
available player pool. Console-first, no UI yet: roster-need-aware
recommendation is recommend.py's job, this file only tracks who's still
on the board.

The board (VBD scores, tiers) is built once at startup, not recomputed
per poll: Ridge/ECR/VBD don't change during the draft, only which
players are still available does.

Matching a live pick to the board reuses ecr_blend.normalize_name /
normalize_team, the same approach already proven on this exact problem
in Phase 3 (matching FantasyPros ECR to our board). Sleeper's own ID
system doesn't help here: its gsis_id crosswalk to our nflverse
player_id is real but only ~19% covered on our own relevant players
(confirmed directly against the API), so name-based matching, not an ID
join, has to be the primary mechanism.

Unmatched picks are never force-matched: a wrong guess could silently
remove the *wrong* player from the pool, worse than leaving a drafted
player visible. Instead they're tracked and re-printed at the top of
every status update until manually noticed, not just logged once and
allowed to scroll off screen during a multi-hour draft.

Graceful degradation on API failure (PROJECT_PLAN.md's Phase 6
checkpoint): a poll failure prints a warning and retries next cycle, it
never crashes the loop. This works cleanly with the pick_no-based
diffing, a transient outage just means the next successful poll picks
up everything that accumulated since the last one.
"""

import difflib
import sys
import time

import pandas as pd

from src.data_pipeline import pull_weekly_stats
from src.ecr_blend import current_player_pool, normalize_name, normalize_team
from src.recommend import recommend_next_pick
from src.sleeper_api import pull_draft_metadata, pull_draft_picks, pull_user_id
from src.vbd import _build_blended_table, assign_tiers, compute_vbd

POLL_INTERVAL_SECONDS = 5
FUZZY_CUTOFF = 0.85  # matches ecr_blend.py's cutoff, same reasoning


def build_tracker_board():
    """The VBD+tier board, built once, enriched with normalized_name and
    team/normalized_team for matching against live picks."""
    blended, config = _build_blended_table()
    vbd_df, _replacement, _startable = compute_vbd(blended, config)
    vbd_df, _gap_debug = assign_tiers(vbd_df)

    board = vbd_df.reset_index(drop=True)
    board["normalized_name"] = board["player_display_name"].apply(normalize_name)

    seasons = config["data"]["seasons"]
    current_season = max(seasons)
    df = pull_weekly_stats(seasons)
    pool = current_player_pool(df, current_season)
    board = board.merge(pool[["player_id", "team"]], on="player_id", how="left")
    board["normalized_team"] = board["team"].apply(
        lambda t: normalize_team(t) if pd.notna(t) else None
    )
    return board, config


def match_pick(pick, available_board):
    """Match one live pick to the current available board. Returns
    (matched_index, match_type); match_type in {"exact", "fuzzy",
    "ambiguous", "unmatched"}, matched_index is None for the latter two.
    A single incremental match, not ecr_blend.match_ecr_to_players
    (shaped for batch-matching hundreds of rows at once), but reusing
    the same proven normalization functions."""
    full_name = f"{pick['first_name']} {pick['last_name']}"
    normalized = normalize_name(full_name)

    candidates = available_board[available_board["normalized_name"] == normalized]

    if len(candidates) == 1:
        return candidates.index[0], "exact"

    if len(candidates) > 1:
        pick_team = normalize_team(pick["team"]) if pd.notna(pick.get("team")) else None
        if pick_team is not None:
            team_matches = candidates[candidates["normalized_team"] == pick_team]
            if len(team_matches) == 1:
                return team_matches.index[0], "exact"
        return None, "ambiguous"  # multiple candidates, no clean tiebreak: never guess

    close = difflib.get_close_matches(
        normalized, available_board["normalized_name"].tolist(), n=1, cutoff=FUZZY_CUTOFF
    )
    if close:
        match_idx = available_board[available_board["normalized_name"] == close[0]].index[0]
        return match_idx, "fuzzy"

    return None, "unmatched"


def resolve_my_roster_id(username, draft_id):
    """Resolve a Sleeper username to the roster_id it owns in a specific
    draft, via user_id -> draft_order -> slot_to_roster_id. Resolved
    fresh each run rather than hardcoded in config.yaml, so it can't go
    stale if draft order were ever reset before the draft locks in."""
    user_id = pull_user_id(username)
    draft_meta = pull_draft_metadata(draft_id)
    my_slot = draft_meta["draft_order"].get(user_id)
    if my_slot is None:
        raise ValueError(f"{username} (user_id {user_id}) not found in this draft's draft_order")
    return draft_meta["slot_to_roster_id"][str(my_slot)]


def _print_unresolved(unresolved_picks):
    """Reprints every cycle, not just once when a pick first fails to
    match, so it can't scroll off screen and get forgotten mid-draft."""
    if unresolved_picks:
        print(f"*** {len(unresolved_picks)} unresolved pick(s) need manual review: ***")
        for p in unresolved_picks:
            print(f"    Pick {p['pick_no']}: {p['name']} ({p['position']}, {p['team']})")


def _print_recommendation(available, my_positions, config):
    """The one highlighted answer at the top of every cycle, not a list
    to interpret. Printed every cycle, not just when a new pick lands:
    it should be answerable at a glance at any moment mid-draft."""
    top, needs, reason = recommend_next_pick(available, my_positions, config)
    print("*** RECOMMENDED NEXT PICK ***")
    if top is not None:
        print(
            f"    {top['player_display_name']} ({top['position']}), "
            f"vbd_value={top['vbd_value']:.1f}, tier {top['tier']}"
        )
    print(f"    Reason: {reason}")


def run_tracker(draft_id, board, config, my_roster_id,
                 poll_interval=POLL_INTERVAL_SECONDS, max_iterations=None):
    """Poll, diff against last-seen picks, match, shrink the pool, and
    print the roster-need-aware recommendation every cycle.
    max_iterations is for testing only, a real draft-day run leaves it
    None and relies on Ctrl+C. Returns (drafted_indices, my_drafted_indices,
    unresolved_picks) so tests can inspect the final state."""
    drafted_indices = set()
    my_drafted_indices = set()
    last_seen_pick_no = 0
    unresolved_picks = []
    consecutive_failures = 0

    print(f"Tracking draft {draft_id}, polling every {poll_interval}s. Ctrl+C to stop.")

    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                picks = pull_draft_picks(draft_id)
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                print(
                    f"[warning] Sleeper API poll failed ({consecutive_failures} in a row): "
                    f"{exc}. Retrying in {poll_interval}s."
                )
                _print_unresolved(unresolved_picks)
                available = board[~board.index.isin(drafted_indices)]
                my_positions = board.loc[list(my_drafted_indices), "position"]
                _print_recommendation(available, my_positions, config)
                if max_iterations is None or iterations < max_iterations:
                    time.sleep(poll_interval)
                continue

            new_picks = picks[picks["pick_no"] > last_seen_pick_no]

            if len(new_picks):
                available = board[~board.index.isin(drafted_indices)]
                pick_lines = []  # buffered, not printed yet: the recommendation
                # must print before these per-pick details, per PROJECT_PLAN.md's
                # "one highlighted answer at the top," but still needs to be
                # computed *after* this cycle's picks are processed, using the
                # up-to-date pool/roster, not a stale pre-cycle state.
                for _, pick in new_picks.iterrows():
                    idx, match_type = match_pick(pick, available)
                    name = f"{pick['first_name']} {pick['last_name']}"
                    if idx is not None:
                        drafted_indices.add(idx)
                        if str(pick["roster_id"]) == str(my_roster_id):
                            my_drafted_indices.add(idx)
                        available = available.drop(index=idx)
                        pick_lines.append(
                            f"Pick {pick['pick_no']}: {name} ({pick['position']}, "
                            f"{pick['team']}) -> matched [{match_type}], removed from pool"
                        )
                    else:
                        pick_lines.append(
                            f"Pick {pick['pick_no']}: {name} ({pick['position']}, "
                            f"{pick['team']}) -> {match_type.upper()}, NOT removed from pool"
                        )
                        unresolved_picks.append({
                            "pick_no": pick["pick_no"], "name": name,
                            "position": pick["position"], "team": pick["team"],
                        })

                last_seen_pick_no = picks["pick_no"].max()

                _print_unresolved(unresolved_picks)
                my_positions = board.loc[list(my_drafted_indices), "position"]
                _print_recommendation(available, my_positions, config)
                print()
                for line in pick_lines:
                    print(line)
                print()
                print("=== Top 10 available by VBD ===")
                top10 = available.sort_values("vbd_value", ascending=False).head(10)
                print(top10[["player_display_name", "position", "vbd_value", "tier"]].to_string(index=False))
                print()
            else:
                _print_unresolved(unresolved_picks)
                available = board[~board.index.isin(drafted_indices)]
                my_positions = board.loc[list(my_drafted_indices), "position"]
                _print_recommendation(available, my_positions, config)
                print(f"[poll] no new picks ({len(picks)} total so far)")

            if max_iterations is None or iterations < max_iterations:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")

    return drafted_indices, my_drafted_indices, unresolved_picks


if __name__ == "__main__":
    board, config = build_tracker_board()
    print(f"Board built: {len(board)} players tracked.")

    # Optional CLI override so a mock-draft dry run never has to touch the
    # real league's draft_id in config.yaml. No arg (draft day) behaves
    # exactly as before: the real draft_id from config. Printed explicitly
    # either way, mixing up which draft is live is the one mistake that's
    # actually dangerous (recommending off the wrong board).
    if len(sys.argv) > 1:
        draft_id = sys.argv[1]
        print(f"Using draft_id: {draft_id} (override)")
    else:
        draft_id = config["league"]["draft_id"]
        print(f"Using draft_id: {draft_id} (from config.yaml)")

    my_roster_id = resolve_my_roster_id(config["league"]["sleeper_username"], draft_id)
    print(f"Resolved my roster_id: {my_roster_id}")

    drafted_indices, my_drafted_indices, unresolved_picks = run_tracker(
        draft_id, board, config, my_roster_id
    )

    print()
    print("=== Session summary ===")
    print(f"{len(drafted_indices)} total picks tracked, {len(my_drafted_indices)} of them mine")
    if unresolved_picks:
        print(f"*** {len(unresolved_picks)} unresolved pick(s) still need manual review: ***")
        for p in unresolved_picks:
            print(f"    Pick {p['pick_no']}: {p['name']} ({p['position']}, {p['team']})")
    else:
        print("No unresolved picks.")
