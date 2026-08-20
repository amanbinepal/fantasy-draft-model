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
from pathlib import Path

import pandas as pd
import yaml

from src.data_pipeline import pull_weekly_stats
from src.ecr_blend import current_player_pool, normalize_name, normalize_team
from src.recommend import recommend_next_pick
from src.sleeper_api import pull_draft_metadata, pull_draft_picks, pull_user_id
from src.vbd import _build_blended_table, assign_tiers, compute_vbd, recompute_dynamic_vbd

POLL_INTERVAL_SECONDS = 5
FUZZY_CUTOFF = 0.85  # matches ecr_blend.py's cutoff, same reasoning
WATCHLIST_PATH = "data/watchlist.yaml"


def load_watchlist(path=WATCHLIST_PATH):
    """Hand-maintained personal risk list (team situation, ADP concerns,
    depth-chart battles, ...), not derived from any model: PROJECT_PLAN.md's
    own words, the part the model structurally can't see. Missing file
    means an empty watchlist, not an error, this is optional context, not
    a Phase requirement. Purely a visible flag, never touches VBD,
    ranking, needs, or caps: it's context for a personal judgment call,
    not automation of one."""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return {normalize_name(name): note for name, note in raw.items()}


def build_tracker_board():
    """The VBD+tier board, built once, enriched with normalized_name and
    team/normalized_team for matching against live picks, plus the
    watchlist flag/note for display."""
    blended, config = _build_blended_table()
    vbd_df, _replacement, _startable = compute_vbd(blended, config)
    vbd_df, _gap_debug = assign_tiers(vbd_df)

    board = vbd_df.reset_index(drop=True)
    board["normalized_name"] = board["player_display_name"].apply(normalize_name)

    watchlist = load_watchlist()
    board["on_watchlist"] = board["normalized_name"].isin(watchlist.keys())
    board["watchlist_note"] = board["normalized_name"].map(watchlist).fillna("")

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


def resolve_my_draft_slot(username, draft_id):
    """Resolve a Sleeper username to its draft_slot in a specific draft,
    via user_id -> draft_order. Resolved fresh each run rather than
    hardcoded in config.yaml, so it can't go stale if draft order were
    ever reset before the draft locks in.

    Matches on draft_slot rather than roster_id (a slot_to_roster_id
    lookup used to sit here) because roster_id turned out to be null on
    every single pick in a real mock draft (confirmed directly against
    1395984031421599744's completed picks), silently breaking the
    "is this pick mine" check the whole run. draft_slot is present
    directly on every pick object and stable per team for the entire
    draft (confirmed: exactly 14 picks, one per round, at the same
    draft_slot), so it doesn't depend on that broken hop at all."""
    user_id = pull_user_id(username)
    draft_meta = pull_draft_metadata(draft_id)
    my_slot = draft_meta["draft_order"].get(user_id)
    if my_slot is None:
        raise ValueError(f"{username} (user_id {user_id}) not found in this draft's draft_order")
    return my_slot


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
    it should be answerable at a glance at any moment mid-draft.
    vbd_value/tier are the dynamic, re-baselined-to-the-current-pool
    values (recompute_dynamic_vbd); the static figure is shown alongside
    since the pre-draft number is still worth seeing, not just the live
    one."""
    top, needs, reason, alternative = recommend_next_pick(available, my_positions, config)
    print("*** RECOMMENDED NEXT PICK ***")
    if top is not None:
        print(
            f"    {top['player_display_name']} ({top['position']}), "
            f"vbd_value={top['vbd_value']:.1f} (static {top['static_vbd_value']:.1f}), "
            f"tier {top['tier']} (static {top['static_tier']})"
        )
        if top.get("on_watchlist"):
            note = f": {top['watchlist_note']}" if top["watchlist_note"] else ""
            print(f"    /!\\ ON YOUR WATCHLIST{note}")
            if alternative is not None:
                print(
                    f"    Instead, consider: {alternative['player_display_name']} "
                    f"({alternative['position']}), vbd_value={alternative['vbd_value']:.1f} "
                    f"(static {alternative['static_vbd_value']:.1f}), tier {alternative['tier']}"
                )
            else:
                print("    No non-watchlisted alternative among current candidates.")
    print(f"    Reason: {reason}")


def run_tracker(draft_id, board, config, my_draft_slot,
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

    # available carries the dynamic, re-baselined-each-cycle VBD/tier
    # (recompute_dynamic_vbd), and persists across iterations rather than
    # getting rebuilt from `board` every cycle: rebuilding from `board`
    # would silently drop the dynamic columns back to static and force a
    # redundant recompute every poll even when nothing changed. Only
    # refreshed below when this cycle's picks actually shrink the pool.
    # Dynamic == static at pick 0, nothing's been drafted yet.
    available = recompute_dynamic_vbd(board, config)

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
                # Reuse the persisted, dynamically-scored `available` as-is:
                # nothing about the pool changed, just the poll failed.
                my_positions = board.loc[list(my_drafted_indices), "position"]
                _print_recommendation(available, my_positions, config)
                if max_iterations is None or iterations < max_iterations:
                    time.sleep(poll_interval)
                continue

            new_picks = picks[picks["pick_no"] > last_seen_pick_no]

            if len(new_picks):
                # Plain, unscored view for matching/removal against, kept
                # distinct from the persisted, dynamically-scored `available`
                # below: match_pick only ever reads normalized_name/team, it
                # doesn't care about vbd_value/tier at all.
                working = board[~board.index.isin(drafted_indices)]
                pick_lines = []  # buffered, not printed yet: the recommendation
                # must print before these per-pick details, per PROJECT_PLAN.md's
                # "one highlighted answer at the top," but still needs to be
                # computed *after* this cycle's picks are processed, using the
                # up-to-date pool/roster, not a stale pre-cycle state.
                for _, pick in new_picks.iterrows():
                    idx, match_type = match_pick(pick, working)
                    name = f"{pick['first_name']} {pick['last_name']}"
                    if idx is not None:
                        drafted_indices.add(idx)
                        if pick["draft_slot"] == my_draft_slot:
                            my_drafted_indices.add(idx)
                        working = working.drop(index=idx)
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

                # Only re-baseline VBD when the pool actually changed, not
                # every poll: cheap, but no reason to redo it for nothing.
                available = recompute_dynamic_vbd(working, config)

                _print_unresolved(unresolved_picks)
                my_positions = board.loc[list(my_drafted_indices), "position"]
                _print_recommendation(available, my_positions, config)
                print()
                for line in pick_lines:
                    print(line)
                print()
                print("=== Top 10 available by VBD (dynamic, static alongside) ===")
                top10 = available.sort_values("vbd_value", ascending=False).head(10)
                print(top10[[
                    "player_display_name", "position",
                    "vbd_value", "static_vbd_value", "tier", "static_tier",
                    "on_watchlist",
                ]].to_string(index=False))
                print()
            else:
                _print_unresolved(unresolved_picks)
                # Reuse the persisted, dynamically-scored `available` as-is:
                # the pool hasn't changed since the last cycle that did.
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

    my_draft_slot = resolve_my_draft_slot(config["league"]["sleeper_username"], draft_id)
    print(f"Resolved my draft_slot: {my_draft_slot}")

    drafted_indices, my_drafted_indices, unresolved_picks = run_tracker(
        draft_id, board, config, my_draft_slot
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
