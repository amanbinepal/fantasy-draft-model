"""Blend the VBD board with my roster's current state: surface the best
value at positions I still need, not just best overall.

Roster-need computation is mechanically derived from config.yaml's
roster shape, not a judgment call: QB has no flex eligibility in this
league, so it's needed iff I haven't drafted one yet. RB/WR/TE share the
2 flex slots on top of their own fixed slots, so they share a single
"still needed" condition: as long as total RB+WR+TE drafted is under
(fixed RB + fixed WR + fixed TE + flex slots), any of the three could
still fill an open starting slot. Once that combined capacity is full,
none of them open a starting slot anymore, further picks at those
positions are pure bench value, which the recommendation still
surfaces (this league has 6 BN + 2 IR slots, real value), just not
flagged as a "need."
"""

FLEX_ELIGIBLE = ["RB", "WR", "TE"]


def roster_needs(my_positions, config):
    """my_positions: an iterable of position strings already drafted for
    my team. Returns {"QB": bool, "RB": bool, "WR": bool, "TE": bool},
    True meaning a starting slot (fixed or flex) is still open."""
    roster = config["league"]["roster"]
    counts = {}
    for position in my_positions:
        counts[position] = counts.get(position, 0) + 1

    qb_needed = counts.get("QB", 0) < roster["QB"]

    # Each flex-eligible position's own fixed floor is a guaranteed
    # requirement first, not a soft target inside the shared pool: every
    # team fields exactly 1 TE starter regardless of how many RB/WR are
    # already rostered. (Real bug this fixes, not theoretical: a real
    # mock draft closed "flex capacity" with 3 RB + 4 WR and zero TEs,
    # confirmed against real pick data, before this function ever
    # flagged TE as needed again.) Only the count *beyond* each
    # position's own floor competes for the shared FLEX slots, the same
    # fixed-floor-before-flex-competes pattern vbd.py's
    # compute_replacement_levels already established.
    own_floor_met = {p: counts.get(p, 0) >= roster[p] for p in FLEX_ELIGIBLE}
    extra_flex_eligible_drafted = sum(
        max(0, counts.get(p, 0) - roster[p]) for p in FLEX_ELIGIBLE
    )
    flex_slots_open = extra_flex_eligible_drafted < roster["FLEX"]

    return {
        "QB": qb_needed,
        **{p: (not own_floor_met[p]) or flex_slots_open for p in FLEX_ELIGIBLE},
    }


def recommend_next_pick(available_board, my_positions, config):
    """Filter to still-needed positions, recommend the highest-VBD
    player among them. Falls back to best-available on the whole board
    if no starting need remains. Returns (recommended_row_or_None,
    needs_dict, reason_string).

    draft_caps (config.yaml, league.draft_caps, e.g. {"QB": 2}) exclude
    a position from consideration entirely, in both branches below, once
    its count is reached, regardless of remaining VBD value: a strategy
    preference (last season's own roster construction), not something
    derivable from the roster shape the way `needs` is, so kept as a
    separate, explicit filtering step rather than folded into
    roster_needs itself."""
    needs = roster_needs(my_positions, config)
    counts = {}
    for position in my_positions:
        counts[position] = counts.get(position, 0) + 1
    caps = config["league"].get("draft_caps", {})
    at_cap = {position for position, cap in caps.items() if counts.get(position, 0) >= cap}

    eligible_board = available_board[~available_board["position"].isin(at_cap)]
    needed_positions = [
        position for position, needed in needs.items() if needed and position not in at_cap
    ]

    if needed_positions:
        candidates = eligible_board[eligible_board["position"].isin(needed_positions)]
        reason = f"starting slot(s) still open at: {', '.join(needed_positions)}"
    else:
        candidates = eligible_board
        reason = "all starting slots filled, best player available (bench/depth value)"

    if at_cap:
        reason += f" (excluded, at draft cap: {', '.join(sorted(at_cap))})"

    if len(candidates) == 0:
        return None, needs, "no eligible players left on the board"

    top = candidates.sort_values("vbd_value", ascending=False).iloc[0]
    return top, needs, reason


if __name__ == "__main__":
    # Imported here, not at module level: live_tracker imports
    # recommend_next_pick, so a module-level import here would be circular.
    from src.live_tracker import build_tracker_board

    board, config = build_tracker_board()
    print(f"Board built: {len(board)} players tracked.")

    scenarios = {
        "Empty roster": [],
        "QB already drafted": ["QB"],
        "RB/WR/TE flex capacity full (2 RB, 2 WR, 1 TE, 2 more flex-eligible)": [
            "RB", "RB", "WR", "WR", "TE", "RB", "WR",
        ],
    }

    for label, my_positions in scenarios.items():
        print()
        print(f"=== Scenario: {label} ===")
        top, needs, reason = recommend_next_pick(board, my_positions, config)
        print(f"needs: {needs}")
        if top is not None:
            print(
                f"recommended: {top['player_display_name']} ({top['position']}), "
                f"vbd_value={top['vbd_value']:.1f}, tier {top['tier']}"
            )
        print(f"reason: {reason}")
