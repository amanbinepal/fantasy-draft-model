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
    """Recommend the best-value player among still-eligible positions.
    Returns (recommended_row_or_None, needs_dict, reason_string,
    alternative_row_or_None). `alternative` is only ever set when `top`
    is on the watchlist: the next-best candidate under the exact same
    rules (caps, deprioritization, needed-position scope) that isn't
    flagged, answering "so who instead" rather than leaving that as a
    manual lookup mid-draft. `None` when `top` isn't watchlisted, or
    when every remaining candidate happens to be too.

    Ranking is unified around each position's own floor status, not a
    needed-vs-fallback branch split: a position uses dynamic vbd_value
    while its own required count is genuinely unmet (don't wrongly skip
    a position that's quietly running out, if it's actually still
    required), and static_vbd_value once that floor is met, whether it's
    only competing for a shared FLEX slot or fully in bench/depth mode.
    Confirmed real, not theoretical: a real mock draft's needed branch
    let a WR (own floor already met, only flex-competing) win on dynamic
    value over a TE with a genuinely unmet floor, purely because the WR
    pool happened to be thinner at that moment; per-position floor
    status fixes that directly. `needs`/reason text stay exactly what
    they were, this only changes how `top` gets picked.

    draft_caps (config.yaml, league.draft_caps, e.g. {"QB": 2}) exclude
    a position from consideration entirely once its count is reached,
    regardless of value: a strategy preference (last season's own roster
    construction), not derivable from the roster shape the way `needs`
    is, kept as a separate, explicit filtering step.

    bench_deprioritize_until_pick (config.yaml, league, e.g. {"QB": 12})
    additionally excludes a position, once its own floor is met, from
    the *primary* comparison until this many of my own picks have
    happened: positions with no FLEX path (today, just QB, this league
    has no superflex) offer close to zero real bench value once their
    floor is met, a backup QB only ever plays a bye/injury week, unlike
    bench RB/WR/TE, which can fill either FLEX slot any week. Confirmed
    real: a mock draft's backup QB won the bench comparison on raw
    static value alone despite that. Deprioritized, not excluded
    outright, there's a safety net below so it never actually returns
    "nothing left" over this, and it re-enters normally past the
    configured pick count, so a 2nd QB still gets recommended
    eventually, just genuinely late."""
    roster = config["league"]["roster"]
    needs = roster_needs(my_positions, config)
    counts = {}
    for position in my_positions:
        counts[position] = counts.get(position, 0) + 1

    floor_met = {"QB": counts.get("QB", 0) >= roster["QB"]}
    floor_met.update({p: counts.get(p, 0) >= roster[p] for p in FLEX_ELIGIBLE})

    caps = config["league"].get("draft_caps", {})
    at_cap = {position for position, cap in caps.items() if counts.get(position, 0) >= cap}
    eligible_board = available_board[~available_board["position"].isin(at_cap)]

    needed_positions = [
        position for position, needed in needs.items() if needed and position not in at_cap
    ]
    # Real starting need still restricts candidates to just those
    # positions, same as before, this is the core "roster-need-aware, not
    # a raw sorted list" behavior from Phase 6: only fall through to the
    # whole eligible board once no genuine starting need remains.
    scope_board = eligible_board[eligible_board["position"].isin(needed_positions)] \
        if needed_positions else eligible_board

    deprioritize_until = config["league"].get("bench_deprioritize_until_pick", {})
    pick_number = len(my_positions) + 1
    deprioritized = {
        position for position, until_pick in deprioritize_until.items()
        if floor_met.get(position, False) and pick_number < until_pick
    }
    primary_board = scope_board[~scope_board["position"].isin(deprioritized)]
    # Safety net: never actually return "nothing left" just because the
    # only remaining players happen to be in a deprioritized position.
    candidate_pool = primary_board if len(primary_board) > 0 else scope_board

    if needed_positions:
        reason = f"starting slot(s) still open at: {', '.join(needed_positions)}"
    else:
        reason = "all starting slots filled, best player available (bench/depth value)"
    if at_cap:
        reason += f" (excluded, at draft cap: {', '.join(sorted(at_cap))})"
    if deprioritized and len(primary_board) > 0:
        soonest = min(deprioritize_until[p] for p in deprioritized)
        reason += f" (deprioritized until pick {soonest}: {', '.join(sorted(deprioritized))})"

    if len(candidate_pool) == 0:
        return None, needs, "no eligible players left on the board", None

    candidate_pool = candidate_pool.copy()
    is_floor_met = candidate_pool["position"].map(floor_met).fillna(True)
    if "static_vbd_value" in candidate_pool.columns:
        candidate_pool["_effective_value"] = candidate_pool["vbd_value"].where(
            ~is_floor_met, candidate_pool["static_vbd_value"]
        )
    else:
        candidate_pool["_effective_value"] = candidate_pool["vbd_value"]

    candidate_pool = candidate_pool.sort_values("_effective_value", ascending=False)
    top = candidate_pool.iloc[0]

    # A watchlist hit answers "be wary" but not "so who instead": find
    # the next-best candidate under the exact same rules (same caps,
    # same deprioritization, same needed-position scope) that isn't
    # flagged, rather than leaving that as a manual lookup mid-draft.
    alternative = None
    if top.get("on_watchlist"):
        non_watchlist = candidate_pool[~candidate_pool["on_watchlist"]]
        if len(non_watchlist) > 0:
            alternative = non_watchlist.iloc[0]

    return top, needs, reason, alternative


if __name__ == "__main__":
    # Imported here, not at module level: live_tracker imports
    # recommend_next_pick, so a module-level import here would be circular.
    from src.live_tracker import build_tracker_board
    from src.vbd import recompute_dynamic_vbd

    board, config = build_tracker_board()
    # Built the same way live_tracker.py actually feeds recommend_next_pick,
    # not the plain static board: adds the static_* columns the bench-mode
    # branch above reads. Dynamic == static pre-draft, so this is a no-op
    # on the numbers here, but it means this demo actually exercises the
    # real code path instead of only the vbd_value fallback.
    board = recompute_dynamic_vbd(board, config)
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
        top, needs, reason, alternative = recommend_next_pick(board, my_positions, config)
        print(f"needs: {needs}")
        if top is not None:
            print(
                f"recommended: {top['player_display_name']} ({top['position']}), "
                f"vbd_value={top['vbd_value']:.1f}, tier {top['tier']}"
            )
        if alternative is not None:
            print(
                f"  (on watchlist, alternative: {alternative['player_display_name']} "
                f"({alternative['position']}))"
            )
        print(f"reason: {reason}")
