# Fantasy Draft Model: Project Plan (from scratch)

Draft day: **Aug 21, 2026**. Today: Aug 10. ~11 days, worked in evening
sessions. Priority: a projection model you actually understand and can
defend, first. The live tracker's UI can stay simple, but its
recommendation logic (roster-need-aware, not just a raw sorted list) is
core, not optional. See Phase 6 for exactly what "core" means here.

## Ground rules for how we build this

- A working agreement kept locally, not checked into the repo: a
  plan-first workflow where the approach gets proposed and reviewed
  before any code is written, one phase/file at a time, small commits.
- End of each session: 2-3 sentences in `DECISIONS.md` on what you
  decided and *why*. This becomes your interview talk-track later,
  things like "why Ridge over XGBoost for this" or "why walk-forward
  not a single split." Write it in your own words.
- At each phase below, don't move on until you can explain the
  "checkpoint" question in your own words. If you can't, that's the
  signal to slow down, not a signal something's broken.

## Repo skeleton (create fresh, nothing copied in)

```
fantasy-draft-model/
  data/
    raw/            # cached pulls, gitignored
    processed/       # cleaned training tables
  src/
    data_pipeline.py   # pull + cache historical stats
    scoring.py           # league's exact scoring rules as a function
    features.py           # historical stats -> training table
    model.py                # Ridge regression, per position
    vbd.py                    # value-based drafting / replacement level
    sleeper_api.py               # Sleeper API wrapper
    live_tracker.py                # simple polling loop, draft day
  backtest/
    run_backtest.py
  notebooks/           # exploration, kept messy on purpose
  config.yaml           # league settings, no code
  environment.yml
  DECISIONS.md
  README.md
```

(`CLAUDE.local.md` also lives at the repo root, but it's gitignored and
won't show up when you list tracked files.)

## League settings to put in `config.yaml` up front

10-team full-PPR snake, 40+ yard TD bonus, 1QB/2RB/2WR/1TE/2FLEX/6BN/2IR,
14 rounds, `league_id: 1312630302115377152`. Get the current `draft_id`
via `GET /v1/league/<league_id>/drafts` once Sleeper has it scheduled.

## Phase 0: Setup (Day 1, ~30 min)

- `conda env create -f environment.yml`, `git init`, first commit.
- **Checkpoint:** can you explain what each dependency is for, without
  looking it up?

## Phase 1: Historical data (Day 1-2)

- Pull weekly NFL player stats, a few recent seasons, via `nfl_data_py`
  (or raw nflverse CSVs if the sandbox blocks the package's release
  downloads; that CDN block is a known infra quirk, not a bug in your
  code, so don't burn hours assuming you did something wrong).
- Heads-up on a real gotcha: this data won't have your league's 40+
  yard TD bonus computed anywhere. That needs play-by-play data
  (yardage per TD) or an approximation. Decide now whether that
  precision is worth the extra data pull, or whether you approximate
  it and note the limitation.
- **Checkpoint:** what does one row of your raw table represent, and
  what's your definition of "games played" (roster presence vs. actual
  snaps, this bit a past version of this project)?

## Phase 2: Feature engineering (Day 3-4)

- Build the training table: prior-season usage stats to next-season
  fantasy points, using your league's exact scoring from `scoring.py`.
- Decide train/test splits with leakage in mind. You're predicting
  forward, so nothing from a player's target season should leak into
  their features.
- **Checkpoint:** if someone asked "what's your target variable and
  what would leak into it if you weren't careful," what's your answer?

## Phase 3: Projection model (Day 5-6)

- Ridge regression per position (QB/RB/WR/TE separately, since sample
  sizes are small and position usage patterns differ a lot).
- Rookies have no NFL history. Decide now: build a separate
  draft-capital-based heuristic for them, or explicitly scope them out
  as a known limitation for v1. Either is fine; know which you picked
  and why.
- Compare against a naive baseline (e.g. last season's points). If
  Ridge doesn't clearly beat naive, that's worth knowing, not hiding.
- **Checkpoint:** why Ridge (regularization + small samples) instead
  of a more flexible model like XGBoost, for this specific problem?

## Phase 4: Validation (Day 6-7)

- Walk-forward backtest: for each past season, train on prior years,
  predict that season, compare to what actually happened.
- **Checkpoint:** what does "the model beat the baseline X% of the
  time" actually mean, and what would make you *not* trust that number?

## Phase 5: VBD scoring (Day 7-8)

- Convert projections to draft rank via value-based drafting: points
  above replacement, replacement level set by your league's actual
  roster shape (10 teams, double FLEX matters here).
- **Checkpoint:** why does VBD beat raw projected points for ranking
  across positions? What breaks if you skip it and just sort by points?

## Phase 6: Live tracker (Day 9-10), built to be genuinely useful mid-draft

Core, this is what makes the tool worth having open during the draft:

- `live_tracker.py`: poll `GET /v1/draft/<draft_id>/picks` every 3-5s,
  diff against last seen picks, remove drafted players from the pool.
- `recommend.py`: blend the VBD rankings with your roster's current
  state (which starting slots are already filled) so the tool surfaces
  the best value at positions you still need, not just best overall.
- One highlighted "recommended next pick" at the top of the view, not
  just a sorted list you have to interpret yourself. In a 60-90 second
  window you want an answer, not raw data to parse.

Stretch, add only if the phases above finish with time to spare:

- Tier-break alerts ("last elite TE on the board, next tier drops off
  hard").
- Positional-run detection ("3 RBs gone in the last 4 picks league-wide").
- Auto-refreshing UI instead of a manual refresh button.
- FantasyPros/expert-ECR blending as a sanity check layered on top of
  your own model.

If time runs short close to Aug 21, cut stretch items and UI polish
before cutting anything in "core." A plain-text list with the right
recommendation logic beats a pretty dashboard that's just sorting by
raw points.

- **Checkpoint:** what's your plan if Sleeper's API is briefly
  unreachable mid-draft? Does the tool crash or degrade gracefully?

## Phase 7: Dry run and draft day (Day 10-11)

- Start a free Sleeper mock draft, run the tool against it end to end
  on whatever device you'll actually use on Aug 21.
- Refresh the underlying stats one more time the morning of the 21st
  if anything material changed (injury news, depth chart shifts).

## Kicking off the first working session

Scope the first session to Phase 0 and Phase 1 only: set up the
environment and repo skeleton, then work through an approach for
pulling a few seasons of weekly NFL player stats, with the reasoning
and the games-played definition tradeoff talked through before any
code gets written.

Stop there for the first session. Everything past Phase 1 waits until
you've actually looked at the raw data with your own eyes.
