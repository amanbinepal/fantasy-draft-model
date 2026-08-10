# Fantasy Draft Model

A custom fantasy football draft assistant, built from scratch as a personal
data science project ahead of my Aug 21, 2026 Sleeper draft.

## The problem

Snake drafts move fast (roughly 60-90 seconds per pick) and most tools
hand you the same generic expert-consensus rankings everyone else at the
table already has. Those rankings also don't reflect quirks specific to my
league: a 40+ yard TD bonus and a double-FLEX roster shape that changes who's
actually valuable at the margins. I wanted rankings grounded in a model I
built and validated myself, not a black box.

## What this does

- Trains a position-specific Ridge regression model on historical NFL player
  stats to project next-season fantasy points under my league's exact
  scoring rules
- Validates those projections with a walk-forward backtest against past
  seasons, rather than trusting the model on faith
- Converts projections into draft rank using Value-Based Drafting (VBD),
  accounting for this league's 10-team, double-FLEX roster shape
- Polls Sleeper's live draft API on draft day to track picks and surface
  best-available players in real time

## Why build this instead of using a free tool

This is as much a learning project as a practical one. I wanted to actually
understand value-based drafting, walk-forward validation, and where a
from-scratch model beats (or doesn't beat) an off-the-shelf consensus
ranking, not just consume someone else's number.

## Status

Actively being built in the lead-up to Aug 21, 2026. See `PROJECT_PLAN.md`
for the phase-by-phase build and `DECISIONS.md` for the reasoning behind key
choices as they're made.

## League context

- Platform: Sleeper
- 10 teams, full PPR, 40+ yard TD bonus
- Roster: 1 QB / 2 RB / 2 WR / 1 TE / 2 FLEX / 6 BN / 2 IR
- 14-round snake draft

## Tech stack

Python, pandas, NumPy, scikit-learn, the Sleeper public API, conda for
environment management.

## Project structure

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
  CLAUDE.md
  README.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate fantasy-draft-model
```

## Roadmap / known limitations

See `PROJECT_PLAN.md` for the full phase breakdown, checkpoints, and known
gotchas (rookie projections, the 40+ yard TD bonus, etc).
