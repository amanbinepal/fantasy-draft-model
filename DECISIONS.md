# Decisions log

Short notes on what I decided and why, written after each session, in my own
words. This is the part I actually want to remember when I talk about this
project later: not just what the code does, but why I chose it.

## 2026-08-10: Phase 0/1

- Picked 7 seasons (2019-2025) for training data. Too few seasons makes the
  model too volatile, not enough examples to learn from. Too many (10+)
  means the older seasons start adding noise instead of signal, since the
  NFL itself has changed. 7 seasons is a good middle ground.
- Defined games played as stat-line presence, not roster presence. If it
  just logged every game a player was rostered for, a player who didn't
  actually play, or played a minute and got injured, would still count as
  a game played. That skews their per-game averages. Logging a game only
  when they have meaningful recorded stats makes the data more accurate.
- Approximating the 40+ yard TD bonus for now instead of pulling
  play-by-play data. Getting it exactly right needs a lot more data and
  compute, and the amount of benefit right now isn't worth that cost.

## 2026-08-10: Phase 1, data source swap and re-validation

- nfl_data_py (0.3.3, the latest published version) turned out to be
  pulling from nflverse's `player_stats` release, which nflverse
  deprecated Aug 1 2025 in favor of `stats_player`. No 2025 data exists
  under the old release, so nfl_data_py can't reach it and won't get
  fixed (no newer version exists). Switched data_pipeline.py to pull
  directly from nflverse's `stats_player_week_{year}.parquet` files via
  pandas.read_parquet for all seasons, for one consistent schema instead
  of stitching two together. Dropped nfl_data_py from environment.yml
  since nothing else needed it, which also undid the pandas/numpy 1.x
  downgrade from earlier today, added pyarrow as the parquet engine
  since that dependency was only ever there transitively through
  nfl_data_py.
- Re-checked the games-played definition against the new source instead
  of assuming the old validation still held. All-zero-stat rows jumped
  from 0.04% (old source, offense-only) to 9.86% (new source, all
  seasons). Investigated rather than treated that as a red flag: the new
  source generates a row for any offense-position player with any
  recorded involvement in a game, including special-teams tackles and
  penalties, not just offensive touches. Those rows are real players who
  actually played and correctly recorded zero fantasy production that
  week, not the roster-presence trap the original definition was meant
  to avoid. Also checked for duplicate player-week rows from the switch:
  none. McCaffrey's 2024 spot check still holds too: 4 games in 2024,
  unchanged from the old source.
- Found `passing_40` / `rushing_40` / `receiving_40` columns in the new
  source: counts of plays gaining 40+ yards, per player per week. These
  don't confirm a play was a touchdown, just that it gained 40+ yards, on
  their own. But see the next note: pulling the league's real scoring
  settings turned this from "approximate it" into "almost solved," since
  we now know the exact bonus these columns would need to be crossed
  with a TD count to reproduce.
- Pulled this league's actual scoring settings from the Sleeper API
  (league_id 1312630302115377152) instead of assuming standard PPR.
  Checked specifically whether return yards/TDs score: return
  touchdowns do (`st_td`: 6.0, same as any other TD), return yardage
  does not (no `st_yd`/`kr_yd`/`pr_yd` key exists at all). Confirms the
  all-zero-stat rows found earlier are genuinely 0-point weeks under
  this league's real rules, not just under an assumed generic PPR, e.g.
  a return specialist with 90 combined return yards and no TD scores 0.
  Also found the actual 40+ yard TD bonus while in there: `rec_td_40p`,
  `rush_td_40p`, `pass_td_40p` are each a flat `2.0`, added on top of the
  normal TD points, same bonus regardless of play type. That changes the
  Phase 0 approximation decision: this isn't "guess a bonus and note the
  limitation" anymore, it's "count TDs where the player's `passing_40` /
  `rushing_40` / `receiving_40` for that game is nonzero, add 2." Still
  imperfect (a player could have a 40+ yard non-TD play and a separate
  shorter TD in the same game and this would over-credit them), but a
  real, mostly-solved approximation instead of a placeholder, ready for
  scoring.py in Phase 3.

## 2026-08-19: Phase 2, scoring.py

- Last session's Sleeper scoring findings never got saved anywhere, just
  written up as prose in this file, so scoring.py couldn't be built
  without re-fetching. Added sleeper_api.py to pull the league's real
  `scoring_settings` and cache it to `data/raw/league_scoring_settings.json`
  with no refresh path: unlike weekly stats, a league's scoring settings
  are fixed once the league is created for the season, so a stale cache
  isn't a real risk here.
- Full scoring weights, not just the pieces checked before: standard PPR
  shape (`rec`: 1.0, `pass_yd`: 0.04, `rush_yd`/`rec_yd`: 0.1, `pass_td`:
  4.0, `rush_td`/`rec_td`: 6.0, `pass_int`/`fum_lost`: -2.0 each), plus
  one real surprise: 2-point conversions score 1.0 here, not the more
  common 2.0. Also confirmed `pass_int` and `fum_lost` come back negative
  from Sleeper directly, not positive magnitudes scoring.py has to negate
  itself.
- `fum_lost` is one Sleeper key but three nflverse columns
  (`rushing_fumbles_lost`, `receiving_fumbles_lost`, `sack_fumbles_lost`),
  since Sleeper scores total fumbles lost regardless of how the ball came
  out. Summed all three under that one weight.
- Kept the 40+ yard TD bonus approximation from Phase 0/1 but scaled it:
  `2 * min(tds, 40+ yard plays)` per category per game instead of a flat
  flag, so a real two-40+-TD game doesn't get under-credited. Confirmed
  against Mark Ingram's 2019 week 1 (2 rush TDs, 1 rushing 40+ play): our
  score came out exactly 2.0 above nflverse's own PPR calc for that game,
  the expected single bonus. Still has the same known edge case as
  before, just named precisely now: a game with one short (non-40+) TD
  and a separate, unrelated 40+ yard play that wasn't a score still
  computes as `min(1, 1) = 1` and gets a bonus it shouldn't, because the
  count-only `passing_40`/`rushing_40`/`receiving_40` columns can't tell
  us the 40+ play and the scoring play were different plays.
