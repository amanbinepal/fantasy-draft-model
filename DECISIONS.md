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

## 2026-08-19: Phase 2, features.py

- Target variable: a player's total fantasy_points in the season right
  after their feature season (target_season = feature_season + 1), one
  row per (player, target_season) pair. What would leak into it if built
  carelessly: computing a season's aggregates from a dataframe that
  still contains other seasons' rows (features and target need to come
  from two genuinely separate slices before any joining happens), or
  matching a player's feature season to their target season by name
  instead of player_id, since a name-string quirk or a mid-career
  position reclassification could silently break the pairing and get
  misread as the player having dropped off entirely.
- Dropout handling: left join, defaulting target_fantasy_points to 0 for
  a player who has a feature-season row but no rows at all next season.
  Chose this over an inner join (which would silently drop those
  players) because bust/injury/retirement is a real outcome a draft pick
  can face, and only ever training on players who stuck around would be
  survivorship bias, the model should see the zero, not just the players
  who kept producing.
- Found this while implementing, not something the original plan called
  out: that dropout rule only means what it's supposed to mean if the
  target season actually exists in our pulled data. The most recent
  feature season (2025) has no 2026 data to pair against, because the
  2026 season hasn't been played yet, that's the season this whole
  project is drafting for. Without checking for that, every single 2025
  player would've gotten target_fantasy_points = 0, indistinguishable
  from a real dropout, when the real reason is "that season doesn't
  exist yet," not "this player produced nothing." Fixed by requiring the
  target season to be a season we actually have data for before pairing
  it, so 2019-2020 through 2024-2025 are real training pairs, and the
  2025 season's rows fall out of the training table entirely. Those 2025
  rows aren't wasted, they become the actual input Phase 3 uses to
  generate live 2026 projections, just not a labeled training row, since
  we don't have a real answer to check them against.
- Feature breadth: lean for this first pass. Season totals and per-game
  rates for the core counting stats, games played, and prior-season
  fantasy_points, no advanced route/target-share metrics (target_share,
  air_yards_share, wopr, racr, pacr) yet. Those are pass-catcher-only and
  add explaining surface without being needed to get a first working
  model going; easy to add later once backtest results show where the
  lean version falls short.
- Known simplification, not fixed today: position is taken from the
  feature season only and never re-checked against the target season, so
  a mid-career position change (rare, but real) would carry a stale
  position label forward into that training row.

## 2026-08-19: Phase 3, model.py

- Ridge regression, fit separately per position (QB/RB/WR/TE), each with
  its own feature subset instead of one uniform list: QB gets passing +
  rushing columns (no receiving), RB/WR get rushing + receiving (no
  passing), TE gets receiving only (no passing or rushing, TE rushing
  usage is negligible enough not to carry even as a near-constant
  column). Chose Ridge over a more flexible model like XGBoost because
  these training sets are small, a few hundred to ~1400 rows per
  position, small enough that a flexible model would likely fit noise
  that doesn't generalize. Ridge's L2 penalty shrinks noisy coefficients
  toward zero instead.
- Wrapped each position's model in a Pipeline(StandardScaler, RidgeCV):
  scaling first because the penalty only makes sense when features are
  on a comparable scale (passing_yards in the hundreds vs.
  passing_interceptions under 5, an unscaled penalty would shrink the
  large-scale column unfairly hard), RidgeCV so alpha gets picked per
  position via built-in leave-one-out CV instead of one hand-picked
  number, since QB's ~476 rows and WR's ~1400 plausibly want different
  amounts of shrinkage. Known limitation: that CV assumes roughly
  independent rows, but the same player contributes a row per season
  pair (e.g. Tom Brady's 2019 and 2020 rows both exist), so folds aren't
  fully independent. Affects alpha selection only, not the held-out
  evaluation, which splits by season, not row. Phase 4's walk-forward
  backtest is the real generalization check regardless of how alpha got
  picked here.
- Naive baseline: a player's own fantasy_points total from their feature
  season, unchanged, as the prediction for next season ("assume it
  repeats"). Notably, this exact number is also one of Ridge's own input
  features for every position, so Ridge isn't blind to it, it has full
  ability to learn a coefficient near 1 on that column and near 0 on
  everything else if that were truly optimal.
- Evaluated on a single forward-chaining split (train: target_season <=
  2024, test: target_season == 2025) rather than a random split, so nothing
  from the test season touches training. Real result, not hidden: Ridge
  beat naive on MAE for WR (33.24 vs. 37.23) and TE (25.03 vs. 27.53), but
  lost narrowly at QB (61.68 vs. 60.83, n=78) and RB (40.76 vs. 39.48,
  n=148). Spearman rank correlation tells a slightly different story,
  Ridge is ahead at QB (0.733 vs. 0.726) despite losing on MAE, and only
  barely behind at RB (0.738 vs. 0.746), so the ranking-quality picture is
  closer to a wash than the MAE numbers alone suggest. Not chasing a fix
  for this today: QB and RB's test sets are thin enough (78 and 148 rows,
  one single season) that this could easily flip on a different fold, and
  Phase 4's full walk-forward (every season as its own test fold, not
  just one) is the real answer to whether Ridge is actually earning its
  complexity over the naive baseline.

## 2026-08-19: Phase 3, ecr_blend.py

- FantasyPros' free API tier turned out to only return sample/non-
  production data (real access needs an $8.99/mo HOF subscription), so
  used the manual CSV cheat-sheet export instead, per PROJECT_PLAN.md's
  stated fallback: `data/raw/fantasypros_ecr.csv`. No points column at
  all, rank only (`RK`, `TIERS`, `POS` as position + positional rank
  combined like `"WR12"`).
- Blend mechanics: since ECR has no points but Phase 5's VBD math needs
  real point values, `ecr_implied_points` for a player ECR ranks Nth at
  a position is whatever value sits at Ridge's own Nth spot in that
  position's sorted projection list, not Ridge's own specific prediction
  for that named player. That distinction is the entire mechanism: using
  the player's own Ridge number instead would make `ecr_implied_points`
  identically equal `ridge_points` for everyone, so ECR's actual rank
  would never factor into the computation and the blend would collapse
  to Ridge alone regardless of weight. Blend weight: 0.5 (equal), a
  starting point with no a priori claim either source is more trustworthy,
  to be tuned once Phase 4's backtest exists.
- Matching: normalized name (lowercase, punctuation stripped, Jr./Sr./
  II/III/IV stripped) matched across all positions, not scoped to
  matching position, since a position disagreement between ECR and our
  stale 2025 label is a real signal (ECR's more current), not noise.
  Fell back to `difflib` fuzzy matching, team-restricted when team is
  known, for anything without an exact name match. Real results: QB
  47/0/5 (exact/fuzzy/unmatched), RB 107/1/30, TE 64/0/14, WR 131/2/42.
  Only 3 fuzzy matches total (Audric Estime, Joshua Palmer, Mitch
  Tinsley), spot-checked, all correct. Most unmatched rows are genuine
  2026 rookies with no 2025 stat-line presence at all (Jeremiyah Love,
  Carnell Tate, Jonathon Brooks, ...), exactly the case the design
  intends to fall back to ECR-implied-points-alone for.
- Team-code normalization (`JAC`->`JAX`, `LAR`->`LA`): confirmed real by
  diffing both team-code sets directly, but its measured rescue count on
  this actual export is 0, not a bug, a mechanical fact. Team only gets
  consulted for tie-breaking a name that matches multiple candidates, or
  restricting the fuzzy-fallback candidate pool. Checked all 29 matched
  JAC/LAR rows directly: every one resolved via a unique exact name
  match, so team normalization never had a case where it was actually
  needed to matter. Kept the mapping anyway, it's still factually
  correct and could matter on a future export where a name collision or
  fuzzy case happens to land on a Jacksonville or Rams player.
- Caught and fixed a real bug in my own verification step while
  computing that rescue count: `matched['player_id'] != matched_raw['player_id']`
  is `NaN != NaN` in pandas, which evaluates `True`, not `False`, so
  every row unmatched in both the normalized and raw-team runs (mostly
  rookies, unrelated to team codes at all) was being counted as
  "changed." First run reported 91 rescued rows; the real number, after
  excluding the both-null case explicitly, is 0. Worth remembering:
  never diff nullable columns with a plain `!=` in pandas without
  handling the both-null case first.
- Flagged, not fixed today: Travis Hunter (Jacksonville, a real 2025
  two-way rookie) shows up unmatched, plausibly because nflverse
  classifies him primarily by his defensive position (CB), which
  `current_player_pool`'s offense-position filter excludes even though
  he has real offensive stats too. A different issue than team
  normalization, noted for later, not chased down now.

## 2026-08-19: Phase 4, backtest/run_backtest.py

- Walk-forward backtest, Ridge vs. naive, across every season we can
  honestly test: test_season in [2021, 2022, 2023, 2024, 2025], training
  on everything strictly before it each time. 2021 is the earliest valid
  test season, 2020 has no earlier target_season to train on at all
  (2019 is the earliest feature season in our data).
- Scoped to Ridge vs. naive only, not the three-way (Ridge, ECR, blend)
  comparison PROJECT_PLAN.md originally asks for. Checked directly:
  FantasyPros' historical rankings aren't available for free, historical/
  bulk access is a paid Commercial-tier API feature. There's no honest
  way to know what FantasyPros actually said in August of 2021-2024
  without that access, and any proxy built from data we already have
  would just be re-derived from box scores, defeating the reason ECR is
  useful in the first place (it carries information box scores can't
  see). The ECR/blend three-way comparison stays un-validated against
  history. Real limitation, not a shortcut.
- Verified the loop itself before trusting its output: train_n grows
  monotonically within every position across folds (QB: 71/153/234/317/
  398, RB: 146/303/468/626/774, WR: 230/463/719/958/1182, TE: 125/255/
  386/514/637), confirming forward-chaining, not leakage. The 2025 fold
  reproduces model.py's single-fold numbers exactly (QB 61.68/60.83, RB
  40.76/39.48, WR 33.24/37.23, TE 25.03/27.53), confirming the walk-
  forward loop is built correctly rather than silently diverging from
  the earlier check.
- Full results, no softening. Ridge only clearly beats naive on MAE at
  WR:

  | Position | Ridge wins | Mean Ridge MAE | Mean Naive MAE | Mean Ridge Spearman | Mean Naive Spearman | Winner (MAE) |
  |---|---|---|---|---|---|---|
  | QB | 1/5 (20%) | 63.24 | 61.65 | 0.692 | 0.682 | Naive |
  | RB | 2/5 (40%) | 43.27 | 43.16 | 0.699 | 0.705 | Naive |
  | TE | 3/5 (60%) | 28.00 | 27.42 | 0.708 | 0.724 | Naive |
  | WR | 4/5 (80%) | 36.75 | 38.52 | 0.735 | 0.742 | Ridge |

  The single Phase 3 fold (2025 only) undersold how bad QB and RB are on
  MAE: that fold showed both losing narrowly, the full walk-forward shows
  QB losing badly (1/5 folds) and RB losing more often than not (2/5).
- TE is the concrete answer to the checkpoint question ("what does
  'beating the baseline X% of the time' actually mean, and what would
  make you not trust that number"): Ridge wins the *majority* of TE
  folds by count (3/5), but still loses on mean MAE, because its two
  losing folds (2021: 34.70 vs. 29.35, 2022: 28.08 vs. 26.22) are bigger
  misses than its three winning folds are wins (0.17, 1.67, 2.50 point
  margins). Win rate and win magnitude point in different directions
  here. A headline of "Ridge wins 3 out of 5 folds at TE" would have
  been true and also would have been the wrong takeaway. With only 5
  folds total at any position, a single flipped fold swings the rate by
  20 points either way, that fragility is real, not a caveat to gloss
  past.
- QB shows the mirror-image pattern, worth stating plainly since it's a
  genuinely different conclusion than "Ridge loses at QB": Ridge loses
  on MAE (63.24 vs. 61.65) and loses on fold count (1/5), but its mean
  Spearman rank correlation actually favors Ridge over naive (0.692 vs.
  0.682). That means Ridge may be getting QB's relative ordering more
  right than naive, more correctly separating who's better than whom,
  even while missing on absolute point totals more often. Whether that
  matters depends entirely on what Phase 5's VBD math ends up caring
  about most: if VBD is more sensitive to getting the rank order right
  than to hitting an exact point total, QB's story isn't as simply
  "Ridge loses" as the MAE table alone suggests.
- Open question flagged for Phase 5, not decided today: the ECR blend
  weight (`ecr_blend.py`'s `BLEND_WEIGHT = 0.5`) was picked before this
  backtest existed, uniform across all four positions. Now that QB and
  RB have real evidence Ridge underperforms naive on MAE while WR
  clearly beats it (and QB's Spearman result complicates even that),
  that uniform 0.5 is carrying forward an assumption this backtest
  didn't have when it was made. Worth deciding in Phase 5 whether the
  blend weight should vary by position instead of staying flat, rather
  than carrying the old assumption forward silently.
