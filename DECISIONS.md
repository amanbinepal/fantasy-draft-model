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
