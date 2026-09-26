# Fantasy Football Project Plan

## Raw Datasets

*Can update later*

*Every fetch script takes `--years` so one season can be refreshed without
re-downloading the rest: `python data_fetching/fetch_play_by_play.py --years 2026`.*

- [x] Player box scores / game stats — 2010–2026 (2026 through week 2)
    - RESOLVED: `nfl_data_py.import_weekly_data` 404s for 2025+ (nflverse retired that release). `fetch_box_scores.py` now falls back to `nflreadpy.load_player_stats`, renaming `team`/`passing_interceptions`/`sacks_suffered`/`sack_yards_lost` back to the legacy header. Validated on 2024 where both feeds exist: counting stats match 100%, `fantasy_points_ppr` 99.8%, and the offensive-activity filter reproduces 5,339 of 5,340 legacy player-weeks. `dakota` is the one legacy column nflverse dropped — written as blank.
    - 2025 and 2026 box score CSVs now exist, so Layer 1's play-by-play rebuild no longer fires for 2025. Delete `box_scores_2025.csv` to go back to the derived path.
- [x] Player biographical data (age, experience, position) — 2010–2026
- [x] Schedules — 2010–2026 (2026 weeks 1–2 have final scores; lines posted through week 4)
- [x] Team & coaching records — 2010–2026 (`import_team_desc` is static, no season dimension)
- [x] Weather — 2010–2026
    - pulled from nflverse's schedules dataset (`nfl_data_py.import_schedules`, sourced from habitatring.com/games.csv), not scraped — pro-football-reference.com blocks scrapers via Cloudflare
    - only populated once a game is near; 2026 has temp/wind for 20 of 32 played games (domes are null by design) and nothing for future weeks
- [ ] Vegas betting lines — 2013–2020 ONLY
    - nflverse stopped publishing the scoring-lines release after 2020. `import_sc_lines` still resolves but returns zero rows for 2021–2026, so `vegas_team_line` is NaN for every season since. Not a 2026 problem — it has been dead for five seasons.
    - the spread/total the model actually uses (`spread_line`, `total_line`) come from the schedules feed and are current
- [x] Injury reports — 2010–2026 (2026 through week 3)
    - on Windows, `nfl_data_py.import_injuries` needs the `tzdata` package installed (Windows has no built-in IANA timezone database, which pandas needs for the tz-aware `date_modified` column)
- [x] Play-by-play / target share — 2010–2026 (2026 through week 2, 5,489 plays)
- [x] Next Gen Stats (NGS) — 2016–2026 (2026 through week 2)
    - covers passing, rushing, and receiving via `nfl_data_py.import_ngs_data`, saved to separate subfolders under `data/raw/ngs/`
- [x] Depth charts — 2010–2026 (2026 through the upcoming week)
    - a week's chart is the last snapshot published before that week's first kickoff (by game time, not date). The one week that has not kicked off yet gets the newest snapshot. The feed publishes about twice a day, and `prediction.py` re-fetches the 2026 file when it is over 12h old (`--no-refresh` to skip)
    - the old rule skipped the upcoming week, so week-3 projections fell back to the week-2 chart and missed JAX moving Parker Washington to WR1 on Sep 21
- [x] O-line rankings (weekly) — 2010–2026 (2026 through week 2, derived from play-by-play)
- [x] Strength of schedule — 2010–2026 (2026 `adjusted_sos` now real, but off a 2-game sample)
- [~] Offensive coordinators — 2010–2026
    - nflverse has no coaching-staff endpoint, so 2026 coordinator NAMES are still carried forward from 2025 and will be wrong wherever a team changed OCs
    - the tendency columns ARE real 2026 now (computed from 2026 play-by-play); the `source` column records which is which



## Weekly projections

`prediction.py "Player Name"` projects the next unplayed week off the 2026
schedule; `--week N` projects a specific one. Baseline features still come from
the player's 2025 game log (models are trained through 2025), with everything
2026 has settled before kickoff layered on top: games played, target share,
usage/target-share trends, depth chart, o-line rank, injuries, and the game's
own line and rest. A past week can be re-run as a backtest and correctly sees no
2026 data it should not.

STILL PINNED TO 2025 in projections:
- `def_strength_vs_*`, `opp_def_strength_for_position`, `matchup_advantage_score`
  come from `team_aggs.parquet`, which Layer 2 has only built through 2025.
  Running Layers 1–3 with 2026 included would make these current.

ADD LATER-
- oline rankings PER WEEK (effects the qb play and overall offense)
- offensive coordinator tracking for each offnse
    - goal is to have the model track the offensive coordinator success or     lack of success (via patterns/correlations) to find the bad and good offesive coordinatorss
    - alsop have it detect patterns in offensive coordinator playcalling (ie target shares to wr1/rb/te, qb performance correlations with OC, etc.)
-strength of schedule data



ISSUES:
- Kaelon Black is a backup running back to christian mccaffery and is projected 17 points which is insane.
       - backup running backs need to be prokected a lot lower. this projection only makes sense if mccaffery is out/injured. (Maybe pay more attention to individual history)??
            - maybe add depth chart information to deduce the rb heirarchies
                - Normal backs tho:
                    - Tank Bigsby
                    - Justice Hill (What are consistent about these?)
                    
- DNP Issue (in claude)
- Tight End formula is fucked
- Qbs a little too conservative in projection
- Jags wr core kinda weird 
    - why is meyers projecterd more than parker washingotn and brian thomas when he is literally wr3
        - *Heirarichies seem messed up*
- Is the answer to import a depth chart? (I would want to prioritize this first before the other issues)
- 
