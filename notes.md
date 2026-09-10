# Fantasy Football Project Plan

## Raw Datasets

*Can update later*

- [x] Player box scores / game stats — 2010–2025
    -2025 data not available yet (ask claude what to do)
- [x] Player biographical data (age, experience, position) — 2010–2025
- [x] Schedules — 2010–2025
- [x] Team & coaching records — 2010–2025
- [x] Weather — 2010–2025
    - pulled from nflverse's schedules dataset (`nfl_data_py.import_schedules`, sourced from habitatring.com/games.csv), not scraped — pro-football-reference.com blocks scrapers via Cloudflare
- [x] Vegas betting lines — 2010–2025
- [x] Injury reports — 2010–2025
    - on Windows, `nfl_data_py.import_injuries` needs the `tzdata` package installed (Windows has no built-in IANA timezone database, which pandas needs for the tz-aware `date_modified` column)
- [x] Play-by-play / target share — 2010–2025
- [x] Next Gen Stats (NGS) — 2016–2025 only
    - covers passing, rushing, and receiving via `nfl_data_py.import_ngs_data`, saved to separate subfolders under `data/raw/ngs/`



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
