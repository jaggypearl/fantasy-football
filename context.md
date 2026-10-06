# Project Context

Handoff doc for coding agents (written for Codex). Snapshot of the repo as of 2026-10-06, commit `d459d68` ("frontend created"). Everything here was read from the code unless marked **(unverified)**.

## 1. What this is

A full-stack ML project that projects weekly PPR fantasy football output for every QB, RB, WR and TE in the 2026 NFL regular season. Four pieces:

1. **Data fetching**: scripts pull raw nflverse data (2010-2026) into CSVs.
2. **Processing (Layers 1-3)**: join and feature-engineer it into one player-week training table.
3. **Models + projection engine**: one XGBoost model per position per stat, rolled up into PPR points.
4. **Serving**: a batch job precomputes every player's projection for every remaining week into Postgres (Neon). A read-only FastAPI service serves it. A Next.js frontend displays it.

The owner is a CS student who is still learning, and wants to understand each piece, not just receive it. When making changes, explain the reasoning and keep layers cleanly separated. Honest assessments are preferred over optimistic ones. Casual, concise communication.

Target season: `TARGET_SEASON = 2026`. Models are validated against 2025. Week-1 Thursday of 2026 is 2026-09-10.

## 2. Repo layout

```
.
├── api/                    FastAPI read-only service
│   ├── main.py             endpoints, error handling, week logic
│   ├── db.py               SQLAlchemy engine, retrying fetch(), queries
│   └── names.py            player-name resolution (exact / nickname / last name)
├── batch_compute.py        runs prediction.predict for every player-week, writes Postgres
├── prediction.py           the projection engine (1,197 lines; CLI + library)
├── data_fetching/          14 fetchers + season_args.py (shared --years arg)
├── processing/
│   ├── layer1_base_join.py   joins raw sources -> base_processed.parquet
│   ├── layer2_team_aggs.py   team-level aggregates -> team_aggs.parquet
│   ├── layer3_player_features.py  player features -> final_training_data.parquet
│   ├── pbp_box_scores.py     rebuilds box scores from play-by-play (fallback)
│   └── debug_layer1.py       debugging helper
├── model/
│   ├── train.py            trains all per-position/per-stat models
│   ├── model_metadata.json features, scaler params, metrics (4,730 lines, mostly scaler arrays)
│   ├── *_model.pkl         19 live pickled XGBoost models
│   └── archive/            baseline_2026-09-10/ and pre_efficiency_2026-09-24/ snapshots
├── frontend/               Next.js app (see section 9)
├── notes.md                running project plan, dataset status, known issues
├── limitation.md           documented model limitations / v2 ideas
├── requirements-api.txt    API-only dependencies
└── .claude/                Claude Code settings (permissions only; not relevant to Codex)
```

**Not in git** (see `.gitignore`): `data/raw/`, `data/processed/`, `*.parquet`, `.venv/`, `.env`. A fresh clone has **no data and no parquet files**. The pipeline cannot run until the fetchers and Layers 1-3 are re-run. Models (`.pkl`) and `model_metadata.json` ARE committed.

## 3. Environment and dependencies

- Developed on **Windows** (Python 3.14, `.venv/Scripts/python.exe`, PowerShell/Git Bash). Paths in code use `pathlib`, so Linux/macOS should work.
- There is **no root `requirements.txt`**. Pipeline deps inferred from imports: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `nfl_data_py`, `nflreadpy`, `sqlalchemy`, `psycopg2-binary`, `python-dotenv`. On Windows, `nfl_data_py.import_injuries` also needs `tzdata`.
- API deps in `requirements-api.txt`: fastapi, uvicorn, SQLAlchemy 2, psycopg2-binary, python-dotenv, pydantic 2.
- Frontend: Next `^16.3.8`, React `^19.3.0`, TypeScript `^7.0.2`. No UI library; hand-written CSS in `app/globals.css`.
- **No tests exist.** No CI config in the repo **(unverified: no `.github/` was in the file list)**.
- Secrets live in `.env` (not committed): `DATABASE_URL` (used by `batch_compute.py`, for writing) and `DATABASE_URL_POOLED` / `DATABASE_URL` (used by the API).

## 4. Data fetching (`data_fetching/`)

Every fetcher defaults to `START_YEAR=2010`..`END_YEAR=2026` and accepts `--years` via `season_args.py`:

```
python data_fetching/fetch_play_by_play.py --years 2026
python data_fetching/fetch_ngs.py --years 2024-2026
```

Output convention: `data/raw/<dataset>/<dataset>_<year>.csv`. Play-by-play is about 140MB per season. NGS saves into per-stat subfolders under `data/raw/ngs/` (passing, rushing, receiving).

| Dataset | Script | Notes |
|---|---|---|
| Box scores | `fetch_box_scores.py` | `nfl_data_py.import_weekly_data` 404s for 2025+, so it falls back to `nflreadpy.load_player_stats` and renames columns back to the legacy schema. `dakota` is written blank. |
| Bio data | `fetch_bio_data.py` | age, height, weight, draft number, rookie/entry year, position |
| Schedules | `fetch_schedules.py` | includes `spread_line`, `total_line`, moneylines, rest days, temp/wind |
| Weather | `fetch_weather.py` | from nflverse schedules (pro-football-reference blocks scrapers). Only populated close to game time; domes are null. |
| Vegas lines | `fetch_vegas_lines.py` | **dead after 2020**: `import_sc_lines` returns zero rows for 2021+ |
| Injury reports | `fetch_injury_reports.py` | |
| Play-by-play | `fetch_play_by_play.py` | |
| NGS | `fetch_ngs.py` | 2016+ only |
| Depth charts | `fetch_depth_charts.py` | each week gets the last snapshot published before that week's first kickoff; the upcoming week gets the newest snapshot. Encodes rank as 1->1.0, 2->0.5, 3->0.2, deeper->0.1, unknown->0.7. `prediction.py` re-fetches the 2026 file if older than 12h. |
| O-line rankings | `fetch_oline_rankings_weekly.py` | derived from play-by-play (sacks, QB hits, stuffed runs, etc.) |
| Strength of schedule | `fetch_strength_of_schedule.py` | win pct from schedules; `preseason_sos` and `adjusted_sos` |
| Team coaching | `fetch_team_coaching.py` | `import_team_desc` is static, no season dimension |
| Offensive coordinators | `fetch_offensive_coordinators.py` and `_2026.py` | no nflverse coaching endpoint exists, so 2026 coordinator **names are carried forward from 2025** and will be wrong wherever a team changed OCs. Tendency columns (pass/rush pct, target shares) are real 2026 numbers once games are played; the `source` column says which is which. |

Team abbreviation normalization (`LAR/STL->LA`, `OAK->LV`, `SD->LAC`, `WSH->WAS`, etc.) is **duplicated** in `prediction.py`, `fetch_depth_charts.py` and (in a different form) `layer1_base_join.py`.

## 5. Processing pipeline

Run order, each step reads the previous step's parquet:

```
fetchers -> processing/layer1_base_join.py -> data/processed/base_processed.parquet
         -> processing/layer2_team_aggs.py -> data/processed/team_aggs.parquet
         -> processing/layer3_player_features.py -> data/processed/final_training_data.parquet
         -> model/train.py -> model/*.pkl + model_metadata.json
```

`layer1_base_join.py` does `from pbp_box_scores import ...`, so run it as a script (the script directory goes on `sys.path`), not as a module from the repo root.

### Layer 1: base join
- The **schedule is the spine**. It is expanded to one row per team-game (home and away copies), then to player-weeks.
- Left-joins: box scores, play-by-play aggregates, NGS (passing/rushing/receiving, all three as columns, prefixed `ngs_<type>_`, sparse where irrelevant), injury reports, bio data, offensive coordinators, o-line rankings, strength of schedule, weather, Vegas lines.
- Regular season only (`game_type == "REG"`, `season_type == "REG"`).
- `dnp_flag = 1` if injury status is `Out` or `IR`. Box stat columns are zero-filled for active players. NGS is zero-filled from 2016 and left NaN before that (the era does not exist).
- If a season has no box score CSV, `fill_missing_seasons_from_pbp` rebuilds the stat line from play-by-play via `pbp_box_scores.py` (validated against 2024: PPR matches exactly on 99.4% of player-weeks).
- Output is about 400k rows x 100+ columns. There is a verification report printed at the end.

### Layer 2: team aggregates (per season x team)
- `def_strength_vs_rb` = RB rush yards / RB carries allowed. `def_strength_vs_wr` / `_te` = receiving yards / targets allowed. `def_strength_vs_qb` = (sacks + interceptions) / games.
- `team_target_share_to_{wr,te,rb}` and `team_rush_share_to_rb`.
- `preseason_sos`, `adjusted_sos`.
- **These are season-level averages applied uniformly to every week** (see known issues).

### Layer 3: player features
Built in this order: matchup features (`opp_def_strength_for_position`, `matchup_advantage_score`), `usage_trend` / `target_share_trend`, rushing features, receiving features, depth chart score, `player_target_share_of_team`, seasonal context (`weeks_into_season`, `games_played_this_szn`, `bye_week_passed`), injury recovery (`weeks_since_injury`, `prev_injuries_this_szn`), `rest_days`.

Key design rule: **every in-season number uses only weeks strictly before the target week**, and is shrunk toward the player's prior season with weight `n / (n + k)`, where n is games played. The k constants were fit on 2016-2025 data:
- `RUSH_VOLUME_SHRINKAGE_GAMES = 2` (carries/game, carry share), `RUSH_YARDS_SHRINKAGE_GAMES = 3`
- `REC_VOLUME_SHRINKAGE_GAMES = 4`, `TEAM_PASS_SHRINKAGE_GAMES = 6`, per-rate constants in `REC_RATE_SHRINKAGE_GAMES`
- `MIN_CARRIES_FOR_YPC = 10`, `MIN_TARGETS_FOR_RATE = 10`, `RECENT_GAMES = 3`

`prediction.py` imports these constants from Layer 3 so training-time and prediction-time features stay in sync. **If you change a shrinkage constant or a feature definition in Layer 3, `SeasonForm` in `prediction.py` must change to match, and the models must be retrained.**

## 6. Models (`model/train.py`)

- **Split by season**: train 2010-2023, validation 2024 (passed as `eval_set` only; there is no early stopping, `n_estimators` is fixed), test 2025.
- **Algorithm**: `XGBRegressor`, `max_depth=6, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8`, `random_state=42`, squared-error objective. Predictions clipped at 0.
- **Features** are everything numeric that is not an outcome or ID. Exclusion is by prefix (`passing_`, `rushing_`, `receiving_`, `sack`, `ngs_`, `pbp_`, `fantasy_points`) plus named sets (`OUTCOME_COLS`, `ID_COLS`). A leakage review flags any retained feature with |corr| >= 0.5 with PPR (currently none).
- **Imputation**: drop columns >80% null; zero-fill columns 20-80% null; median-fill the rest. Then `StandardScaler`. Scaler mean/scale are stored in the metadata and reapplied in `prediction.py`.
- **Targets per position**:
  - QB: passing_attempts, passing_yards, passing_tds, interceptions, rushing_yards, rushing_tds
  - RB: carries, rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds
  - WR and TE: receptions, receiving_yards, receiving_tds
  - That is 19 models, saved as `model/<stat>_<position>_model.pkl`.
- **PPR is reconstructed** from predicted stats: rec 1.0, rec yd 0.1, rec TD 6, rush yd 0.1, rush TD 6, pass yd 0.04, pass TD 4, INT -2. Fumbles and 2-pt conversions are not modeled.
- QB and RB models exclude eight receiving-form features (`RECEIVING_FORM_FEATURES`), because ablation over four seeds showed they helped WR/TE (MAE down about 0.04) but not QB/RB.
- Feature counts: QB 56, RB 57, WR 64, TE 64.
- `--model-dir` trains a candidate into a scratch folder without overwriting live models; `--drop-features` supports ablations; `prediction.py --model-dir` can score a candidate.
- `train.py` also deletes a legacy `trained_model.pkl` if present (an earlier leaked model).

### Current metrics (2025 test season, from `model_metadata.json`)

PPR end-to-end: **MAE 4.36, RMSE 5.99, R2 0.42** overall. By position: QB MAE 6.14 / R2 0.39, RB 4.45 / 0.46, WR 4.21 / 0.36, TE 3.45 / 0.35. The "formula floor" (actual stats scored with the same weights) is MAE 0.18, so the gap is entirely stat-prediction error.

Per-stat R2 is uneven. Volume stats are decent (QB attempts 0.52, RB carries 0.58). Touchdown and interception models are close to useless: QB rushing TDs 0.004, RB receiving TDs 0.02, WR/TE receiving TDs about 0.05, INT 0.07. TDs are mostly noise at this level, so don't expect big wins there.

Feature importance is dominated by `depth_chart_position` for QB attempts/yards/TDs (0.20-0.37), then `weeks_since_injury`, `prev_injuries_this_szn`, and (for RBs) `carries_last3`. A QB model leaning that hard on a depth-chart label is worth being suspicious of.

## 7. Projection engine (`prediction.py`)

CLI: `python prediction.py "Player Name" [--week N] [--no-refresh] [--model-dir DIR]`. Also imported as a library by `batch_compute.py` (it calls `predict()` directly).

Flow of `predict(name, week)`:
1. Load metadata, 2026 bio and schedule. `resolve_week` picks the next unplayed week unless `--week` is given (past weeks work as backtests).
2. `find_player` -> position (must be QB/RB/WR/TE) -> team -> `find_game` (raises a bye-week error if none).
3. **Baseline feature vector** = the player's last non-DNP row from the **2025** training data (`player_context_row`). Rookies get the 2025 position median.
4. `SeasonForm(target_week, schedule)` loads 2026 data strictly before the target week: box scores, injuries (Out/IR only), depth chart, o-line ranks, bye weeks.
5. `build_overrides` replaces baseline features with fresher 2026 values: game context (spread, total, implied totals, weather with defaults of 70F / 5mph, rest days), games played, bye passed, injury history, shrunk usage and target-share trends, rushing and receiving form, o-line rank (blended with the 2025 average), bio fields, coordinator and SOS rows, defensive strength.
6. `assemble_vector`: any feature still missing falls back to the scaler mean, and the count is reported in the output.
7. **Role blending for WR and RB**: rather than editing `depth_chart_position` or `bio_draft_number`, it averages model predictions over alternative values (`expand_vector` + `mixture_predict`) because trees split these features into steps.
   - Depth chart label weight = `0.25 + 0.75 * 1.5/(1.5+n)`, with the rest going to the role implied by 2026 usage rank.
   - Draft slot weight decays 0.25x per NFL season (rookie 100%, year 2 25%, year 3 6%); the remainder is spread over deciles of that position's draft slots.
8. **RB floor**: for rushing stats only, it also predicts with target share raised to the depth-tier median and trend floored at 1.0, and keeps the higher prediction (so receiving work can help but a thin target share can't sink carries).
9. Predicted stats are summed with the PPR weights; the CLI prints projection, stat line, and diagnostic notes.

Important constants are at the top of the file with long comments explaining how they were fit. Read those before changing anything.

### Known gaps in the engine
- **Defensive strength and matchup features are pinned to 2025** (`team_aggs.parquet` only built through 2025, `CONTEXT_SEASON`). Running Layers 1-3 with 2026 data would update them. `SEASON_FORM_NOTES` prints this on every projection.
- Coordinator **names** for 2026 are stale; only tendencies are fresh.
- `vegas_team_line` is NaN for every season after 2020. It is still a model feature, so it's effectively median/zero-filled noise. `sched_temp`/`sched_wind` and `weather_temp`/`weather_wind` are set to identical values at prediction time.
- **Discrepancy to check**: the `prediction.py` docstring says models are "trained through 2025", but `model_metadata.json` says `trained_on_seasons` is 2010-2023, validated 2024, tested 2025. The live models appear **not** to have seen 2024 or 2025 as training data. Confirm before assuming; retraining on 2010-2025 for the in-season run is an obvious next experiment (but then there is no clean holdout).

## 8. Serving

### Batch job (`batch_compute.py`)
```
python batch_compute.py                 # every week with no game kicked off
python batch_compute.py --weeks 5 6
python batch_compute.py --dry-run       # compute only, skip DB
python batch_compute.py --limit 20      # first N players, upserts only
python batch_compute.py --no-refresh    # skip depth chart refetch
```
- Players: 2026 rows of QB/RB/WR/TE in `final_training_data.parquet`.
- `BatchPredictor` runs the real `prediction.predict` unchanged, but monkeypatches its loaders with cached versions (via `unittest.mock`) and captures the stat line instead of printing it. This avoids about 1.7s of reloading per player. **If you refactor `predict()`'s internal function names (`load_json`, `load_csv`, `load_models`, `SeasonForm`, `find_player`, `find_game`, `format_detail`, `refresh_depth_chart`), you will silently break the batch patching.**
- The player is pinned by `player_id` (training and roster names differ for about a dozen players, e.g. Kenneth/Kenny Gainwell).
- Writes one transaction: creates the table if missing, deletes all rows for the projected weeks (unless `--limit`), then bulk upserts.
- Table `projections`, **primary key `(player_name, week)`**, with `player_id` stored but not part of the key. Two players sharing a name makes the script abort. Columns: player_name, player_id, position, team, week, opponent, projected_ppr, ten stat columns, `computed_at`.

### API (`api/`)
```
python -m uvicorn api.main:app --reload          # http://127.0.0.1:8000/docs
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```
Env: `DATABASE_URL_POOLED` or `DATABASE_URL`, `CORS_ORIGINS` (default `*`), `CORS_ORIGIN_REGEX`, `SEASON_WEEK1_THURSDAY` (default `2026-09-10`), `DB_CONNECT_TIMEOUT`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + DB round trip (wakes Neon) |
| `GET /weeks` | weeks with projections + `default_week` |
| `GET /players?position=&team=&q=` | all projected players, name substring search |
| `GET /projections/player?player_id=` or `?name=` (+ `position`, `team`, `week`) | one player-week; name lookups can 409 with candidates |
| `GET /projections/rankings?position=&week=&limit=` | top N by projected PPR (limit max 100, default 10) |

Notable behavior:
- Errors are always `{"error": {"code", "message", "details"}}`: 400 invalid input, 404 unknown player / week, 409 ambiguous name, 503 DB unreachable (with `Retry-After: 5`), 500 unexpected.
- Default week is computed from the calendar (week N starts the Thursday N-1 weeks after week 1's Thursday, using a UTC-5 offset), not from the schedule file, because the deployed API doesn't have the data directory. **That fixed UTC-5 offset ignores daylight saving**, so it is off by an hour during CST months. Probably harmless for a day-granularity calculation, but worth knowing.
- Name resolution (`names.py`) goes in tiers: exact normalized name -> nickname (same last name, first names share first 3 letters) -> last name (single-word queries). Suffixes (Jr, III...) and punctuation are ignored; difflib suggestions on a miss.
- `db.py` handles Neon cold starts: `pool_pre_ping`, `pool_recycle=240`, and three retries (0.5s, 1.5s, 3s) on connection-level errors only before raising `DatabaseUnavailable`.
- Read-only: CORS allows `GET` only. Rankings and players queries load into Python and filter there for `/players` (fine at this size).

Deployment as understood from the owner: Postgres on **Neon**, frontend on **Vercel**. The API host isn't named anywhere in the repo (it reads `$PORT`) **(unverified where it is hosted)**.

## 9. Frontend (`frontend/`)

Next.js App Router, React 19, TypeScript, **no component library**.

**Read `frontend/AGENTS.md` first.** It states this Next.js version has breaking changes from what most models know, and says to read the guides in `frontend/node_modules/next/dist/docs/` before writing code (run `npm install` in `frontend/` first). `frontend/CLAUDE.md` just includes it with `@AGENTS.md`, so Codex picks up the same rules through `AGENTS.md`.

Structure:
- `app/layout.tsx`: shell with `Sidebar` + main content.
- `app/page.tsx` -> `app/LookupView.tsx`: player lookup. The URL (`?player_id=` or `?name=`, plus `?week=`) is the source of truth so rankings can deep-link here.
- `app/rankings/page.tsx` -> `RankingsView.tsx`: top-N by position and week.
- `components/`: `PlayerSearch`, `ProjectionCard`, `RankingsTable`, `Sidebar`, `StatusMessage`, `WeekSelect`.
- `lib/api.ts`: **every API call lives here**; typed response shapes, `ApiError` with `kind` (`ambiguous`, `player_not_found`, `week_not_found`, `unavailable`, `other`), abort handling. A response-shape change should only touch this file (and the TypeScript types must be kept in sync with the Pydantic models in `api/main.py` by hand).
- `lib/useRequest.ts`: small fetch hook with abort-on-change and `retry`. `lib/params.ts`: `parseWeek`.
- Config: `NEXT_PUBLIC_API_URL` (defaults to `http://localhost:8000`); copy `.env.local.example` to `.env.local`.
- Scripts: `npm run dev | build | start`.

**(unverified)**: I did not read `globals.css`, `RankingsView.tsx`, or the individual components in full. The owner intends to restyle the frontend later and hand designs over for implementation; MVP styling is meant to be polished but is not final.

## 10. Roadmap and status

Where the project stands: models trained and validated, API and frontend built and wired, 2026 season in progress (weeks 1-3 played as of the last notes). Immediate work per the owner: finish deploying/validating the website, then start internship applications.

Planned / wished-for (from `notes.md` and `limitation.md`):
- `/compare` endpoint and start/sit features (post-MVP).
- Rolling per-week defensive stats in Layer 2 (use weeks 1..N-1 when predicting week N) plus defensive trend features. `limitation.md` estimates 5-10% accuracy gain; treat that figure as the author's guess, not measured.
- Defensive-coordinator tenure feature (weeks since DC started) and DC x week-of-season interactions (author's guess: 3-7%).
- **Boom/bust distribution**: prediction intervals from training residuals per position/stat, floor/ceiling at 10th/90th percentiles, probability of user-defined boom/bust thresholds. Marked medium-high priority for v2.
- Offensive-coordinator success tracking and playcalling-pattern detection; per-week o-line rankings feeding QB/offense models.
- Possible LSTM v2 after the XGBoost baseline (earlier plan).
- Live weekly refresh cycle: re-fetch with `--years 2026`, rerun Layers 1-3, rerun batch, retrain periodically.

### Known issues (from `notes.md`, unresolved as written)
- Backup RBs can be projected unrealistically high (example: a backup to a star RB at 17 points; the logic only makes sense if the starter is out). The depth chart / usage-role blend and RB floor were partly aimed at this; `notes.md` still lists it.
- TE projections are described by the owner as bad ("tight end formula is fucked").
- QBs project a bit too conservatively.
- WR hierarchies sometimes look wrong (a WR3 projected above WR1/WR2 on one team).
- A "DNP issue" is listed with no details **(unverified what it refers to)**.
- `notes.md` is a scratchpad: some of its issue bullets may already be addressed by later commits (the repo has only one commit, so history won't help). Verify against current behavior before acting on any of them.

## 11. Conventions and gotchas for editing

- **Leakage discipline is central.** Anything that feeds a feature must only use data from before kickoff of the target week. When adding features, mirror the logic in both Layer 3 (training) and `prediction.py` (inference), and re-run the leakage review in `train.py`.
- **Training/serving parity**: feature names, order and scaler params come from `model_metadata.json`. Column name oddities like `*_x` / `*_y` (`adjusted_sos_x`, `team_target_share_to_wr_y`) come from pandas merge suffixes in Layer 1/2; models were trained on those exact names, so don't rename them without retraining. `prediction.py` deliberately writes to `_x`, `_y` and unsuffixed variants.
- Retraining overwrites `model/*.pkl` and `model_metadata.json`. Snapshots in `model/archive/` are the manual versioning convention (dated folders). Use `--model-dir` for candidates.
- Pickled XGBoost models are tied to the xgboost version they were trained with. Keep the version consistent or retrain.
- Play-by-play and some feeds change schema between seasons; fetchers contain fallbacks (box scores) and Layer 1 prints `[flag]` / `[missing]` messages instead of failing. Read those logs.
- Dev machine is Windows; keep scripts path-safe and avoid shell-specific assumptions.
- No tests means verification is done by running the pipeline and reading the verification reports each layer prints, plus `python prediction.py "<name>" --week N` for spot checks and backtests on completed weeks. If you add a change, prefer adding a small test or a printed check rather than relying on eyeballing.
- Git: single branch history, one commit so far. Don't commit data, parquet, `.env`, or `.venv`.

## 12. Quick-start commands

```bash
# Python side
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy pyarrow scikit-learn xgboost nfl_data_py nflreadpy sqlalchemy psycopg2-binary python-dotenv
pip install -r requirements-api.txt

# Data -> features -> models
python data_fetching/fetch_schedules.py            # ...and the other fetchers
python processing/layer1_base_join.py
python processing/layer2_team_aggs.py
python processing/layer3_player_features.py
python model/train.py

# Spot check, batch, serve
python prediction.py "Ja'Marr Chase" --week 5
python batch_compute.py --dry-run
python -m uvicorn api.main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```
