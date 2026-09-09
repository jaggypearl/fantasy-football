import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "final_training_data.parquet"
MODEL_DIR = PROJECT_DIR / "model"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
LEGACY_MODEL_PATH = MODEL_DIR / "trained_model.pkl"

PPR_COL = "fantasy_points_ppr"
TRAIN_SEASONS = list(range(2010, 2024))
VAL_SEASON = 2024
TEST_SEASON = 2025

STAT_COLUMNS = {
    "passing_attempts": "attempts",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "interceptions": "interceptions",
    "carries": "carries",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
}

POSITION_TARGETS = {
    "QB": ["passing_attempts", "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"],
    "WR": ["receptions", "receiving_yards", "receiving_tds"],
    "TE": ["receptions", "receiving_yards", "receiving_tds"],
}

PPR_WEIGHTS = {
    "receptions": 1.0,
    "receiving_yards": 0.1,
    "receiving_tds": 6.0,
    "rushing_yards": 0.1,
    "rushing_tds": 6.0,
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -2.0,
}

OUTCOME_PREFIXES = (
    "passing_",
    "rushing_",
    "receiving_",
    "sack",
    "ngs_",
    "pbp_",
    "fantasy_points",
)

OUTCOME_COLS = {
    "completions",
    "attempts",
    "interceptions",
    "carries",
    "receptions",
    "targets",
    "target_share",
    "air_yards_share",
    "wopr",
    "racr",
    "pacr",
    "dakota",
    "special_teams_tds",
    "fumbles",
    "snap_count",
    "snap_pct",
    "routes_run",
    "dnp_flag",
    "away_score",
    "home_score",
    "result",
    "total",
    "overtime",
}

ID_COLS = {
    "season",
    "old_game_id",
    "gsis",
    "pff",
    "espn",
    "ftn",
    "bio_espn_id",
    "bio_yahoo_id",
    "bio_rotowire_id",
    "bio_pff_id",
    "bio_fantasy_data_id",
    "bio_sleeper_id",
    "bio_gsis_it_id",
    "bio_jersey_number",
    "player_name",
    "game_id",
}

HYPERPARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

HIGH_NAN_DROP = 0.80
ZERO_FILL_FLOOR = 0.20
REVIEW_CORR = 0.50


def is_outcome_column(col: str) -> bool:
    return col.startswith(OUTCOME_PREFIXES) or col in OUTCOME_COLS


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    excluded = [c for c in numeric if is_outcome_column(c) or c in ID_COLS]
    kept = [c for c in numeric if c not in excluded]
    return kept, excluded


def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] == TEST_SEASON]
    return train, val, test


def build_imputation_plan(train_features: pd.DataFrame) -> tuple[list[str], list[str], dict[str, float]]:
    nan_frac = train_features.isna().mean()
    drop_cols = nan_frac[nan_frac > HIGH_NAN_DROP].index.tolist()
    zero_fill_cols = nan_frac[(nan_frac >= ZERO_FILL_FLOOR) & (nan_frac <= HIGH_NAN_DROP)].index.tolist()
    median_cols = nan_frac[(nan_frac > 0) & (nan_frac < ZERO_FILL_FLOOR)].index.tolist()
    medians = {}
    for col in median_cols:
        value = float(train_features[col].median())
        medians[col] = value if np.isfinite(value) else 0.0
    return drop_cols, zero_fill_cols, medians


def apply_imputation(features: pd.DataFrame, zero_fill_cols: list[str], medians: dict[str, float]) -> pd.DataFrame:
    out = features.astype("float64").copy()
    if zero_fill_cols:
        out[zero_fill_cols] = out[zero_fill_cols].fillna(0.0)
    out = out.fillna(medians)
    return out.fillna(0.0)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "is_home" in out.columns and out["is_home"].dtype == object:
        mapping = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, 1: 1.0, 0: 0.0}
        out["is_home"] = out["is_home"].map(mapping).astype("float64")
    return out


def review_residual_leakage(train_df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    target = train_df[PPR_COL]
    flagged = []
    for col in feature_cols:
        series = train_df[col]
        if series.notna().sum() < 100 or series.nunique(dropna=True) < 2:
            continue
        corr = series.corr(target)
        if pd.notna(corr) and abs(corr) >= REVIEW_CORR:
            flagged.append({"feature": col, "corr_with_ppr": float(corr)})
    return sorted(flagged, key=lambda r: abs(r["corr_with_ppr"]), reverse=True)


def usable_target(series: pd.Series) -> bool:
    filled = series.fillna(0.0)
    return bool(filled.abs().sum() > 0 and filled.nunique() > 1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def ppr_from_stats(stats: dict[str, np.ndarray], n_rows: int) -> np.ndarray:
    total = np.zeros(n_rows, dtype="float64")
    for stat, weight in PPR_WEIGHTS.items():
        if stat in stats:
            total += np.asarray(stats[stat], dtype="float64") * weight
    return total


def train_position(df: pd.DataFrame, position: str, feature_cols: list[str]) -> tuple[dict, dict]:
    subset = df[df["bio_position"] == position]
    train_df, val_df, test_df = split_by_season(subset)

    print("\n" + "=" * 80)
    print(f"POSITION: {position}")
    print("=" * 80)
    print(f"  rows  TRAIN {len(train_df):>7,}   VAL {len(val_df):>6,}   TEST {len(test_df):>6,}")

    if len(train_df) < 200 or len(test_df) == 0:
        print("  skipped: not enough rows for a temporal split")
        return {}, {}

    drop_cols, zero_fill_cols, medians = build_imputation_plan(train_df[feature_cols])
    kept_cols = [c for c in feature_cols if c not in drop_cols]
    kept_cols = [c for c in kept_cols if train_df[c].nunique(dropna=True) > 1]
    zero_fill_kept = [c for c in zero_fill_cols if c in kept_cols]
    medians_kept = {c: v for c, v in medians.items() if c in kept_cols}

    print(f"  null handling: dropped {len(drop_cols)} (>{HIGH_NAN_DROP:.0%} null), "
          f"zero-filled {len(zero_fill_kept)}, median-filled {len(medians_kept)}")
    print(f"  features used: {len(kept_cols)}")

    X_train = apply_imputation(train_df[kept_cols], zero_fill_kept, medians_kept)
    X_val = apply_imputation(val_df[kept_cols], zero_fill_kept, medians_kept)
    X_test = apply_imputation(test_df[kept_cols], zero_fill_kept, medians_kept)

    for name, frame in (("train", X_train), ("val", X_val), ("test", X_test)):
        remaining = int(frame.isna().sum().sum())
        if remaining:
            raise ValueError(f"{position}: {remaining} NaN values remain in {name} after imputation")

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    stat_meta: dict[str, dict] = {}
    predicted_stats: dict[str, np.ndarray] = {}
    actual_stats: dict[str, np.ndarray] = {}

    for stat in POSITION_TARGETS[position]:
        column = STAT_COLUMNS[stat]
        if column not in df.columns:
            print(f"  - {stat}: skipped, column '{column}' not present")
            continue
        if not usable_target(train_df[column]) or not usable_target(test_df[column]):
            print(f"  - {stat}: skipped, all zero/NaN for {position}")
            continue

        y_train = train_df[column].fillna(0.0).to_numpy()
        y_val = val_df[column].fillna(0.0).to_numpy()
        y_test = test_df[column].fillna(0.0).to_numpy()

        model = XGBRegressor(**HYPERPARAMS, random_state=42, n_jobs=-1, objective="reg:squarederror")
        eval_set = [(X_val_scaled, y_val)] if len(val_df) else None
        model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)

        y_pred = np.clip(model.predict(X_test_scaled), 0.0, None)
        stat_metrics = metrics(y_test, y_pred)

        importance = (
            pd.DataFrame({"feature": kept_cols, "importance": model.feature_importances_.astype("float64")})
            .sort_values("importance", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )

        print(f"  - {stat:<18} MAE {stat_metrics['mae']:>7.3f}   RMSE {stat_metrics['rmse']:>7.3f}   "
              f"R2 {stat_metrics['r2']:>7.4f}")
        for rank, row in enumerate(importance.itertuples(index=False), start=1):
            print(f"        {rank:2d}. {row.feature:<45} {row.importance:.4f}")

        model_path = MODEL_DIR / f"{stat}_{position.lower()}_model.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(model, fh)

        predicted_stats[stat] = y_pred
        actual_stats[stat] = y_test
        stat_meta[stat] = {
            "source_column": column,
            "features": kept_cols,
            "feature_importance": importance.to_dict(orient="records"),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "test_mae": stat_metrics["mae"],
            "test_rmse": stat_metrics["rmse"],
            "test_r2": stat_metrics["r2"],
            "model_path": model_path.name,
        }

    ppr_info = {
        "predicted": ppr_from_stats(predicted_stats, len(test_df)),
        "oracle": ppr_from_stats(actual_stats, len(test_df)),
        "actual": test_df[PPR_COL].fillna(0.0).to_numpy(),
    }
    return stat_meta, ppr_info


def main() -> None:
    print("=" * 80)
    print("POSITION-SPECIFIC STAT MODELS -> PPR RECONSTRUCTION")
    print("=" * 80)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if LEGACY_MODEL_PATH.exists():
        LEGACY_MODEL_PATH.unlink()
        print(f"\nRemoved leaked legacy model: {LEGACY_MODEL_PATH.name}")

    df = pd.read_parquet(DATA_PATH)
    df = coerce_numeric(df)
    print(f"Loaded {DATA_PATH.name}: {len(df):,} rows x {len(df.columns)} columns")

    feature_cols, excluded = select_feature_columns(df)
    print(f"\nExcluded {len(excluded)} numeric columns as game outcomes or identifiers")
    print(f"Pre-game feature candidates: {len(feature_cols)}")
    for col in feature_cols:
        print(f"    {col}")

    train_all, _, _ = split_by_season(df)
    flagged = review_residual_leakage(train_all, feature_cols)
    print(f"\nLeakage review (|corr| with {PPR_COL} >= {REVIEW_CORR} on training rows):")
    if flagged:
        for row in flagged:
            print(f"    REVIEW {row['feature']:<45} corr {row['corr_with_ppr']:+.4f}")
    else:
        print("    none - no retained feature correlates at or above the threshold with PPR")

    positions_meta: dict[str, dict] = {}
    ppr_by_position: dict[str, dict] = {}
    all_pred: list[np.ndarray] = []
    all_actual: list[np.ndarray] = []
    all_oracle: list[np.ndarray] = []

    for position in POSITION_TARGETS:
        stat_meta, ppr_info = train_position(df, position, feature_cols)
        if not stat_meta:
            continue
        positions_meta[position] = stat_meta
        pos_metrics = metrics(ppr_info["actual"], ppr_info["predicted"])
        oracle_metrics = metrics(ppr_info["actual"], ppr_info["oracle"])
        ppr_by_position[position] = pos_metrics
        all_pred.append(ppr_info["predicted"])
        all_actual.append(ppr_info["actual"])
        all_oracle.append(ppr_info["oracle"])
        print(f"  PPR reconstruction ({TEST_SEASON}): MAE {pos_metrics['mae']:.3f}   "
              f"RMSE {pos_metrics['rmse']:.3f}   R2 {pos_metrics['r2']:.4f}")
        print(f"  formula floor (actual stats -> PPR): MAE {oracle_metrics['mae']:.3f}   "
              f"R2 {oracle_metrics['r2']:.4f}")

    if not all_pred:
        raise RuntimeError("no position models were trained")

    overall_pred = np.concatenate(all_pred)
    overall_actual = np.concatenate(all_actual)
    overall_oracle = np.concatenate(all_oracle)
    overall_metrics = metrics(overall_actual, overall_pred)
    overall_oracle_metrics = metrics(overall_actual, overall_oracle)

    metadata = {
        "positions": positions_meta,
        "ppr_reconstruction": {
            "overall": overall_metrics,
            "by_position": ppr_by_position,
            "formula_floor_overall": overall_oracle_metrics,
        },
        "trained_on_seasons": TRAIN_SEASONS,
        "validated_on": VAL_SEASON,
        "tested_on": TEST_SEASON,
        "model_type": "XGBoost per position per stat",
        "ppr_weights": PPR_WEIGHTS,
        "hyperparameters": HYPERPARAMS,
        "excluded_columns": excluded,
        "leakage_review": flagged,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for position, stat_meta in positions_meta.items():
        print(f"\n{position}")
        for stat, info in stat_meta.items():
            top3 = ", ".join(r["feature"] for r in info["feature_importance"][:3])
            print(f"  {stat:<18} MAE {info['test_mae']:>7.3f}  RMSE {info['test_rmse']:>7.3f}  "
                  f"R2 {info['test_r2']:>7.4f}")
            print(f"  {'':<18} top3: {top3}")
        pos = ppr_by_position[position]
        print(f"  {'PPR end-to-end':<18} MAE {pos['mae']:>7.3f}  RMSE {pos['rmse']:>7.3f}  R2 {pos['r2']:>7.4f}")

    print("\n" + "-" * 80)
    print(f"PPR RECONSTRUCTION, ALL POSITIONS, {TEST_SEASON} TEST SEASON")
    print("-" * 80)
    print(f"  MAE  {overall_metrics['mae']:.4f}")
    print(f"  RMSE {overall_metrics['rmse']:.4f}")
    print(f"  R2   {overall_metrics['r2']:.4f}")
    print(f"  formula floor (actual stats scored by the same weights): "
          f"MAE {overall_oracle_metrics['mae']:.4f}, R2 {overall_oracle_metrics['r2']:.4f}")
    print()
    print("  Features are pre-game context only. No box-score column reaches a model, so these")
    print("  numbers measure forward prediction rather than recovery of the scoring formula.")
    if overall_metrics["mae"] < 2.0:
        print("  FLAG: MAE below 2.0 is suspiciously low for pre-game prediction. Re-check the")
        print("        feature list above for a column that encodes in-game outcomes.")
    print(f"\n  Metadata: {METADATA_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
