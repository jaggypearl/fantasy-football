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
MODEL_PATH = MODEL_DIR / "trained_model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TARGET = "fantasy_points_ppr"
TRAIN_SEASONS = list(range(2010, 2024))
VAL_SEASON = 2024
TEST_SEASON = 2025

# Alternative target and row identifiers - never patterns the model should learn.
EXCLUDED_COLS = ["fantasy_points_ppr", "fantasy_points", "player_name", "game_id"]

HYPERPARAMS = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

HIGH_NAN_DROP = 0.80    # above this share of nulls a column carries no learnable pattern
ZERO_FILL_FLOOR = 0.20  # 20-80% null means "missing context"; 0 encodes that meaningfully


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=["number", "bool"]).columns
    return [c for c in numeric if c not in EXCLUDED_COLS]


def split_by_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] == TEST_SEASON]
    return train, val, test


def build_imputation_plan(train_features: pd.DataFrame) -> tuple[list[str], list[str], dict[str, float]]:
    """Decide per column how to handle nulls, reading only the training seasons.

    Null rates and medians come from training rows alone so no test-season
    information reaches the fitted pipeline.
    """
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
    # A column can be null in val/test without being null in train; 0 reuses the
    # same "no context" encoding.
    return out.fillna(0.0)


def main() -> None:
    print("=" * 80)
    print("XGBOOST FANTASY POINTS MODEL - TRAINING")
    print("=" * 80)

    df = pd.read_parquet(DATA_PATH)
    print(f"\nLoaded {DATA_PATH.name}: {len(df):,} rows x {len(df.columns)} columns")

    feature_cols = select_feature_columns(df)
    print(f"Candidate numeric features: {len(feature_cols)}")

    train_df, val_df, test_df = split_by_season(df)
    print("\nTemporal split (no season appears in more than one split):")
    print(f"  TRAIN {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}: {len(train_df):>8,} rows")
    print(f"  VAL   {VAL_SEASON}:      {len(val_df):>8,} rows")
    print(f"  TEST  {TEST_SEASON}:      {len(test_df):>8,} rows")

    drop_cols, zero_fill_cols, medians = build_imputation_plan(train_df[feature_cols])
    kept_cols = [c for c in feature_cols if c not in drop_cols]
    zero_fill_kept = [c for c in zero_fill_cols if c in kept_cols]
    medians_kept = {c: v for c, v in medians.items() if c in kept_cols}

    print("\nNull handling (rates and medians measured on training seasons only):")
    print(f"  dropped (>{HIGH_NAN_DROP:.0%} null):                 {len(drop_cols)} -> {drop_cols}")
    print(f"  filled with 0 ({ZERO_FILL_FLOOR:.0%}-{HIGH_NAN_DROP:.0%} null):        {len(zero_fill_kept)}")
    print(f"  filled with train median (<{ZERO_FILL_FLOOR:.0%} null): {len(medians_kept)}")
    print(f"  features used:                        {len(kept_cols)}")

    X_train = apply_imputation(train_df[kept_cols], zero_fill_kept, medians_kept)
    X_val = apply_imputation(val_df[kept_cols], zero_fill_kept, medians_kept)
    X_test = apply_imputation(test_df[kept_cols], zero_fill_kept, medians_kept)

    for name, frame in (("train", X_train), ("val", X_val), ("test", X_test)):
        remaining = int(frame.isna().sum().sum())
        if remaining:
            raise ValueError(f"{remaining} NaN values remain in {name} after imputation")
    print("  verified: 0 NaN values remain in train/val/test")

    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()
    y_test = test_df[TARGET].to_numpy()

    # Fit on train only; val and test are transformed with the training statistics.
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("  scaler fit on training rows only, then applied to val and test")

    model = XGBRegressor(
        **HYPERPARAMS,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    print(f"\nTraining {HYPERPARAMS['n_estimators']} trees (validation RMSE every 10)...")
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=10)

    y_test_pred = model.predict(X_test_scaled)
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(root_mean_squared_error(y_test, y_test_pred))
    test_r2 = float(r2_score(y_test, y_test_pred))
    train_mae = float(mean_absolute_error(y_train, model.predict(X_train_scaled)))

    importance = (
        pd.DataFrame({"feature": kept_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "-" * 80)
    print("TOP 20 FEATURES BY IMPORTANCE")
    print("-" * 80)
    for rank, row in enumerate(importance.head(20).itertuples(index=False), start=1):
        print(f"  {rank:2d}. {row.feature:<45} {row.importance:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(model, fh)

    metadata = {
        "features": kept_cols,
        "feature_importance": importance.head(20).to_dict(orient="records"),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "trained_on_seasons": TRAIN_SEASONS,
        "validated_on": VAL_SEASON,
        "tested_on": TEST_SEASON,
        "model_type": "XGBoost",
        "hyperparameters": HYPERPARAMS,
        "imputation": {
            "dropped_columns": drop_cols,
            "zero_filled_columns": zero_fill_kept,
            "median_fill_values": medians_kept,
        },
    }
    with METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"  Train rows ({TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}): {len(train_df):>8,}")
    print(f"  Val rows   ({VAL_SEASON}):        {len(val_df):>8,}")
    print(f"  Test rows  ({TEST_SEASON}):        {len(test_df):>8,}")
    print(f"  Features used:             {len(kept_cols):>8}")
    print()
    print(f"  Test MAE:   {test_mae:.4f}  (avg fantasy points off per player-week)")
    print(f"  Test RMSE:  {test_rmse:.4f}")
    print(f"  Test R2:    {test_r2:.4f}  (share of scoring variance explained)")
    print(f"  Train MAE:  {train_mae:.4f}  (compare with test MAE to gauge overfitting)")
    print()
    print("  Top 3 features:")
    for rank, row in enumerate(importance.head(3).itertuples(index=False), start=1):
        print(f"    {rank}. {row.feature} ({row.importance:.4f})")
    print()
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Metadata: {METADATA_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
