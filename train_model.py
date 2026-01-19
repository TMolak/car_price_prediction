import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


USE_LOG_TARGET = True
TEST_SIZE = 0.20
VALID_SIZE_FROM_TRAIN = 0.20

DROP_COLS = {"Index"}


def fmt_pln(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main(
    data_path: str = "data/Car_sale_ads_cleaned_v2.csv",
    model_dir: str = "models",
    target: str = "Price",
):
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Brak kolumny '{target}' w danych. Dostępne: {df.columns.tolist()}")

    df = df[df[target].notna()].copy()

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[target]).copy()
    y = df[target].astype(float).copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Brak danych")
        X[c] = X[c].replace({"nan": "Brak danych", "None": "Brak danych"})

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

    # SPLIT: train / valid / test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALID_SIZE_FROM_TRAIN,
        random_state=42
    )

    # LOG TARGET
    if USE_LOG_TARGET:
        y_train_fit = np.log1p(y_train)
        y_valid_fit = np.log1p(y_valid)
    else:
        y_train_fit = y_train
        y_valid_fit = y_valid

    train_pool = Pool(X_train, y_train_fit, cat_features=cat_feature_indices)
    valid_pool = Pool(X_valid, y_valid_fit, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, cat_features=cat_feature_indices)

    # MODEL
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=10000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6,
        random_strength=1.0,
        bagging_temperature=0.5,
        rsm=0.9,
        od_type="Iter",
        od_wait=300,
        random_state=42,
        verbose=200,
        allow_writing_files=False,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    best_it = model.get_best_iteration()

    # EWALUACJA NA TEST
    pred_fit = model.predict(test_pool)

    if USE_LOG_TARGET:
        y_pred = np.expm1(pred_fit)
    else:
        y_pred = pred_fit

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse_val = rmse(y_test, y_pred)
    r2 = float(r2_score(y_test, y_pred))

    print("\n===== CatBoost - wyniki na zbiorze testowym =====")
    print(f"Best iteration: {best_it}")
    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse_val:.2f}")

    # feature importance
    importances = model.get_feature_importance(train_pool)
    fi_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    print("\nTop 20 najważniejszych cech:")
    print(fi_df.head(20).to_string(index=False))

    # ZAPIS
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(model_dir, "catboost_price.joblib")
    schema_path = os.path.join(model_dir, "feature_schema.joblib")

    joblib.dump(model, model_path)

    schema = {
        "feature_columns": X.columns.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_feature_indices": cat_feature_indices,
        "use_log_target": USE_LOG_TARGET,
        "drop_cols": sorted(list(DROP_COLS)),
        "best_iteration": int(best_it) if best_it is not None else None,
        "model_params": model.get_params(),
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse_val},
    }
    joblib.dump(schema, schema_path)

    print(f"\n[OK] Zapisano model:  {model_path}")
    print(f"[OK] Zapisano schema: {schema_path}")

    example_price = float(np.median(y_test))
    print(f"\nPrzykładowo medianowa cena w teście: {fmt_pln(example_price)} PLN")


if __name__ == "__main__":
    main()
