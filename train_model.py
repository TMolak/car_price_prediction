import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def main(
    data_path: str = "data/Car_sale_ads_cleaned_v2.csv",
    model_dir: str = "models",
    target: str = "Price",
):
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Brak kolumny '{target}' w danych. Dostępne: {df.columns.tolist()}")

    # tylko wiersze z targetem
    df = df[df[target].notna()].copy()

    # X / y
    X = df.drop(columns=[target])
    y = df[target].astype(float)

    # usuń Index jeśli jest
    if "Index" in X.columns:
        X = X.drop(columns=["Index"])

    # kategorie / numery
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ujednolicenie kategorycznych
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Brak danych")

    # (opcjonalnie) ujednolicenie numerycznych
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    cat_feature_indices = [X.columns.get_loc(c) for c in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        depth=10,
        learning_rate=0.05,
        iterations=500,
        loss_function="RMSE",
        random_state=42,
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
    )

    # ewaluacja
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print("\n===== CatBoost - wyniki na zbiorze testowym =====")
    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # feature importance
    importances = model.get_feature_importance()
    fi_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(
        by="importance", ascending=False
    )
    print("\nTop 20 najważniejszych cech:")
    print(fi_df.head(20).to_string(index=False))

    # ===== ZAPIS (joblib) =====
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    model_path = os.path.join(model_dir, "catboost_price.joblib")
    schema_path = os.path.join(model_dir, "feature_schema.joblib")

    joblib.dump(model, model_path)

    schema = {
        "feature_columns": X.columns.tolist(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        # do debugowania / sanity check:
        "metrics": {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)},
    }
    joblib.dump(schema, schema_path)

    print(f"\n[OK] Zapisano model:  {model_path}")
    print(f"[OK] Zapisano schema: {schema_path}")


if __name__ == "__main__":
    main()
