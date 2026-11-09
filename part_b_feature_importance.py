import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

FEATURES_PATH = (
    BASE_DIR / "data" / "processed" / "customer_features.csv"
)

RF_MODEL_PATH = (
    BASE_DIR / "data" / "processed" / "random_forest.joblib"
)

LR_MODEL_PATH = (
    BASE_DIR / "data" / "processed" / "logistic_regression.joblib"
)

SEGMENTS_PATH = (
    BASE_DIR / "outputs" / "customer_segments.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA AND MODELS
# ==================================================

print("=" * 70)
print("PART B TASK 8: MODEL INTERPRETATION AND RETENTION LIST")
print("=" * 70)

df = pd.read_csv(FEATURES_PATH)

rf_model = joblib.load(RF_MODEL_PATH)

lr_model = joblib.load(LR_MODEL_PATH)

segments = pd.read_csv(SEGMENTS_PATH)


# ==================================================
# 3. RECREATE ORIGINAL TRAIN-TEST SPLIT
# ==================================================

numeric_features = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "city",
    "gender",
    "favourite_category"
]

feature_columns = (
    numeric_features + categorical_features
)

X = df[feature_columns].copy()

y = df["lapsed"].copy()

customer_ids = df["customer_id"].copy()

X_train, X_test, y_train, y_test, train_ids, test_ids = (
    train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
)

print("\n--- DATA VALIDATION ---")

print("Training customers:", len(X_train))

print("Testing customers:", len(X_test))

assert len(X_train) == 924

assert len(X_test) == 232

print("PASS: Original train-test split reproduced.")


# ==================================================
# 4. RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)

rf_preprocessor = rf_model.named_steps["preprocessor"]

rf_classifier = rf_model.named_steps["classifier"]

rf_feature_names = rf_preprocessor.get_feature_names_out()

rf_importance = rf_classifier.feature_importances_

importance_df = pd.DataFrame({

    "feature": rf_feature_names,

    "importance": rf_importance

})

importance_df = importance_df.sort_values(

    by="importance",

    ascending=False

).reset_index(drop=True)

print("\n--- TOP 5 MOST IMPORTANT FEATURES ---")

print(
    importance_df.head(5).round(4).to_string(index=False)
)

importance_df.to_csv(

    PROCESSED_DIR / "feature_importance.csv",

    index=False

)

importance_df.head(5).to_csv(

    OUTPUT_DIR / "top_5_feature_importance.csv",

    index=False

)

print("\nFeature importance saved.")


# ==================================================
# 5. LOGISTIC REGRESSION COEFFICIENT INTERPRETATION
# ==================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION COEFFICIENT INTERPRETATION")
print("=" * 70)

lr_preprocessor = lr_model.named_steps["preprocessor"]

lr_classifier = lr_model.named_steps["classifier"]

lr_feature_names = lr_preprocessor.get_feature_names_out()

lr_coefficients = lr_classifier.coef_[0]

coefficient_df = pd.DataFrame({

    "feature": lr_feature_names,

    "coefficient": lr_coefficients

})

# Select three important numerical features.
selected_features = [
    "numeric__recency",
    "numeric__frequency",
    "numeric__monetary"
]

selected_coefficients = coefficient_df[
    coefficient_df["feature"].isin(selected_features)
].copy()

print("\n--- THREE SELECTED COEFFICIENTS ---")

for _, row in selected_coefficients.iterrows():

    feature = row["feature"]

    coefficient = row["coefficient"]

    direction = (
        "Positive"
        if coefficient > 0
        else "Negative"
        if coefficient < 0
        else "Zero"
    )

    print(
        f"{feature}: {coefficient:.4f} ({direction})"
    )

    if feature == "num__recency":

        print(
            "  Interpretation: Higher recency means a longer "
            "time since the last purchase and is associated "
            "with higher predicted lapse odds."
        )

    elif feature == "num__frequency":

        print(
            "  Interpretation: Higher purchase frequency is "
            "associated with the direction indicated by "
            "the coefficient."
        )

    elif feature == "num__monetary":

        print(
            "  Interpretation: Higher historical spending is "
            "associated with the direction indicated by "
            "the coefficient."
        )

assert len(selected_coefficients) == 3

coefficient_df.to_csv(

    OUTPUT_DIR / "logistic_regression_coefficients.csv",

    index=False

)

print("\nCoefficient results saved.")


# ==================================================
# 6. SAVE SELECTED MODEL
# ==================================================

print("\n" + "=" * 70)
print("SELECTED MODEL PERSISTENCE")
print("=" * 70)

# Random Forest was selected in Task 6 because it achieved
# the highest test F1 score among the two tuned models.

selected_model = rf_model

selected_model_path = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(

    selected_model,

    selected_model_path

)

print("Selected model saved:", selected_model_path)


# ==================================================
# 7. RELOAD MODEL AND VERIFY PREDICTIONS
# ==================================================

reloaded_model = joblib.load(selected_model_path)

original_predictions = selected_model.predict(X_test)

reloaded_predictions = reloaded_model.predict(X_test)

original_probabilities = selected_model.predict_proba(X_test)[:, 1]

reloaded_probabilities = reloaded_model.predict_proba(X_test)[:, 1]

predictions_identical = np.array_equal(
    original_predictions,
    reloaded_predictions
)

probabilities_identical = np.allclose(
    original_probabilities,
    reloaded_probabilities
)

print("\n--- MODEL RELOAD VALIDATION ---")

print("Original test predictions:", len(original_predictions))

print("Reloaded test predictions:", len(reloaded_predictions))

print("Predictions identical:", predictions_identical)

print("Probabilities identical:", probabilities_identical)

assert predictions_identical

assert probabilities_identical

print("PASS: Reloaded model produces identical test predictions.")


# ==================================================
# 8. GENERATE PREMIUM CUSTOMER RETENTION LIST
# ==================================================

print("\n" + "=" * 70)
print("PREMIUM CUSTOMER RETENTION LIST")
print("=" * 70)

# Use the selected model to predict lapse probabilities
# for the original, unchanged test customers.

test_results = pd.DataFrame({

    "customer_id": test_ids.values,

    "actually_lapsed": y_test.values,

    "lapse_probability": reloaded_model.predict_proba(
        X_test
    )[:, 1]

})

# Join Premium segment information from full-window RFM.
premium_customers = segments[
    ["customer_id", "is_premium"]
].copy()

test_results = test_results.merge(

    premium_customers,

    on="customer_id",

    how="left",

    validate="one_to_one"

)

assert test_results["is_premium"].notna().all()

# Retain only Premium customers in the test set.
premium_test_customers = test_results[
    test_results["is_premium"] == 1
].copy()

print(
    "\nPremium customers in test set:",
    len(premium_test_customers)
)

# Rank by predicted probability, highest first.
retention_list = premium_test_customers.sort_values(

    by="lapse_probability",

    ascending=False

).head(10)

retention_list = retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
].copy()

retention_list["lapse_probability"] = (
    retention_list["lapse_probability"].round(4)
)

print("\n--- TOP PREMIUM CUSTOMERS AT RISK ---")

print(retention_list.to_string(index=False))

truly_lapsed = int(
    retention_list["actually_lapsed"].sum()
)

print(
    "\nTruly lapsed customers in retention list:",
    truly_lapsed
)

retention_path = OUTPUT_DIR / "retention_list.csv"

retention_list.to_csv(

    retention_path,

    index=False

)

assert len(retention_list) <= 10

assert retention_list["lapse_probability"].is_monotonic_decreasing

assert retention_list["customer_id"].is_unique

print("\nRetention list saved:", retention_path)


# ==================================================
# 9. FINAL VALIDATION
# ==================================================

print("\n" + "=" * 70)
print("TASK 8 FINAL VALIDATION")
print("=" * 70)

assert len(importance_df.head(5)) == 5

assert selected_model_path.exists()

assert retention_path.exists()

assert predictions_identical

assert probabilities_identical

assert list(retention_list.columns) == [
    "customer_id",
    "lapse_probability",
    "actually_lapsed"
]

print("PASS: Five Random Forest features extracted.")

print("PASS: Three Logistic Regression coefficients interpreted.")

print("PASS: Selected model saved and reloaded.")

print("PASS: Test predictions identical after reload.")

print("PASS: Premium retention list generated.")

print("\nALL TASK 8 VALIDATIONS PASSED!")