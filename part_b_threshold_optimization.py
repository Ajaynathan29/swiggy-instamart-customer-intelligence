import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.base import clone

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"

MODEL_PATH = BASE_DIR / "data" / "processed" / "logistic_regression.joblib"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("INSTAMART CLASS-BALANCE EXPERIMENT")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

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

feature_columns = numeric_features + categorical_features

X = df[feature_columns].copy()

y = df["lapsed"].copy()


# --------------------------------------------------
# 3. RECREATE ORIGINAL TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42

)

print("\n--- ORIGINAL TRAINING DISTRIBUTION ---")

print(y_train.value_counts().sort_index())

print("\n--- ORIGINAL TEST DISTRIBUTION ---")

print(y_test.value_counts().sort_index())

assert len(X_train) == 924

assert len(X_test) == 232


# --------------------------------------------------
# 4. LOAD BEST LOGISTIC REGRESSION MODEL
# --------------------------------------------------

original_model = joblib.load(MODEL_PATH)

print("\nLoaded tuned Logistic Regression model.")

print(
    "Selected parameters:",
    original_model.named_steps["classifier"].get_params()["C"]
)


# --------------------------------------------------
# 5. EVALUATE ORIGINAL MODEL
# --------------------------------------------------

print("\n" + "=" * 70)
print("ORIGINAL UNBALANCED MODEL")
print("=" * 70)

original_predictions = original_model.predict(X_test)

original_precision = precision_score(
    y_test,
    original_predictions,
    zero_division=0
)

original_recall = recall_score(
    y_test,
    original_predictions,
    zero_division=0
)

original_f1 = f1_score(
    y_test,
    original_predictions,
    zero_division=0
)

original_accuracy = accuracy_score(
    y_test,
    original_predictions
)

print("Accuracy:", round(original_accuracy, 4))

print("Precision:", round(original_precision, 4))

print("Recall:", round(original_recall, 4))

print("F1 Score:", round(original_f1, 4))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, original_predictions))


# --------------------------------------------------
# 6. OVERSAMPLE MINORITY CLASS
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING-ONLY RANDOM OVERSAMPLING")
print("=" * 70)

training_data = X_train.copy()

training_data["lapsed"] = y_train.values

majority_class = training_data[
    training_data["lapsed"] == 0
]

minority_class = training_data[
    training_data["lapsed"] == 1
]

print("\nBefore oversampling:")

print(training_data["lapsed"].value_counts().sort_index())

majority_count = len(majority_class)

minority_count = len(minority_class)

additional_samples = majority_count - minority_count

assert additional_samples == 216

# Draw additional minority-class rows with replacement.
minority_oversampled = minority_class.sample(

    n=additional_samples,

    replace=True,

    random_state=42

)

# Keep all original rows and add duplicated minority rows.
balanced_training_data = pd.concat(

    [
        training_data,
        minority_oversampled
    ],

    ignore_index=True

)

# Shuffle the balanced training dataset.
balanced_training_data = balanced_training_data.sample(

    frac=1,

    random_state=42

).reset_index(drop=True)


X_balanced = balanced_training_data[feature_columns]

y_balanced = balanced_training_data["lapsed"]


print("\nAfter oversampling:")

print(y_balanced.value_counts().sort_index())

print("\nOriginal training rows:", len(X_train))

print("Additional duplicated rows:", additional_samples)

print("Balanced training rows:", len(X_balanced))

assert len(X_balanced) == 1140

assert y_balanced.value_counts()[0] == 570

assert y_balanced.value_counts()[1] == 570


# --------------------------------------------------
# 7. TRAIN BALANCED LOGISTIC REGRESSION
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING OVERSAMPLED LOGISTIC REGRESSION")
print("=" * 70)

# Clone the previously tuned pipeline.
# This preserves the selected hyperparameter C=100
# without reusing the original fitted preprocessing.

balanced_model = clone(original_model)

balanced_model.fit(
    X_balanced,
    y_balanced
)

print("Balanced Logistic Regression training completed.")


# --------------------------------------------------
# 8. EVALUATE ON ORIGINAL TEST SET
# --------------------------------------------------

print("\n" + "=" * 70)
print("OVERSAMPLED MODEL TEST RESULTS")
print("=" * 70)

balanced_predictions = balanced_model.predict(X_test)

balanced_precision = precision_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_recall = recall_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_f1 = f1_score(
    y_test,
    balanced_predictions,
    zero_division=0
)

balanced_accuracy = accuracy_score(
    y_test,
    balanced_predictions
)

print("Accuracy:", round(balanced_accuracy, 4))

print("Precision:", round(balanced_precision, 4))

print("Recall:", round(balanced_recall, 4))

print("F1 Score:", round(balanced_f1, 4))

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, balanced_predictions))


# --------------------------------------------------
# 9. COMPARE BOTH MODELS
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLASS-BALANCE EXPERIMENT COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Original Model": [
        original_accuracy,
        original_precision,
        original_recall,
        original_f1
    ],

    "Oversampled Model": [
        balanced_accuracy,
        balanced_precision,
        balanced_recall,
        balanced_f1
    ]

})

comparison["Change"] = (

    comparison["Oversampled Model"]
    - comparison["Original Model"]

)

print(comparison.round(4).to_string(index=False))


# --------------------------------------------------
# 10. IDENTIFY BIGGEST METRIC CHANGE
# --------------------------------------------------

metric_changes = {

    "Accuracy": abs(balanced_accuracy - original_accuracy),

    "Precision": abs(balanced_precision - original_precision),

    "Recall": abs(balanced_recall - original_recall),

    "F1 Score": abs(balanced_f1 - original_f1)

}

largest_change = max(
    metric_changes,
    key=metric_changes.get
)

print("\nLargest metric change:", largest_change)

print(
    "Absolute change:",
    round(metric_changes[largest_change], 4)
)


# --------------------------------------------------
# 11. SAVE BALANCED MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(

    balanced_model,

    OUTPUT_DIR / "logistic_regression_oversampled.joblib"

)

comparison.to_csv(

    OUTPUT_DIR / "oversampling_comparison.csv",

    index=False

)


# --------------------------------------------------
# 12. FINAL VALIDATION
# --------------------------------------------------

assert len(X_test) == 232

assert len(y_test) == 232

assert len(X_balanced) == 1140

assert balanced_model is not None

print("\n" + "=" * 70)

print("ALL OVERSAMPLING VALIDATIONS PASSED!")

print("=" * 70)

print("\nSaved files:")

print(OUTPUT_DIR / "logistic_regression_oversampled.joblib")

print(OUTPUT_DIR / "oversampling_comparison.csv")