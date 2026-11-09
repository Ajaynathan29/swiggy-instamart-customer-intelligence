import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"


# --------------------------------------------------
# 2. EXPECTED DATASET STRUCTURE
# --------------------------------------------------

EXPECTED_ROWS = 1156

EXPECTED_COLUMNS = [
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "favourite_category",
    "city",
    "age",
    "gender",
    "tenure_days",
    "lapsed"
]

NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount_pct",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

CATEGORICAL_FEATURES = [
    "favourite_category",
    "city",
    "gender"
]


# --------------------------------------------------
# 3. VALIDATION HELPER
# --------------------------------------------------

passed = 0
failed = 0


def validate(condition, description):
    global passed, failed

    if condition:
        print(f"PASS: {description}")
        passed += 1
    else:
        print(f"FAIL: {description}")
        failed += 1


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("PART B - TASK 5: FEATURE DATA VALIDATION")
print("=" * 70)

if not INPUT_PATH.exists():
    print(f"ERROR: Feature dataset not found: {INPUT_PATH}")
    sys.exit(1)

df = pd.read_csv(INPUT_PATH)

print("\n--- DATASET OVERVIEW ---")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:", df.columns.tolist())


# --------------------------------------------------
# 5. REQUIRED COLUMN VALIDATION
# --------------------------------------------------

print("\n--- REQUIRED COLUMN VALIDATION ---")

missing_columns = [
    column for column in EXPECTED_COLUMNS
    if column not in df.columns
]

extra_columns = [
    column for column in df.columns
    if column not in EXPECTED_COLUMNS
]

validate(
    len(missing_columns) == 0,
    "All 13 expected feature columns exist"
)

validate(
    len(extra_columns) == 0,
    "No unexpected columns exist"
)

if missing_columns:
    print("Missing columns:", missing_columns)

if extra_columns:
    print("Unexpected columns:", extra_columns)

if missing_columns:
    print("\nCannot continue: required columns are missing.")
    sys.exit(1)


# --------------------------------------------------
# 6. DATASET SIZE AND CUSTOMER ID VALIDATION
# --------------------------------------------------

print("\n--- DATASET INTEGRITY ---")

validate(
    len(df) == EXPECTED_ROWS,
    f"Expected {EXPECTED_ROWS} customers; found {len(df)}"
)

validate(
    df["customer_id"].notna().all(),
    "Customer IDs contain no missing values"
)

validate(
    df["customer_id"].is_unique,
    "Customer IDs are unique"
)

validate(
    df.duplicated().sum() == 0,
    "No duplicate rows"
)


# --------------------------------------------------
# 7. MISSING VALUE VALIDATION
# --------------------------------------------------

print("\n--- MISSING VALUE ANALYSIS ---")

missing_counts = df.isna().sum()

print(missing_counts.to_string())

total_missing = int(missing_counts.sum())

validate(
    total_missing == 0,
    "No missing values in the feature dataset"
)


# --------------------------------------------------
# 8. NUMERICAL FEATURE SUMMARY
# --------------------------------------------------

print("\n--- NUMERICAL FEATURE SUMMARY ---")

print(
    df[NUMERIC_FEATURES]
    .describe()
    .round(2)
    .to_string()
)

numeric_conversion_ok = True

for column in NUMERIC_FEATURES:
    converted = pd.to_numeric(df[column], errors="coerce")

    if converted.isna().any():
        numeric_conversion_ok = False

    if not np.isfinite(converted.dropna()).all():
        numeric_conversion_ok = False

validate(
    numeric_conversion_ok,
    "All numerical features contain valid finite numbers"
)


# --------------------------------------------------
# 9. FEATURE RANGE VALIDATION
# --------------------------------------------------

print("\n--- FEATURE RANGE CHECKS ---")

validate(
    (df["recency"] >= 0).all(),
    "Recency values are non-negative"
)

validate(
    (df["frequency"] >= 1).all(),
    "Frequency values are at least 1"
)

validate(
    (df["monetary"] >= 0).all(),
    "Monetary values are non-negative"
)

validate(
    df["avg_discount_pct"].between(0, 100).all(),
    "Average discount percentage is between 0 and 100"
)

validate(
    df["cancellation_rate"].between(0, 1).all(),
    "Cancellation rate is between 0 and 1"
)

validate(
    (df["avg_delivery_minutes"] > 0).all(),
    "Average delivery time is positive"
)

validate(
    df["age"].between(18, 60).all(),
    "Customer ages are between 18 and 60"
)

validate(
    (df["tenure_days"] >= 0).all(),
    "Customer tenure is non-negative"
)


# --------------------------------------------------
# 10. CATEGORICAL FEATURE VALIDATION
# --------------------------------------------------

print("\n--- CATEGORICAL DISTRIBUTIONS ---")

for column in CATEGORICAL_FEATURES:
    print(f"\n{column.upper()}")
    print(df[column].value_counts(dropna=False).to_string())

validate(
    df[CATEGORICAL_FEATURES].notna().all().all(),
    "Categorical features contain no missing values"
)


# --------------------------------------------------
# 11. TARGET DISTRIBUTION
# --------------------------------------------------

print("\n--- LAPSE TARGET DISTRIBUTION ---")

target_counts = df["lapsed"].value_counts().sort_index()

print("Class counts:")
print(target_counts.to_string())

print("\nClass percentages:")
print(
    df["lapsed"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .to_string()
)

validate(
    set(df["lapsed"].unique()).issubset({0, 1}),
    "Lapse target contains only binary labels (0 and 1)"
)

validate(
    int((df["lapsed"] == 0).sum()) == 713,
    "Active customer count matches reference (713)"
)

validate(
    int((df["lapsed"] == 1).sum()) == 443,
    "Lapsed customer count matches reference (443)"
)


# --------------------------------------------------
# 12. PREVENT TARGET / ID LEAKAGE
# --------------------------------------------------

print("\n--- FEATURE LEAKAGE CHECK ---")

validate(
    "customer_id" not in NUMERIC_FEATURES
    and "lapsed" not in NUMERIC_FEATURES,
    "Customer ID and target are excluded from numerical features"
)

validate(
    not any("segment" in column.lower() for column in df.columns),
    "Customer segment labels are not included in the feature dataset"
)


# --------------------------------------------------
# 13. FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE VALIDATION SUMMARY")
print("=" * 70)

print("Total checks:", passed + failed)
print("Passed:", passed)
print("Failed:", failed)

if failed == 0:
    print("\nALL FEATURE VALIDATIONS PASSED!")
    sys.exit(0)
else:
    print("\nFEATURE VALIDATION FAILED.")
    sys.exit(1)