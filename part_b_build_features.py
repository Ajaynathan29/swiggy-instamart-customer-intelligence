import sqlite3
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "instamart.db"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "customer_features.csv"

CUTOFF_DATE = "2025-09-01"
PREDICTION_END_DATE = "2025-10-01"


# --------------------------------------------------
# 2. SQL QUERY
# --------------------------------------------------

query = """

WITH historical_orders AS (

    SELECT *
    FROM orders
    WHERE order_ts < ?

),

historical_completed AS (

    SELECT *
    FROM historical_orders
    WHERE status = 'completed'

),

customer_rfm AS (

    SELECT

        customer_id,

        CAST(
            julianday(?) - julianday(MAX(order_ts))
            AS INTEGER
        ) AS recency,

        COUNT(DISTINCT order_id) AS frequency,

        ROUND(SUM(order_amount), 2) AS monetary,

        AVG(discount_pct) AS avg_discount_pct,

        AVG(delivery_minutes) AS avg_delivery_minutes

    FROM historical_completed

    GROUP BY customer_id

),

customer_cancellations AS (

    SELECT

        customer_id,

        SUM(
            CASE
                WHEN status = 'cancelled' THEN 1
                ELSE 0
            END
        ) AS cancelled_orders,

        COUNT(*) AS final_orders,

        CAST(
            SUM(
                CASE
                    WHEN status = 'cancelled' THEN 1
                    ELSE 0
                END
            ) AS REAL
        ) / COUNT(*) AS cancellation_rate

    FROM historical_orders

    WHERE status IN ('completed', 'cancelled')

    GROUP BY customer_id

),

category_counts AS (

    SELECT

        customer_id,

        product_category,

        COUNT(*) AS category_orders

    FROM (

        SELECT
            o.customer_id,
            p.category AS product_category

        FROM historical_completed o

        JOIN products p
            ON o.product_id = p.product_id

    )

    GROUP BY customer_id, product_category

),

favourite_categories AS (

    SELECT

        customer_id,

        product_category AS favourite_category

    FROM (

        SELECT

            customer_id,

            product_category,

            category_orders,

            ROW_NUMBER() OVER (

                PARTITION BY customer_id

                ORDER BY
                    category_orders DESC,
                    product_category ASC

            ) AS category_rank

        FROM category_counts

    )

    WHERE category_rank = 1

),

future_activity AS (

    SELECT DISTINCT customer_id

    FROM orders

    WHERE status = 'completed'

      AND order_ts >= ?

      AND order_ts < ?

)

SELECT

    c.customer_id,

    r.recency,

    r.frequency,

    r.monetary,

    r.avg_discount_pct,

    COALESCE(
        cc.cancellation_rate,
        0.0
    ) AS cancellation_rate,

    r.avg_delivery_minutes,

    fc.favourite_category,

    c.city,

    c.age,

    c.gender,

    CAST(
        julianday(?) - julianday(c.signup_date)
        AS INTEGER
    ) AS tenure_days,

    CASE

        WHEN fa.customer_id IS NULL THEN 1

        ELSE 0

    END AS lapsed

FROM customer_rfm r

JOIN customers c

    ON r.customer_id = c.customer_id

LEFT JOIN customer_cancellations cc

    ON r.customer_id = cc.customer_id

LEFT JOIN favourite_categories fc

    ON r.customer_id = fc.customer_id

LEFT JOIN future_activity fa

    ON r.customer_id = fa.customer_id

ORDER BY c.customer_id;

"""


# --------------------------------------------------
# 3. BUILD CUSTOMER FEATURES
# --------------------------------------------------

print("=" * 70)
print("INSTAMART LAPSE PREDICTION FEATURE ENGINEERING")
print("=" * 70)

print("\nHistorical feature window:")
print(f"All orders before {CUTOFF_DATE}")

print("\nLapse prediction window:")
print(f"{CUTOFF_DATE} to September 30, 2025")

print("\nLoading database...")

db = sqlite3.connect(DB_PATH)

try:

    df = pd.read_sql_query(

        query,

        db,

        params=(

            CUTOFF_DATE,
            CUTOFF_DATE,
            CUTOFF_DATE,
            PREDICTION_END_DATE,
            CUTOFF_DATE

        )

    )

finally:

    db.close()


# --------------------------------------------------
# 4. CREATE OUTPUT DIRECTORY
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# 5. SAVE FEATURE DATASET
# --------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 6. DATASET OVERVIEW
# --------------------------------------------------

print("\n" + "=" * 70)
print("CUSTOMER FEATURE DATASET")
print("=" * 70)

print("\nDataset dimensions:")

print("Total customers:", len(df))

print("Total columns:", len(df.columns))

print(
    "Unique customers:",
    df["customer_id"].nunique()
)


# --------------------------------------------------
# 7. LAPSE DISTRIBUTION
# --------------------------------------------------

print("\n--- LAPSE DISTRIBUTION ---")

lapse_distribution = (
    df["lapsed"]
    .value_counts()
    .sort_index()
)

print(lapse_distribution)

active_count = (df["lapsed"] == 0).sum()

lapsed_count = (df["lapsed"] == 1).sum()

print("\nActive customers:", active_count)

print("Lapsed customers:", lapsed_count)

print(
    "Lapse percentage:",
    round(lapsed_count / len(df) * 100, 2),
    "%"
)


# --------------------------------------------------
# 8. FEATURE INFORMATION
# --------------------------------------------------

print("\n--- FEATURE COLUMNS ---")

print(df.columns.tolist())


# --------------------------------------------------
# 9. MISSING VALUE ANALYSIS
# --------------------------------------------------

print("\n--- MISSING VALUE ANALYSIS ---")

missing_values = df.isnull().sum()

print(missing_values)

print(
    "\nTotal missing values:",
    missing_values.sum()
)


# --------------------------------------------------
# 10. NUMERICAL FEATURE SUMMARY
# --------------------------------------------------

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

print("\n--- NUMERICAL FEATURE SUMMARY ---")

print(
    df[numeric_features]
    .describe()
    .round(2)
)


# --------------------------------------------------
# 11. CATEGORICAL FEATURE SUMMARY
# --------------------------------------------------

categorical_features = [

    "favourite_category",
    "city",
    "gender"

]

print("\n--- CATEGORICAL FEATURE SUMMARY ---")

for column in categorical_features:

    print(f"\n{column}:")

    print(df[column].value_counts())


# --------------------------------------------------
# 12. FEATURE PREVIEW
# --------------------------------------------------

print("\n--- FEATURE PREVIEW ---")

print(
    df.head(10).to_string(index=False)
)


# --------------------------------------------------
# 13. VALIDATION CHECKS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ENGINEERING VALIDATION")
print("=" * 70)


# Customer count

assert len(df) == 1156, (
    f"Expected 1156 customers, found {len(df)}"
)

print("PASS: Cohort contains 1156 customers")


# Unique customer IDs

assert df["customer_id"].nunique() == 1156, (
    "Duplicate customer IDs detected"
)

print("PASS: Customer IDs are unique")


# Expected columns

expected_columns = [

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

assert df.columns.tolist() == expected_columns, (
    "Feature columns do not match expected structure"
)

print("PASS: Expected 13 columns exist")


# Lapse distribution

assert lapsed_count == 443, (
    f"Expected 443 lapsed customers, found {lapsed_count}"
)

assert active_count == 713, (
    f"Expected 713 active customers, found {active_count}"
)

print("PASS: Lapse distribution matches reference")


# Lapse percentage

lapse_percentage = lapsed_count / len(df) * 100

assert abs(lapse_percentage - 38.32) < 0.1, (
    f"Unexpected lapse percentage: {lapse_percentage}"
)

print("PASS: Lapse percentage matches reference")


# Binary target

assert df["lapsed"].isin([0, 1]).all(), (
    "Invalid lapse labels detected"
)

print("PASS: Lapse labels are binary")


# Positive numerical values

assert (df["frequency"] > 0).all()

assert (df["monetary"] > 0).all()

assert (df["recency"] >= 0).all()

assert (df["tenure_days"] >= 0).all()

print("PASS: Numerical feature ranges validated")


# Cancellation rate

assert df["cancellation_rate"].between(0, 1).all(), (
    "Cancellation rate must be between 0 and 1"
)

print("PASS: Cancellation rate is between 0 and 1")


# Missing values

assert not df.isnull().any().any(), (
    "Missing values detected in feature dataset"
)

print("PASS: No missing values in feature dataset")


# Feature leakage check

forbidden_features = [

    "segment",
    "segment_id",
    "segment_name",
    "is_premium"

]

assert not any(
    column in df.columns
    for column in forbidden_features
), "Segment leakage detected"

print("PASS: No segment labels used as features")


print("\n" + "=" * 70)
print("ALL LAPSE FEATURE VALIDATIONS PASSED!")
print("=" * 70)

print("\nFeature dataset saved to:")

print(OUTPUT_PATH)