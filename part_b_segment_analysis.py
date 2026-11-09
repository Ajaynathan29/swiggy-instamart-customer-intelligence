import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SEGMENTS_PATH = BASE_DIR / "outputs" / "customer_segments.csv"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 70)
print("PART B: LAPSE RATE BY CUSTOMER SEGMENT")
print("=" * 70)

if not SEGMENTS_PATH.exists():
    raise FileNotFoundError(f"Segment file not found: {SEGMENTS_PATH}")

if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"Customer features file not found: {FEATURES_PATH}")

segments = pd.read_csv(SEGMENTS_PATH)
features = pd.read_csv(FEATURES_PATH)

print(f"\nSegment records: {len(segments)}")
print(f"Feature records: {len(features)}")


# --------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_segment_columns = {
    "customer_id",
    "recency",
    "frequency",
    "monetary",
    "segment_id",
    "segment_name",
    "is_premium"
}

required_feature_columns = {
    "customer_id",
    "lapsed"
}

missing_segments = required_segment_columns - set(segments.columns)
missing_features = required_feature_columns - set(features.columns)

if missing_segments:
    raise ValueError(
        f"Missing segment columns: {sorted(missing_segments)}"
    )

if missing_features:
    raise ValueError(
        f"Missing feature columns: {sorted(missing_features)}"
    )

if segments["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer IDs found in segment data.")

if features["customer_id"].duplicated().any():
    raise ValueError("Duplicate customer IDs found in feature data.")


# --------------------------------------------------
# 4. MERGE SEGMENTS WITH LAPSE TARGET
# --------------------------------------------------

df = segments.merge(
    features[["customer_id", "lapsed"]],
    on="customer_id",
    how="inner",
    validate="one_to_one"
)

unmatched_customers = len(segments) - len(df)

print(f"\nMatched customers: {len(df)}")
print(f"Customers without lapse labels: {unmatched_customers}")

if len(df) == 0:
    raise ValueError("No customers could be matched.")

if unmatched_customers > 0:
    print(
        "NOTE: Unmatched customers are excluded from lapse-rate "
        "analysis because they have no lapse label."
    )

if df["lapsed"].isna().any():
    raise ValueError("Missing lapse labels found after merging.")

if not set(df["lapsed"].unique()).issubset({0, 1}):
    raise ValueError("Unexpected values found in the lapsed column.")


# --------------------------------------------------
# 5. LAPSE RATE BY CUSTOMER SEGMENT
# --------------------------------------------------

print("\n--- LAPSE RATE BY CUSTOMER SEGMENT ---")

analysis = df.groupby(
    ["segment_id", "segment_name"],
    sort=True
).agg(
    customers=("customer_id", "count"),
    lapsed_customers=("lapsed", "sum"),
    avg_recency=("recency", "mean"),
    avg_orders=("frequency", "mean"),
    avg_spending=("monetary", "mean")
)

analysis["lapse_rate_pct"] = (
    analysis["lapsed_customers"]
    / analysis["customers"]
    * 100
)

print(analysis.round(2).to_string())


# --------------------------------------------------
# 6. SEGMENT DISTRIBUTION
# --------------------------------------------------

print("\n--- SEGMENT DISTRIBUTION ---")

distribution = (
    df.groupby(["segment_id", "segment_name"])
    .size()
    .rename("customers")
)

print(distribution.to_string())


# --------------------------------------------------
# 7. LAPSE RATE CROSS-TABULATION
# --------------------------------------------------

print("\n--- LAPSE RATE BY SEGMENT (%) ---")

lapse_crosstab = pd.crosstab(
    df["segment_name"],
    df["lapsed"],
    normalize="index"
) * 100

lapse_crosstab = lapse_crosstab.rename(
    columns={
        0: "active_pct",
        1: "lapsed_pct"
    }
)

print(lapse_crosstab.round(2).to_string())


# --------------------------------------------------
# 8. FINAL VALIDATION
# --------------------------------------------------

print("\n--- VALIDATION ---")

print(f"Total segment customers: {len(segments)}")
print(f"Matched customers analysed: {len(df)}")
print(f"Unmatched customers excluded: {unmatched_customers}")
print(f"Total segments: {df['segment_id'].nunique()}")
print(f"Total lapsed customers: {int(df['lapsed'].sum())}")

assert len(df) <= len(segments)
assert df["customer_id"].is_unique
assert analysis["customers"].sum() == len(df)
assert analysis["lapsed_customers"].sum() == int(df["lapsed"].sum())

print("\nPASS: Segment lapse analysis completed successfully.")