import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ORDERS_FILE = RAW_DIR / "orders.csv"
PRODUCTS_FILE = RAW_DIR / "products.csv"

CLEAN_ORDERS_FILE = PROCESSED_DIR / "orders_cleaned.csv"
COMPLETED_ORDERS_FILE = PROCESSED_DIR / "completed_orders.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

orders = pd.read_csv(ORDERS_FILE)
products = pd.read_csv(PRODUCTS_FILE)

print("\n========================================")
print("        ORDER DATA CLEANING")
print("========================================")

print("Original order records:", len(orders))
print("Original unique order IDs:", orders["order_id"].nunique())


# --------------------------------------------------
# 1. REMOVE EXACT DUPLICATES
# --------------------------------------------------

original_count = len(orders)

orders = orders.drop_duplicates().copy()

exact_duplicates_removed = original_count - len(orders)

print("\n--- EXACT DUPLICATE REMOVAL ---")
print("Exact duplicate records removed:", exact_duplicates_removed)
print("Records remaining:", len(orders))


# --------------------------------------------------
# 2. DATA TYPE CONVERSION
# --------------------------------------------------

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

orders["quantity"] = pd.to_numeric(
    orders["quantity"],
    errors="coerce"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

orders["discount_pct"] = pd.to_numeric(
    orders["discount_pct"],
    errors="coerce"
)


# --------------------------------------------------
# 3. HANDLE INVALID DELIVERY TIMES
# --------------------------------------------------

# Negative delivery durations are invalid.
# Convert them to missing values before imputation.

negative_delivery_count = int(
    (orders["delivery_minutes"] < 0).sum()
)

orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

print("\n--- INVALID DELIVERY TIMES ---")
print(
    "Negative delivery times converted to missing:",
    negative_delivery_count
)


# --------------------------------------------------
# 4. KEEP LATEST RECORD FOR EACH ORDER ID
# --------------------------------------------------

records_before_latest = len(orders)

orders = orders.sort_values(
    by="order_ts",
    na_position="first"
)

orders = orders.drop_duplicates(
    subset=["order_id"],
    keep="last"
).copy()

duplicate_order_records_removed = (
    records_before_latest - len(orders)
)

print("\n--- LATEST ORDER RECORD RETENTION ---")
print("Records removed:", duplicate_order_records_removed)
print("Final unique order IDs:", orders["order_id"].nunique())


# --------------------------------------------------
# 5. VALIDATE ORDER AMOUNTS
# --------------------------------------------------

# Create product price mapping.

product_price_map = products.set_index(
    "product_id"
)["price_inr"]

orders["unit_price"] = orders["product_id"].map(
    product_price_map
)

# Calculate expected order amount:
# quantity * product price * (1 - discount percentage / 100)

orders["expected_amount"] = (
    orders["quantity"]
    * orders["unit_price"]
    * (
        1 - orders["discount_pct"].fillna(0) / 100
    )
).round(2)

# Check records where an existing amount can be validated.

amount_validation_mask = (
    orders["order_amount"].notna()
    & orders["expected_amount"].notna()
)

matching_amounts = (
    (
        orders.loc[amount_validation_mask, "order_amount"]
        - orders.loc[amount_validation_mask, "expected_amount"]
    ).abs() < 0.01
)

records_checked = int(amount_validation_mask.sum())
matching_records = int(matching_amounts.sum())
non_matching_records = records_checked - matching_records

print("\n--- PRICING FORMULA VALIDATION ---")
print("Records checked:", records_checked)
print("Matching amounts:", matching_records)
print("Non-matching amounts:", non_matching_records)


# --------------------------------------------------
# 6. RECOVER MISSING ORDER AMOUNTS
# --------------------------------------------------

missing_amounts_before = int(
    orders["order_amount"].isna().sum()
)

# Recover missing amounts for all orders where
# product price and quantity information are available.

recover_amount_mask = (
    orders["order_amount"].isna()
    & orders["expected_amount"].notna()
)

orders.loc[
    recover_amount_mask,
    "order_amount"
] = orders.loc[
    recover_amount_mask,
    "expected_amount"
]

amounts_recovered = int(recover_amount_mask.sum())

missing_amounts_after = int(
    orders["order_amount"].isna().sum()
)

print("\n--- MISSING AMOUNT RECOVERY ---")
print("Missing amounts before:", missing_amounts_before)
print("Amounts recovered:", amounts_recovered)
print("Missing amounts after:", missing_amounts_after)


# --------------------------------------------------
# 7. IMPUTE MISSING DELIVERY TIMES
#    COMPLETED ORDERS ONLY
#    PRODUCT-CATEGORY MEDIAN USING TRANSFORM
# --------------------------------------------------

# Map each product to its category.

product_category_map = products.set_index(
    "product_id"
)["category"]

orders["category"] = orders["product_id"].map(
    product_category_map
)

# Identify completed orders.

completed_mask = orders["status"].eq("completed")

# Count missing completed-order delivery times.

missing_delivery_before = int(
    (
        completed_mask
        & orders["delivery_minutes"].isna()
    ).sum()
)

# Create a separate completed-orders dataset for imputation.

completed_data = orders.loc[
    completed_mask
].copy()

# Calculate category median using groupby and transform.
# Only completed orders are included.

completed_data["category_median"] = (
    completed_data
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

# Calculate overall completed-order median as fallback.

overall_completed_median = (
    completed_data["delivery_minutes"].median()
)

# Fill missing delivery times using category median.

completed_data["delivery_minutes"] = (
    completed_data["delivery_minutes"]
    .fillna(completed_data["category_median"])
)

# Use overall completed-order median if a category
# does not have a valid median.

completed_data["delivery_minutes"] = (
    completed_data["delivery_minutes"]
    .fillna(overall_completed_median)
)

# Stop if imputation is still unsuccessful.

if completed_data["delivery_minutes"].isna().any():
    raise ValueError(
        "Delivery-time imputation failed: "
        "no valid completed-order median is available."
    )

# Write imputed delivery times back into the main dataset.

orders.loc[
    completed_mask,
    "delivery_minutes"
] = completed_data["delivery_minutes"]

# Count missing delivery times after imputation.

missing_delivery_after = int(
    (
        completed_mask
        & orders["delivery_minutes"].isna()
    ).sum()
)

delivery_values_imputed = (
    missing_delivery_before - missing_delivery_after
)

# Display category-wise medians.

category_medians = (
    completed_data
    .groupby("category")["delivery_minutes"]
    .median()
)

print("\n--- DELIVERY TIME IMPUTATION ---")
print(
    "Missing completed-order delivery times before:",
    missing_delivery_before
)

print("Delivery times imputed:", delivery_values_imputed)

print(
    "Missing completed-order delivery times after:",
    missing_delivery_after
)

print(
    "Overall completed-order median:",
    overall_completed_median
)

print("\n--- CATEGORY-WISE DELIVERY MEDIANS ---")
print(category_medians)


# --------------------------------------------------
# 8. REMOVE TEMPORARY COLUMNS
# --------------------------------------------------

temporary_columns = [
    "unit_price",
    "expected_amount",
    "category"
]

orders.drop(
    columns=temporary_columns,
    inplace=True,
    errors="ignore"
)


# --------------------------------------------------
# 9. CREATE COMPLETED AND CANCELLED DATASETS
# --------------------------------------------------

final_orders = orders.loc[
    orders["status"].eq("completed")
].copy()

cancelled_orders = orders.loc[
    orders["status"].eq("cancelled")
].copy()


# --------------------------------------------------
# 10. FINAL VALIDATION
# --------------------------------------------------

print("\n========================================")
print("          FINAL VALIDATION")
print("========================================")

# Every order ID must be unique.

assert orders["order_id"].is_unique, (
    "Duplicate order IDs found after cleaning."
)

# No pending orders should remain after keeping latest records.

assert not orders["status"].eq("pending").any(), (
    "Pending orders remain after latest-record retention."
)

# Completed orders must not have missing delivery times.

assert not final_orders["delivery_minutes"].isna().any(), (
    "Completed orders still contain missing delivery times."
)

# Completed orders must not have negative delivery times.

assert (
    final_orders["delivery_minutes"] >= 0
).all(), (
    "Negative delivery times found in completed orders."
)

# Completed orders must not have missing order amounts.

assert not final_orders["order_amount"].isna().any(), (
    "Completed orders still contain missing order amounts."
)

# Completed order amounts must not be negative.

assert (
    final_orders["order_amount"] >= 0
).all(), (
    "Negative completed-order amounts found."
)

# Cancelled orders should retain missing delivery times
# wherever delivery time was not recorded.

assert (
    cancelled_orders["delivery_minutes"].isna().sum()
    == len(cancelled_orders)
), (
    "Cancelled orders should retain missing delivery times."
)

# No missing order amounts should remain.

assert not orders["order_amount"].isna().any(), (
    "Missing order amounts remain after recovery."
)

# No temporary columns should remain.

assert not any(
    column in orders.columns
    for column in temporary_columns
), (
    "Temporary columns remain in the cleaned dataset."
)

print("Unique order ID validation passed.")
print("No pending orders validation passed.")
print("Completed-order delivery-time validation passed.")
print("Completed-order amount validation passed.")
print("Cancelled-order delivery-time validation passed.")
print("Missing order amount validation passed.")
print("Temporary-column validation passed.")


# --------------------------------------------------
# 11. SAVE CLEANED DATASETS
# --------------------------------------------------

orders.to_csv(
    CLEAN_ORDERS_FILE,
    index=False
)

final_orders.to_csv(
    COMPLETED_ORDERS_FILE,
    index=False
)


# --------------------------------------------------
# 12. FINAL CLEANING SUMMARY
# --------------------------------------------------

print("\n========================================")
print("        ORDER CLEANING SUMMARY")
print("========================================")

print("Original records:", original_count)
print("Exact duplicates removed:", exact_duplicates_removed)
print("Repeated order records removed:", duplicate_order_records_removed)
print("Final unique orders:", len(orders))

print("\n--- ORDER STATUS ---")
print(orders["status"].value_counts(dropna=False))

print("\n--- COMPLETED ORDERS ---")
print("Completed orders:", len(final_orders))

print(
    "Missing completed-order amounts:",
    final_orders["order_amount"].isna().sum()
)

print(
    "Missing completed-order delivery times:",
    final_orders["delivery_minutes"].isna().sum()
)

print("\n--- CANCELLED ORDERS ---")
print("Cancelled orders:", len(cancelled_orders))

print(
    "Missing cancelled-order delivery times:",
    cancelled_orders["delivery_minutes"].isna().sum()
)

print("\n--- REVENUE ---")
print(
    "Total revenue: ₹",
    round(final_orders["order_amount"].sum(), 2)
)

print("\n--- SAVED FILES ---")
print("Cleaned orders:", CLEAN_ORDERS_FILE)
print("Completed orders:", COMPLETED_ORDERS_FILE)

print("\nAll order cleaning validations passed!")