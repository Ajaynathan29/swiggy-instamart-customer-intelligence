import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "orders.csv"

orders = pd.read_csv(INPUT_PATH)

print("=" * 70)
print("RAW ORDERS DATA ANALYSIS")
print("=" * 70)

# --------------------------------------------------
# 1. DATASET OVERVIEW
# --------------------------------------------------

print("\n--- DATASET OVERVIEW ---")

print("Rows:", orders.shape[0])
print("Columns:", orders.shape[1])

print("\nColumn data types:")
print(orders.dtypes)

print("\nMissing values per column:")
missing_report = pd.DataFrame({
    "Missing Count": orders.isna().sum(),
    "Missing Percentage": (
        orders.isna().mean() * 100
    ).round(2)
})

print(missing_report)


# --------------------------------------------------
# 2. ORDER ID ANALYSIS
# --------------------------------------------------

order_counts = orders["order_id"].value_counts()

print("\n--- ORDER ID ANALYSIS ---")

print("Unique order IDs:", orders["order_id"].nunique())

print(
    "Orders with multiple records:",
    (order_counts > 1).sum()
)

print(
    "Orders with one record:",
    (order_counts == 1).sum()
)


# --------------------------------------------------
# 3. ORDER STATUS COMBINATIONS
# --------------------------------------------------

print("\n--- ORDER STATUS COMBINATIONS ---")

status_combinations = (
    orders.groupby("order_id")["status"]
    .agg(lambda x: ", ".join(sorted(set(x))))
)

print(status_combinations.value_counts())


# --------------------------------------------------
# 4. EXACT DUPLICATE ANALYSIS
# --------------------------------------------------

print("\n--- EXACT DUPLICATES ---")

print("Exact duplicate rows:", orders.duplicated().sum())


# --------------------------------------------------
# 5. ORDER STATUS ANALYSIS
# --------------------------------------------------

print("\n--- ORDER STATUS COUNTS ---")

print(orders["status"].value_counts(dropna=False))

print(
    "Pending records:",
    (orders["status"] == "pending").sum()
)


# --------------------------------------------------
# 6. DELIVERY TIME ANALYSIS
# --------------------------------------------------

print("\n--- DELIVERY TIME ANALYSIS ---")

delivery = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)

print("Missing delivery times:", delivery.isna().sum())

print(
    "Negative delivery times:",
    (delivery < 0).sum()
)


# --------------------------------------------------
# 7. ORDER AMOUNT ANALYSIS
# --------------------------------------------------

print("\n--- ORDER AMOUNT ANALYSIS ---")

original_amount = orders["order_amount"]

amount = pd.to_numeric(
    original_amount,
    errors="coerce"
)

print(
    "Originally missing amounts:",
    original_amount.isna().sum()
)

print(
    "Unparseable amounts:",
    (
        original_amount.notna() &
        amount.isna()
    ).sum()
)

print(
    "Total missing or invalid amounts:",
    amount.isna().sum()
)


# --------------------------------------------------
# 8. SAMPLE ORDERS WITH MULTIPLE RECORDS
# --------------------------------------------------

print("\n--- SAMPLE ORDERS WITH MULTIPLE RECORDS ---")

duplicate_orders = orders[
    orders["order_id"].isin(
        order_counts[order_counts > 1].index
    )
]

print(
    duplicate_orders
    .sort_values(["order_id", "order_ts"])
    .head(20)
    .to_string(index=False)
)


# --------------------------------------------------
# 9. TIMESTAMP ANALYSIS
# --------------------------------------------------

print("\n--- TIMESTAMP ANALYSIS ---")

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="coerce"
)

print(
    "Invalid timestamps:",
    orders["order_ts"].isna().sum()
)

print("Earliest order:", orders["order_ts"].min())
print("Latest order:", orders["order_ts"].max())


print("\n" + "=" * 70)
print("RAW ORDERS ANALYSIS COMPLETED")
print("=" * 70)