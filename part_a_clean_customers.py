import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "raw" / "customers.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_PATH = OUTPUT_DIR / "customers_cleaned.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

customers = pd.read_csv(INPUT_PATH)

original_count = len(customers)

print("=" * 60)
print("CUSTOMER DATA CLEANING")
print("=" * 60)

print("Original customers:", original_count)


# --------------------------------------------------
# 3. STANDARDIZE CITY
# --------------------------------------------------

customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

customers["city"] = customers["city"].replace(
    "",
    pd.NA
)

customers["city"] = customers["city"].fillna("Unknown")


# --------------------------------------------------
# 4. STANDARDIZE GENDER
# --------------------------------------------------

customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
)

customers["gender"] = customers["gender"].replace({
    "m": "male",
    "male": "male",
    "f": "female",
    "female": "female"
})

customers["gender"] = customers["gender"].replace(
    "",
    pd.NA
)

customers["gender"] = customers["gender"].fillna("unknown")


# --------------------------------------------------
# 5. CONVERT SIGNUP DATE
# --------------------------------------------------

customers["signup_date"] = pd.to_datetime(
    customers["signup_date"],
    errors="coerce"
)

invalid_signup_dates = customers["signup_date"].isna().sum()


# --------------------------------------------------
# 6. CLEAN AGE VALUES
# --------------------------------------------------

customers["age"] = pd.to_numeric(
    customers["age"],
    errors="coerce"
)

missing_before = customers["age"].isna().sum()

# According to the tutor brief, ages outside 10-100
# are considered invalid.

invalid_age_mask = (
    (customers["age"] < 10)
    | (customers["age"] > 100)
)

invalid_age_count = int(invalid_age_mask.sum())

customers.loc[invalid_age_mask, "age"] = pd.NA

missing_after_invalid = int(customers["age"].isna().sum())


# --------------------------------------------------
# 7. CITY-BASED MEDIAN IMPUTATION
# --------------------------------------------------

# Calculate each customer's city median using transform.
# This preserves the original row count and index alignment.

city_medians = (
    customers.groupby("city")["age"]
    .transform("median")
)

# Overall median is the fallback if a city has no valid ages.

overall_median = customers["age"].median()

if pd.isna(overall_median):
    raise ValueError(
        "Cannot impute age: no valid age values available."
    )

# Fill missing ages using the corresponding city median.

customers["age"] = customers["age"].fillna(city_medians)

# Use overall median only when city median is unavailable.

fallback_count = int(customers["age"].isna().sum())

customers["age"] = customers["age"].fillna(overall_median)

imputed_count = missing_after_invalid


# --------------------------------------------------
# 8. SAVE CLEANED DATA
# --------------------------------------------------

customers.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------
# 9. CLEANING SUMMARY
# --------------------------------------------------

print("\n--- CLEANING SUMMARY ---")

print("Original customers:", original_count)

print("Final customers:", len(customers))

print("Invalid ages detected:", invalid_age_count)

print("Missing ages before cleaning:", missing_before)

print(
    "Missing ages after invalid-value handling:",
    missing_after_invalid
)

print("Missing ages requiring fallback:", fallback_count)

print("Overall median fallback:", overall_median)

print("Age values imputed:", imputed_count)

print("Invalid signup dates:", invalid_signup_dates)


# --------------------------------------------------
# 10. CITY-BASED AGE MEDIANS
# --------------------------------------------------

print("\n--- CITY-BASED AGE MEDIANS ---")

print(
    customers.groupby("city")["age"]
    .median()
    .sort_index()
)


# --------------------------------------------------
# 11. GENDER DISTRIBUTION
# --------------------------------------------------

print("\n--- GENDER DISTRIBUTION ---")

print(
    customers["gender"]
    .value_counts()
    .sort_index()
)


# --------------------------------------------------
# 12. AGE STATISTICS
# --------------------------------------------------

print("\n--- AGE STATISTICS ---")

print(customers["age"].describe())


# --------------------------------------------------
# 13. FINAL VALIDATION
# --------------------------------------------------

print("\n--- FINAL VALIDATION ---")

assert len(customers) == original_count, \
    "Customer record count changed!"

assert customers["customer_id"].is_unique, \
    "Duplicate customer IDs found!"

assert customers["age"].notna().all(), \
    "Missing ages remain!"

assert customers["age"].between(18, 60).all(), \
    "Unexpected age values remain!"

assert customers["city"].notna().all(), \
    "Missing city values remain!"

assert customers["gender"].notna().all(), \
    "Missing gender values remain!"

assert set(customers["gender"].unique()).issubset({
    "male",
    "female",
    "unknown"
}), "Unexpected gender categories found!"

assert customers["signup_date"].notna().all(), \
    "Invalid or missing signup dates remain!"

assert pd.api.types.is_datetime64_any_dtype(
    customers["signup_date"]
), "signup_date was not converted to datetime!"

assert OUTPUT_PATH.exists(), \
    "Cleaned customer file was not created!"

print("All customer cleaning checks passed!")

print("\nSaved to:", OUTPUT_PATH)