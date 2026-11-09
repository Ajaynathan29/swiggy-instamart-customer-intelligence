import sqlite3
import shutil
import pandas as pd

from pathlib import Path


# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"

CUSTOMERS_INPUT = PROCESSED_DIR / "customers_cleaned.csv"
ORDERS_INPUT = PROCESSED_DIR / "orders_cleaned.csv"
PRODUCTS_INPUT = RAW_DIR / "products.csv"

CUSTOMERS_OUTPUT = CLEAN_DIR / "customers.csv"
ORDERS_OUTPUT = CLEAN_DIR / "orders.csv"
PRODUCTS_OUTPUT = CLEAN_DIR / "products.csv"

DATABASE_PATH = DATA_DIR / "instamart.db"
COMPATIBILITY_DATABASE_PATH = PROCESSED_DIR / "instamart.db"


# ==================================================
# 2. CREATE DIRECTORIES
# ==================================================

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 3. LOAD CLEANED DATA
# ==================================================

print("=" * 65)
print("       INSTAMART SQLITE DATABASE CREATION")
print("=" * 65)

required_files = [
    CUSTOMERS_INPUT,
    ORDERS_INPUT,
    PRODUCTS_INPUT
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {file_path}"
        )

customers = pd.read_csv(CUSTOMERS_INPUT)

orders = pd.read_csv(ORDERS_INPUT)

products = pd.read_csv(PRODUCTS_INPUT)

print("\n--- INPUT DATASETS ---")

print("Customers:", customers.shape)
print("Orders:", orders.shape)
print("Products:", products.shape)


# ==================================================
# 4. PREPARE CUSTOMER DATA
# ==================================================

customers["age"] = pd.to_numeric(
    customers["age"],
    errors="raise"
)

customers["city"] = (
    customers["city"]
    .astype("string")
    .str.strip()
    .str.title()
)

customers["gender"] = (
    customers["gender"]
    .astype("string")
    .str.strip()
    .str.lower()
)

if "signup_date" in customers.columns:
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")


# ==================================================
# 5. PREPARE PRODUCT DATA
# ==================================================

products["price_inr"] = pd.to_numeric(
    products["price_inr"],
    errors="raise"
)

products["category"] = (
    products["category"]
    .astype("string")
    .str.strip()
    .str.title()
)


# ==================================================
# 6. PREPARE ORDER DATA
# ==================================================

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"],
    errors="raise"
)

# Store timestamps in SQLite using the required format.

orders["order_ts"] = orders["order_ts"].dt.strftime(
    "%Y-%m-%d %H:%M:%S"
)

orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="raise"
)

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"],
    errors="coerce"
)


# ==================================================
# 7. VALIDATE DATA BEFORE SAVING
# ==================================================

print("\n--- DATA VALIDATION ---")

# Customer validation.

assert customers["customer_id"].notna().all()
assert customers["customer_id"].is_unique
assert customers["age"].notna().all()

# Product validation.

assert products["product_id"].notna().all()
assert products["product_id"].is_unique
assert products["price_inr"].notna().all()
assert (products["price_inr"] > 0).all()

# Order validation.

assert orders["order_id"].notna().all()
assert orders["order_id"].is_unique

assert not orders["status"].eq("pending").any()

assert orders["order_amount"].notna().all()

# Foreign-key validation before database insertion.

assert orders["customer_id"].isin(
    customers["customer_id"]
).all(), "Orders contain unknown customer IDs."

assert orders["product_id"].isin(
    products["product_id"]
).all(), "Orders contain unknown product IDs."

print("Customer validation: PASS")
print("Product validation: PASS")
print("Order validation: PASS")
print("Foreign-key reference validation: PASS")


# ==================================================
# 8. SAVE CANONICAL CLEANED CSV FILES
# ==================================================

customers.to_csv(
    CUSTOMERS_OUTPUT,
    index=False
)

products.to_csv(
    PRODUCTS_OUTPUT,
    index=False
)

orders.to_csv(
    ORDERS_OUTPUT,
    index=False
)

print("\n--- CLEANED CSV FILES CREATED ---")

print(CUSTOMERS_OUTPUT)
print(PRODUCTS_OUTPUT)
print(ORDERS_OUTPUT)


# ==================================================
# 9. SQLITE DATA TYPE MAPPING
# ==================================================

def get_sqlite_type(series):

    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"

    if pd.api.types.is_float_dtype(series):
        return "REAL"

    if pd.api.types.is_bool_dtype(series):
        return "INTEGER"

    return "TEXT"


# ==================================================
# 10. CREATE TABLE SQL DYNAMICALLY
# ==================================================

def create_table_sql(table_name, dataframe, primary_key,
                     foreign_keys=None):

    foreign_keys = foreign_keys or []

    column_definitions = []

    for column in dataframe.columns:

        sqlite_type = get_sqlite_type(
            dataframe[column]
        )

        column_name = f'"{column}"'

        if column == primary_key:
            column_definitions.append(
                f"{column_name} {sqlite_type} PRIMARY KEY"
            )
        else:
            column_definitions.append(
                f"{column_name} {sqlite_type}"
            )

    for column, reference_table, reference_column in foreign_keys:

        column_definitions.append(
            f'FOREIGN KEY ("{column}") '
            f'REFERENCES "{reference_table}"("{reference_column}")'
        )

    columns_sql = ",\n".join(column_definitions)

    return f"""
        CREATE TABLE "{table_name}" (
            {columns_sql}
        )
    """


# ==================================================
# 11. CREATE SQLITE DATABASE
# ==================================================

if DATABASE_PATH.exists():
    DATABASE_PATH.unlink()

connection = sqlite3.connect(DATABASE_PATH)

try:

    connection.execute("PRAGMA foreign_keys = ON")

    print("\n--- CREATING DATABASE TABLES ---")

    # Customers table.

    customers_sql = create_table_sql(
        "customers",
        customers,
        primary_key="customer_id"
    )

    connection.execute(customers_sql)

    customers.to_sql(
        "customers",
        connection,
        if_exists="append",
        index=False
    )

    print("Created customers table:", len(customers))


    # Products table.

    products_sql = create_table_sql(
        "products",
        products,
        primary_key="product_id"
    )

    connection.execute(products_sql)

    products.to_sql(
        "products",
        connection,
        if_exists="append",
        index=False
    )

    print("Created products table:", len(products))


    # Orders table with foreign keys.

    orders_sql = create_table_sql(
        "orders",
        orders,
        primary_key="order_id",
        foreign_keys=[
            ("customer_id", "customers", "customer_id"),
            ("product_id", "products", "product_id")
        ]
    )

    connection.execute(orders_sql)

    orders.to_sql(
        "orders",
        connection,
        if_exists="append",
        index=False
    )

    print("Created orders table:", len(orders))


    # ==================================================
    # 12. CREATE COMPLETED ORDERS VIEW
    # ==================================================

    connection.execute("""
        CREATE VIEW completed_orders AS
        SELECT *
        FROM orders
        WHERE status = 'completed'
    """)

    print("Created completed_orders view")


    # ==================================================
    # 13. CREATE INDEXES
    # ==================================================

    connection.execute("""
        CREATE INDEX idx_orders_customer_id
        ON orders(customer_id)
    """)

    connection.execute("""
        CREATE INDEX idx_orders_product_id
        ON orders(product_id)
    """)

    connection.execute("""
        CREATE INDEX idx_orders_status
        ON orders(status)
    """)

    connection.execute("""
        CREATE INDEX idx_orders_timestamp
        ON orders(order_ts)
    """)

    print("Database indexes created")


    # ==================================================
    # 14. DATABASE VALIDATION
    # ==================================================

    print("\n--- DATABASE VALIDATION ---")

    expected_tables = [
        "customers",
        "products",
        "orders"
    ]

    cursor = connection.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
    """)

    actual_tables = {
        row[0] for row in cursor.fetchall()
    }

    for table in expected_tables:

        assert table in actual_tables, (
            f"Missing database table: {table}"
        )

        print(f"{table} table: PASS")


    # Verify record counts.

    expected_counts = {
        "customers": len(customers),
        "products": len(products),
        "orders": len(orders)
    }

    for table, expected_count in expected_counts.items():

        cursor.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        )

        actual_count = cursor.fetchone()[0]

        assert actual_count == expected_count, (
            f"Record count mismatch in {table}"
        )

        print(
            f"{table} records: {actual_count} | PASS"
        )


    # Verify primary keys.

    for table, primary_key in [
        ("customers", "customer_id"),
        ("products", "product_id"),
        ("orders", "order_id")
    ]:

        cursor.execute(
            f'PRAGMA table_info("{table}")'
        )

        columns = cursor.fetchall()

        primary_key_columns = [
            column[1]
            for column in columns
            if column[5] == 1
        ]

        assert primary_key in primary_key_columns, (
            f"Primary key missing in {table}"
        )

        print(
            f"{table} primary key: {primary_key} | PASS"
        )


    # Verify foreign keys.

    cursor.execute("""
        PRAGMA foreign_key_list(orders)
    """)

    foreign_key_results = cursor.fetchall()

    assert len(foreign_key_results) == 2, (
        "Expected two foreign keys in orders table."
    )

    print("Orders foreign keys: PASS")


    # Verify foreign-key integrity.

    cursor.execute("""
        PRAGMA foreign_key_check
    """)

    foreign_key_errors = cursor.fetchall()

    assert len(foreign_key_errors) == 0, (
        f"Foreign-key violations: {foreign_key_errors}"
    )

    print("Foreign-key integrity: PASS")


    # Verify completed orders view.

    cursor.execute("""
        SELECT COUNT(*)
        FROM completed_orders
    """)

    completed_count = cursor.fetchone()[0]

    expected_completed = int(
        (orders["status"] == "completed").sum()
    )

    assert completed_count == expected_completed

    print(
        f"Completed orders view: {completed_count} | PASS"
    )


    # Verify timestamp storage format.

    cursor.execute("""
        SELECT order_ts
        FROM orders
        LIMIT 1
    """)

    timestamp_value = cursor.fetchone()[0]

    assert len(timestamp_value) == 19, (
        "Unexpected order timestamp format."
    )

    print("Order timestamp format: PASS")


    # Commit database changes.

    connection.commit()

finally:

    connection.close()


# ==================================================
# 15. PRESERVE BACKWARD COMPATIBILITY
# ==================================================

shutil.copy2(
    DATABASE_PATH,
    COMPATIBILITY_DATABASE_PATH
)


# ==================================================
# 16. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("       DATABASE CREATION COMPLETED")
print("=" * 65)

print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))

print("\nCanonical database:")
print(DATABASE_PATH)

print("\nCompatibility database:")
print(COMPATIBILITY_DATABASE_PATH)

print("\nCleaned CSV files:")
print(CUSTOMERS_OUTPUT)
print(PRODUCTS_OUTPUT)
print(ORDERS_OUTPUT)

print("\nALL DATABASE VALIDATIONS PASSED!")