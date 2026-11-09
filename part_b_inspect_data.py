import sqlite3
from pathlib import Path


# --------------------------------------------------
# 1. DATABASE PATH
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "processed" / "instamart.db"

db = sqlite3.connect(DB_PATH)


# --------------------------------------------------
# 2. SQL QUERIES
# --------------------------------------------------

queries = {

    "TABLE ROW COUNTS": """
        SELECT 'customers', COUNT(*) FROM customers

        UNION ALL

        SELECT 'products', COUNT(*) FROM products

        UNION ALL

        SELECT 'orders', COUNT(*) FROM orders

        UNION ALL

        SELECT 'completed_orders', COUNT(*) FROM completed_orders
    """,

    "ORDER DATE RANGE": """
        SELECT
            MIN(order_ts),
            MAX(order_ts)
        FROM orders
    """,

    "ORDER STATUS DISTRIBUTION": """
        SELECT
            status,
            COUNT(*)
        FROM orders
        GROUP BY status
        ORDER BY status
    """,

    "DISTINCT CUSTOMERS WITH COMPLETED ORDERS": """
        SELECT COUNT(DISTINCT customer_id)
        FROM completed_orders
    """,

    "CUSTOMERS WITHOUT COMPLETED ORDERS": """
        SELECT COUNT(*)
        FROM customers c
        WHERE NOT EXISTS (
            SELECT 1
            FROM completed_orders o
            WHERE o.customer_id = c.customer_id
        )
    """,

    "COMPLETED ORDERS BY MONTH": """
        SELECT
            SUBSTR(order_ts, 1, 7) AS order_month,
            COUNT(*) AS completed_orders
        FROM completed_orders
        GROUP BY SUBSTR(order_ts, 1, 7)
        ORDER BY order_month
    """,

    "CUSTOMER COHORT BEFORE AUGUST 2025": """
        SELECT COUNT(DISTINCT customer_id)
        FROM completed_orders
        WHERE order_ts < '2025-08-01 00:00:00'
    """,

    "CUSTOMERS WITH COMPLETED ORDERS BEFORE AUGUST 2025": """
        SELECT COUNT(DISTINCT customer_id)
        FROM completed_orders
        WHERE order_ts < '2025-08-01 00:00:00'
    """
}


# --------------------------------------------------
# 3. EXECUTE QUERIES
# --------------------------------------------------

try:

    print("=" * 70)
    print("INSTAMART DATABASE INSPECTION")
    print("=" * 70)

    for title, query in queries.items():

        print(f"\n--- {title} ---")

        results = db.execute(query).fetchall()

        for row in results:
            print(row)

    print("\n" + "=" * 70)
    print("DATABASE INSPECTION COMPLETED")
    print("=" * 70)

finally:
    db.close()