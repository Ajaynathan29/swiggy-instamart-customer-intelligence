import sqlite3

db = sqlite3.connect("data/processed/instamart.db")

queries = {
    "SIGNUP DATE RANGE": """
        SELECT MIN(signup_date), MAX(signup_date)
        FROM customers
    """,

    "CUSTOMERS SIGNED UP BY JULY 31": """
        SELECT COUNT(*)
        FROM customers
        WHERE signup_date <= '2025-07-31'
    """,

    "CUSTOMERS WITH HISTORICAL COMPLETED ORDERS": """
        SELECT COUNT(DISTINCT customer_id)
        FROM completed_orders
        WHERE order_ts < '2025-08-01'
    """,

    "CUSTOMERS WITH FUTURE-WINDOW COMPLETED ORDERS": """
        SELECT COUNT(DISTINCT customer_id)
        FROM completed_orders
        WHERE order_ts >= '2025-08-01'
          AND order_ts < '2025-10-01'
    """,

    "ELIGIBLE CUSTOMERS WITH HISTORICAL ORDERS": """
        SELECT COUNT(DISTINCT c.customer_id)
        FROM customers c
        JOIN completed_orders o
          ON c.customer_id = o.customer_id
        WHERE c.signup_date <= '2025-07-31'
          AND o.order_ts < '2025-08-01'
    """
}

for title, query in queries.items():
    print(f"\n--- {title} ---")
    print(db.execute(query).fetchone())

db.close()