import sqlite3

db = sqlite3.connect("data/processed/instamart.db")

query = """
WITH historical_customers AS (
    SELECT DISTINCT customer_id
    FROM completed_orders
    WHERE order_ts < '2025-08-01'
),

future_customers AS (
    SELECT DISTINCT customer_id
    FROM completed_orders
    WHERE order_ts >= '2025-08-01'
      AND order_ts < '2025-10-01'
)

SELECT
    CASE
        WHEN f.customer_id IS NULL THEN 1
        ELSE 0
    END AS lapsed,
    COUNT(*) AS customer_count
FROM historical_customers h
LEFT JOIN future_customers f
    ON h.customer_id = f.customer_id
GROUP BY lapsed
ORDER BY lapsed;
"""

results = db.execute(query).fetchall()

print("\n--- LAPSE TARGET DISTRIBUTION ---")

total = sum(row[1] for row in results)

for label, count in results:
    description = "Lapsed" if label == 1 else "Active"
    percentage = count / total * 100

    print(
        f"{description}: {count} customers "
        f"({percentage:.2f}%)"
    )

print("\nTotal eligible customers:", total)

db.close()