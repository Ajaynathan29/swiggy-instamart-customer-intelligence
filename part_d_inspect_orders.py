import sqlite3

DB_PATH = "data/processed/instamart.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("--- ORDERS TABLE SCHEMA ---")

cursor.execute("PRAGMA table_info(orders)")

for column in cursor.fetchall():
    print(column)

print("\n--- REFUND AGENT TEST ORDERS ---")

order_ids = [
    "ORD-100048",
    "ORD-100067",
    "ORD-100145",
    "ORD-100001",
    "ORD-100025"
]

placeholders = ",".join("?" for _ in order_ids)

query = f"""
SELECT *
FROM orders
WHERE order_id IN ({placeholders})
"""

cursor.execute(query, order_ids)

columns = [description[0] for description in cursor.description]

print(columns)

for row in cursor.fetchall():
    print(dict(zip(columns, row)))

conn.close()