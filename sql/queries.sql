-- Instamart Customer Intelligence & Support Copilot
-- Seven analytical SQL queries
-- Database: SQLite

-- Q1. Top 30 customers by completed-order spending
-- Concepts: WHERE, GROUP BY, HAVING, ORDER BY, LIMIT

SELECT
    c.customer_id,
    c.city,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS total_spend_inr
FROM customers c
JOIN orders o
    ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.city
HAVING SUM(o.order_amount) > 0
ORDER BY total_spend_inr DESC
LIMIT 30;


-- Q2. Completed revenue by city and product category
-- Concepts: Three-table JOIN, GROUP BY, aggregation

SELECT
    c.city,
    p.category,
    COUNT(o.order_id) AS completed_orders,
    ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, revenue_inr DESC;


-- Q3. Products priced above the average product price
-- Concepts: Scalar subquery

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4. Customers who have never completed an order
-- Concepts: LEFT JOIN, NULL filtering

SELECT
    c.customer_id,
    c.city,
    c.signup_date
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5. Customers whose completed spending is above average
-- Concepts: CTE, aggregation, scalar comparison

WITH customer_spending AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spending AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spending
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend_inr,
    ROUND(av.avg_spend, 2) AS average_spend_inr
FROM customer_spending cs
CROSS JOIN average_spending av
WHERE cs.total_spend > av.avg_spend
ORDER BY cs.total_spend DESC;


-- Q6. Highest-spending customer or customers in each city
-- Concepts: Window functions, RANK, ROW_NUMBER

WITH customer_city_spending AS (
    SELECT
        c.city,
        c.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
),
ranked_customers AS (
    SELECT
        city,
        customer_id,
        total_spend,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS spend_rank,
        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC, customer_id
        ) AS row_number
    FROM customer_city_spending
)
SELECT
    city,
    customer_id,
    ROUND(total_spend, 2) AS total_spend_inr,
    spend_rank,
    row_number
FROM ranked_customers
ORDER BY city, spend_rank, customer_id;


-- Q7. Daily completed revenue and cumulative revenue
-- Concepts: Date aggregation, window function, running total

WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_date,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_date,
    ROUND(daily_revenue, 2) AS daily_revenue_inr,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY order_date;
