
USE northstar_retail_analytics;

-- =====================================================
-- MODULE 14: NORTHSTAR RETAIL SQL ANALYSIS
-- =====================================================

-- 1. Revenue by product category
SELECT
    Category,
    COUNT(*) AS transaction_lines,
    SUM(Quantity) AS units_sold,
    ROUND(SUM(LineAmount), 2) AS total_revenue
FROM integrated_transactions
GROUP BY Category
ORDER BY total_revenue DESC;


-- 2. Sales channel performance using CASE
SELECT
    Channel,
    COUNT(*) AS transaction_lines,
    COUNT(DISTINCT CustomerID) AS unique_customers,
    ROUND(SUM(LineAmount), 2) AS total_revenue,
    CASE
        WHEN SUM(LineAmount) >= 1000000 THEN 'High'
        WHEN SUM(LineAmount) >= 500000 THEN 'Medium'
        ELSE 'Low'
    END AS revenue_category
FROM integrated_transactions
GROUP BY Channel
ORDER BY total_revenue DESC;


-- 3. Monthly revenue using a CTE
WITH monthly_sales AS (
    SELECT
        DATE_FORMAT(
            STR_TO_DATE(OrderDate, '%Y-%m-%d'),
            '%Y-%m'
        ) AS sales_month,
        SUM(LineAmount) AS monthly_revenue
    FROM integrated_transactions
    GROUP BY DATE_FORMAT(
        STR_TO_DATE(OrderDate, '%Y-%m-%d'),
        '%Y-%m'
    )
)
SELECT
    sales_month,
    ROUND(monthly_revenue, 2) AS monthly_revenue
FROM monthly_sales
ORDER BY sales_month;


-- 4. Store revenue ranking using RANK()
WITH store_sales AS (
    SELECT
        StoreID,
        StoreName,
        Region,
        SUM(LineAmount) AS total_revenue
    FROM integrated_transactions
    GROUP BY StoreID, StoreName, Region
)
SELECT
    StoreID,
    StoreName,
    Region,
    ROUND(total_revenue, 2) AS total_revenue,
    RANK() OVER (ORDER BY total_revenue DESC) AS store_rank
FROM store_sales
ORDER BY store_rank;


-- 5. Top 10 products using ROW_NUMBER()
WITH product_sales AS (
    SELECT
        ProductID,
        ProductName,
        Category,
        SUM(Quantity) AS units_sold,
        SUM(LineAmount) AS total_revenue
    FROM integrated_transactions
    GROUP BY ProductID, ProductName, Category
),
ranked_products AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            ORDER BY total_revenue DESC
        ) AS product_rank
    FROM product_sales
)
SELECT *
FROM ranked_products
WHERE product_rank <= 10
ORDER BY product_rank;


-- 6. Revenue by discount band using CASE
SELECT
    CASE
        WHEN Discount IS NULL OR Discount = 0
            THEN 'No Discount'
        WHEN Discount <= 0.10 THEN 'Low Discount'
        WHEN Discount <= 0.25 THEN 'Medium Discount'
        ELSE 'High Discount'
    END AS discount_band,
    COUNT(*) AS transaction_lines,
    ROUND(SUM(LineAmount), 2) AS total_revenue,
    ROUND(AVG(LineAmount), 2) AS average_line_amount
FROM integrated_transactions
GROUP BY discount_band
ORDER BY total_revenue DESC;


-- 7. Top 20 customers using DENSE_RANK()
WITH customer_sales AS (
    SELECT
        CustomerID,
        MAX(CustomerName) AS CustomerName,
        COUNT(*) AS transaction_lines,
        SUM(Quantity) AS units_purchased,
        SUM(LineAmount) AS total_spent
    FROM integrated_transactions
    GROUP BY CustomerID
),
ranked_customers AS (
    SELECT
        *,
        DENSE_RANK() OVER (
            ORDER BY total_spent DESC
        ) AS customer_rank
    FROM customer_sales
)
SELECT
    CustomerID,
    CustomerName,
    transaction_lines,
    units_purchased,
    ROUND(total_spent, 2) AS total_spent,
    customer_rank
FROM ranked_customers
WHERE customer_rank <= 20
ORDER BY customer_rank;


-- 8. Store performance using JOIN and CASE
SELECT
    t.StoreID,
    s.Region,
    COUNT(*) AS transaction_lines,
    ROUND(SUM(t.LineAmount), 2) AS total_revenue,
    CASE
        WHEN SUM(t.LineAmount) >= 1000000 THEN 'Top Performer'
        WHEN SUM(t.LineAmount) >= 500000 THEN 'Average Performer'
        ELSE 'Needs Review'
    END AS performance_status
FROM integrated_transactions AS t
LEFT JOIN ns_stores AS s
    ON t.StoreID = s.StoreID
GROUP BY t.StoreID, s.Region
ORDER BY total_revenue DESC;


-- 9. Regional performance
SELECT
    Region,
    COUNT(DISTINCT StoreID) AS number_of_stores,
    SUM(Quantity) AS units_sold,
    ROUND(SUM(LineAmount), 2) AS total_revenue,
    CASE
        WHEN SUM(LineAmount) >= 2000000 THEN 'Strong Region'
        WHEN SUM(LineAmount) >= 1000000 THEN 'Moderate Region'
        ELSE 'Developing Region'
    END AS regional_status
FROM integrated_transactions
GROUP BY Region
ORDER BY total_revenue DESC;


-- 10. Loyalty tier analysis using CTE and DENSE_RANK()
WITH loyalty_sales AS (
    SELECT
        LoyaltyTier,
        COUNT(DISTINCT CustomerID) AS unique_customers,
        COUNT(*) AS transaction_lines,
        SUM(LineAmount) AS total_revenue
    FROM integrated_transactions
    GROUP BY LoyaltyTier
)
SELECT
    LoyaltyTier,
    unique_customers,
    transaction_lines,
    ROUND(total_revenue, 2) AS total_revenue,
    DENSE_RANK() OVER (
        ORDER BY total_revenue DESC
    ) AS loyalty_rank
FROM loyalty_sales
ORDER BY loyalty_rank;


-- FINAL DATA VALIDATION
SELECT
    COUNT(*) AS total_transaction_lines,
    COUNT(DISTINCT CustomerID) AS unique_customers,
    COUNT(DISTINCT ProductID) AS unique_products,
    COUNT(DISTINCT StoreID) AS unique_stores,
    SUM(Quantity) AS total_units_sold,
    ROUND(SUM(LineAmount), 2) AS total_revenue
FROM integrated_transactions;

-- END OF MODULE 14
