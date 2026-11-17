USE northstar_retail_dw;

-- ==========================================
-- 1. POPULATE DATE DIMENSION
-- Grain: One row per calendar date
-- ==========================================

INSERT IGNORE INTO dim_date
(date_key, full_date, year_number, month_number, month_name, quarter_number)

SELECT DISTINCT
    DATE_FORMAT(STR_TO_DATE(OrderDate, '%Y-%m-%d'), '%Y%m%d') + 0,
    STR_TO_DATE(OrderDate, '%Y-%m-%d'),
    YEAR(STR_TO_DATE(OrderDate, '%Y-%m-%d')),
    MONTH(STR_TO_DATE(OrderDate, '%Y-%m-%d')),
    MONTHNAME(STR_TO_DATE(OrderDate, '%Y-%m-%d')),
    QUARTER(STR_TO_DATE(OrderDate, '%Y-%m-%d'))

FROM northstar_retail_analytics.integrated_transactions
WHERE STR_TO_DATE(OrderDate, '%Y-%m-%d') IS NOT NULL;


-- ==========================================
-- 2. POPULATE SALES FACT TABLE
-- Grain: One row per transaction line
-- ==========================================

DELETE FROM fact_sales;

INSERT INTO fact_sales
(
    transaction_id,
    date_key,
    customer_key,
    product_key,
    store_key,
    channel_key,
    quantity,
    discount,
    line_amount
)

SELECT
    t.TransactionID,

    DATE_FORMAT(
        STR_TO_DATE(t.OrderDate, '%Y-%m-%d'),
        '%Y%m%d'
    ) + 0,

    c.customer_key,
    p.product_key,
    s.store_key,
    ch.channel_key,

    t.Quantity,
    t.Discount,
    t.LineAmount

FROM northstar_retail_analytics.integrated_transactions t

LEFT JOIN dim_customer c
    ON t.CustomerID = c.customer_id

LEFT JOIN dim_product p
    ON t.ProductID = p.product_id

LEFT JOIN dim_store s
    ON t.StoreID = s.store_id

LEFT JOIN dim_channel ch
    ON t.Channel = ch.channel_name

WHERE STR_TO_DATE(t.OrderDate, '%Y-%m-%d') IS NOT NULL;


-- ==========================================
-- 3. VALIDATE WAREHOUSE TABLES
-- ==========================================

SELECT 'dim_date' AS table_name, COUNT(*) AS total_rows
FROM dim_date

UNION ALL

SELECT 'fact_sales', COUNT(*)
FROM fact_sales;

USE northstar_retail_dw;

-- MODULE 15: FINAL VALIDATION

SELECT COUNT(*) AS total_sales_records
FROM fact_sales;

SELECT
    COUNT(*) AS total_records,
    SUM(sales_amount) AS total_revenue,
    SUM(quantity) AS total_units_sold
FROM fact_sales;

-- Validate fact table relationships
SELECT
    COUNT(*) AS total_records,
    COUNT(d.date_key) AS matched_dates,
    COUNT(c.customer_key) AS matched_customers,
    COUNT(p.product_key) AS matched_products,
    COUNT(s.store_key) AS matched_stores,
    COUNT(ch.channel_key) AS matched_channels
FROM fact_sales f
LEFT JOIN dim_date d ON f.date_key = d.date_key
LEFT JOIN dim_customer c ON f.customer_key = c.customer_key
LEFT JOIN dim_product p ON f.product_key = p.product_key
LEFT JOIN dim_store s ON f.store_key = s.store_key
LEFT JOIN dim_channel ch ON f.channel_key = ch.channel_key;

DESCRIBE fact_sales;

USE northstar_retail_dw;

-- Final Sales Validation
SELECT
    COUNT(*) AS total_sales_records,
    ROUND(SUM(line_amount), 2) AS total_revenue,
    SUM(quantity) AS total_units_sold,
    ROUND(AVG(line_amount), 2) AS average_sales_value
FROM fact_sales;