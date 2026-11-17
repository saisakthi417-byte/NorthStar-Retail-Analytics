
from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt

# MODULE 6: EXPLORATORY DATA ANALYSIS
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = BASE_DIR / "outputs" / "integrated" / "integrated_transactions.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "module_06"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def summarize_group(data, group_column, output_name):
    if group_column not in data.columns:
        logging.warning("Column not found: %s", group_column)
        return None

    work = data.dropna(subset=[group_column, "LineAmount"]).copy()

    if work.empty:
        return None

    summary = (
        work.groupby(group_column)
        .agg(
            Transaction_Rows=("LineAmount", "size"),
            Total_Sales=("LineAmount", "sum"),
            Average_Line_Amount=("LineAmount", "mean"),
        )
        .sort_values("Total_Sales", ascending=False)
        .reset_index()
    )

    summary.to_csv(OUTPUT_DIR / output_name, index=False)
    return summary


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Integrated dataset not found: {INPUT_FILE}. Run Module 4 first."
        )

    # 1. Load integrated dataset
    df = pd.read_csv(INPUT_FILE)
    logging.info(
        "Loaded integrated data: %s rows, %s columns",
        len(df), len(df.columns)
    )

    required_columns = ["OrderDate", "LineAmount", "Channel", "StoreID"]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # 2. Convert date and numeric columns
    df["OrderDate"] = pd.to_datetime(
        df["OrderDate"],
        format="%Y-%m-%d",
        errors="coerce"
    )

    df["LineAmount"] = pd.to_numeric(
        df["LineAmount"], errors="coerce"
    )

    for col in ["Discount", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Standardize text labels BEFORE grouping
    for col in ["Channel", "Region", "Category", "StoreID"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    if "Channel" in df.columns:
        df["Channel"] = df["Channel"].str.title()

    if "Region" in df.columns:
        df["Region"] = df["Region"].str.title()

    if "Category" in df.columns:
        df["Category"] = df["Category"].str.title()

    if "StoreID" in df.columns:
        df["StoreID"] = df["StoreID"].str.upper()

    # 4. Validate sales and dates
    valid_sales = df.dropna(subset=["LineAmount"]).copy()

    valid_dates_sales = valid_sales.dropna(
        subset=["OrderDate"]
    ).copy()

    logging.info("Rows with valid sales amounts: %s", len(valid_sales))
    logging.info(
        "Rows with valid dates and sales amounts: %s",
        len(valid_dates_sales)
    )

    # 5. Sales by channel
    channel_summary = summarize_group(
        valid_sales, "Channel", "sales_by_channel.csv"
    )

    # 6. Sales by category
    category_summary = summarize_group(
        valid_sales, "Category", "sales_by_category.csv"
    )

    # 7. Top 10 physical stores; exclude ONLINE
    physical_stores = valid_sales[
        valid_sales["StoreID"].notna()
        & valid_sales["StoreID"].str.upper().ne("ONLINE")
    ].copy()

    top_stores = (
        physical_stores.dropna(subset=["StoreID", "LineAmount"])
        .groupby("StoreID")
        .agg(
            Transaction_Rows=("LineAmount", "size"),
            Total_Sales=("LineAmount", "sum"),
            Average_Line_Amount=("LineAmount", "mean"),
        )
        .sort_values("Total_Sales", ascending=False)
        .head(10)
        .reset_index()
    )

    top_stores.to_csv(
        OUTPUT_DIR / "top_10_physical_stores.csv",
        index=False
    )

    # 8. Sales by region
    region_summary = summarize_group(
        valid_sales, "Region", "sales_by_region.csv"
    )

    # 9. Sales by discount
    if "Discount" in valid_sales.columns:
        discount_summary = (
            valid_sales.dropna(subset=["Discount"])
            .groupby("Discount")
            .agg(
                Transaction_Rows=("LineAmount", "size"),
                Total_Sales=("LineAmount", "sum"),
                Average_Line_Amount=("LineAmount", "mean"),
            )
            .sort_index()
            .reset_index()
        )

        discount_summary.to_csv(
            OUTPUT_DIR / "sales_by_discount.csv",
            index=False
        )

    # 10. Monthly sales trend
    monthly_data = valid_dates_sales.copy()

    monthly_data["Month"] = (
        monthly_data["OrderDate"].dt.to_period("M").astype(str)
    )

    monthly_summary = (
        monthly_data.groupby("Month")
        .agg(
            Transaction_Rows=("LineAmount", "size"),
            Total_Sales=("LineAmount", "sum"),
            Average_Line_Amount=("LineAmount", "mean"),
        )
        .reset_index()
        .sort_values("Month")
    )

    monthly_summary.to_csv(
        OUTPUT_DIR / "monthly_sales_trend.csv",
        index=False
    )

    # 11. Save monthly sales chart
    if not monthly_summary.empty:
        plt.figure(figsize=(12, 6))

        plt.plot(
            monthly_summary["Month"],
            monthly_summary["Total_Sales"],
            marker="o"
        )

        plt.title("NorthStar Retail - Monthly Sales Trend")
        plt.xlabel("Month")
        plt.ylabel("Total LineAmount")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "monthly_sales_trend.png",
            dpi=150
        )

        plt.close()

    # 12. Save analysis-ready transactions
    valid_dates_sales.to_csv(
        OUTPUT_DIR / "eda_valid_transactions.csv",
        index=False
    )

    # 13. Generate summary report
    report_lines = [
        "NORTHSTAR RETAIL ANALYTICS PLATFORM",
        "MODULE 6 - EXPLORATORY DATA ANALYSIS",
        "",
        f"Integrated transaction rows: {len(df):,}",
        f"Rows with valid LineAmount: {len(valid_sales):,}",
        f"Rows with valid OrderDate and LineAmount: {len(valid_dates_sales):,}",
        f"Missing/invalid dates: {df['OrderDate'].isna().sum():,}",
        f"Missing/invalid LineAmount: {df['LineAmount'].isna().sum():,}",
        f"Total LineAmount: {valid_sales['LineAmount'].sum():,.2f}",
        "",
        "SALES BY CHANNEL",
        channel_summary.to_string(index=False)
        if channel_summary is not None else "No channel data available.",
        "",
        "SALES BY CATEGORY",
        category_summary.to_string(index=False)
        if category_summary is not None else "No category data available.",
        "",
        "TOP 10 PHYSICAL STORES (ONLINE EXCLUDED)",
        top_stores.to_string(index=False),
        "",
        "SALES BY REGION",
        region_summary.to_string(index=False)
        if region_summary is not None else "No region data available.",
        "",
        "MONTHLY SALES TREND",
        monthly_summary.to_string(index=False),
        "",
        "INTERPRETATION NOTES",
        "LineAmount is analyzed as supplied; confirm its business definition.",
        "ONLINE transactions are excluded from the physical-store ranking.",
        "Missing region values are not automatically assigned to a region.",
        "Unmatched product attributes must not be invented.",
        "High sales totals alone do not establish profitability or causation."
    ]

    report_path = REPORT_DIR / "module_06_eda_summary.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    logging.info("MODULE 6 ANALYSIS GENERATED")
    logging.info("Input rows: %s", len(df))
    logging.info("Output folder: %s", OUTPUT_DIR)
    logging.info("Summary report: %s", report_path)


if __name__ == "__main__":
    main()
