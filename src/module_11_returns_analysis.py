from pathlib import Path
import logging
import pandas as pd
import numpy as np

# ============================================================
# MODULE 11: RETURNS ANALYSIS
# Project: NorthStar Retail Analytics Platform
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRANSACTIONS_FILE = (
    PROJECT_DIR / "outputs" / "integrated" / "integrated_transactions.csv"
)

RETURNS_FILE = PROJECT_DIR / "DATA" / "raw" / "returns.csv"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "module_11"
REPORT_DIR = PROJECT_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


def find_column(data, candidates):
    """Find a column without depending on letter case or underscores."""
    column_map = {
        str(column).strip().lower().replace("_", "").replace(" ", ""): column
        for column in data.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower().replace("_", "").replace(" ", "")
        if key in column_map:
            return column_map[key]

    return None


def save_csv(data, filename):
    """Save the analysis results."""
    file_path = OUTPUT_DIR / filename
    data.to_csv(file_path, index=False)
    logging.info("Saved: %s", file_path.name)


def main():

    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Integrated transactions file not found:\n{TRANSACTIONS_FILE}"
        )

    returns_path = RETURNS_FILE

    if not returns_path.exists():
        matches = [
            path for path in PROJECT_DIR.rglob("returns.csv")
            if ".venv" not in path.parts
            and "module_11" not in str(path).lower()
        ]

        if not matches:
            raise FileNotFoundError(
                "returns.csv was not found in the project directory."
            )

        raw_matches = [
            path for path in matches
            if "raw" in str(path).lower()
        ]

        returns_path = raw_matches[0] if raw_matches else matches[0]

    transactions = pd.read_csv(
        TRANSACTIONS_FILE,
        low_memory=False
    )

    returns = pd.read_csv(
        returns_path,
        low_memory=False
    )

    if transactions.empty or returns.empty:
        raise ValueError("One or both datasets are empty.")

    logging.info("Transactions loaded: %s", len(transactions))
    logging.info("Return records loaded: %s", len(returns))
    logging.info("Returns file: %s", returns_path)

    # --------------------------------------------------------
    # STEP 2: IDENTIFY DATASET COLUMNS
    # --------------------------------------------------------

    product_col = find_column(
        transactions,
        ["ProductID", "ProductId", "Product_ID", "SKU", "ItemID"]
    )

    quantity_col = find_column(
        transactions,
        ["Quantity", "Qty", "UnitsSold", "SoldQuantity"]
    )

    sales_col = find_column(
        transactions,
        ["LineAmount", "SalesAmount", "Revenue", "TotalAmount", "NetSales"]
    )

    category_col = find_column(
        transactions,
        ["Category", "ProductCategory", "Product_Category"]
    )

    channel_col = find_column(
        transactions,
        ["Channel", "SalesChannel", "Sales_Channel", "OrderChannel"]
    )

    transaction_id_col = find_column(
        transactions,
        ["TransactionID", "TransactionId", "OrderID", "OrderId",
         "InvoiceID", "InvoiceNo"]
    )

    return_product_col = find_column(
        returns,
        ["ProductID", "ProductId", "Product_ID", "SKU", "ItemID"]
    )

    return_transaction_col = find_column(
        returns,
        ["TransactionID", "TransactionId", "OrderID", "OrderId",
         "InvoiceID", "InvoiceNo"]
    )

    return_quantity_col = find_column(
        returns,
        ["ReturnQuantity", "ReturnedQuantity", "QtyReturned", "Quantity", "Qty"]
    )

    reason_col = find_column(
        returns,
        ["ReturnReason", "Reason", "Return_Reason", "ReasonForReturn"]
    )

    refund_col = find_column(
        returns,
        ["RefundAmount", "RefundedAmount", "ReturnAmount", "Refund",
         "AmountRefunded"]
    )

    if product_col is None or quantity_col is None:
        raise ValueError(
            "ProductID or Quantity was not identified in transactions.\n"
            f"Available columns: {list(transactions.columns)}"
        )

    if return_product_col is None and return_transaction_col is None:
        raise ValueError(
            "ProductID or TransactionID was not identified in returns.csv.\n"
            f"Available columns: {list(returns.columns)}"
        )

    # --------------------------------------------------------
    # STEP 3: CLEAN AND STANDARDISE DATA
    # --------------------------------------------------------

    sales = transactions.copy()
    returned = returns.copy()

    sales["_ProductKey"] = (
        sales[product_col].astype("string").str.strip()
    )

    sales["_SoldQuantity"] = pd.to_numeric(
        sales[quantity_col],
        errors="coerce"
    )

    if sales_col:
        sales["_SalesAmount"] = pd.to_numeric(
            sales[sales_col],
            errors="coerce"
        )
    else:
        sales["_SalesAmount"] = np.nan

    if return_product_col:
        returned["_ProductKey"] = (
            returned[return_product_col].astype("string").str.strip()
        )
    else:
        returned["_ProductKey"] = pd.NA

    if transaction_id_col:
        sales["_TransactionKey"] = (
            sales[transaction_id_col].astype("string").str.strip()
        )
    else:
        sales["_TransactionKey"] = pd.NA

    if return_transaction_col:
        returned["_TransactionKey"] = (
            returned[return_transaction_col].astype("string").str.strip()
        )
    else:
        returned["_TransactionKey"] = pd.NA

    if return_quantity_col:
        returned["_ReturnedQuantity"] = pd.to_numeric(
            returned[return_quantity_col],
            errors="coerce"
        )
    else:
        returned["_ReturnedQuantity"] = np.nan

    if refund_col:
        returned["_RefundAmount"] = pd.to_numeric(
            returned[refund_col],
            errors="coerce"
        )
    else:
        returned["_RefundAmount"] = np.nan

    returned["_ReturnRecord"] = 1

    # Retain records that have at least one usable matching key.
    usable_key = returned["_ProductKey"].notna()

    if return_transaction_col:
        usable_key = usable_key | returned["_TransactionKey"].notna()

    returned = returned.loc[usable_key].copy()

    # --------------------------------------------------------
    # STEP 4: DATA QUALITY AUDIT
    # --------------------------------------------------------

    quality_summary = pd.DataFrame({
        "Metric": [
            "Transaction rows",
            "Original return rows",
            "Usable return rows",
            "Duplicate transaction rows",
            "Duplicate return rows",
            "Missing product IDs in transactions",
            "Return rows with missing quantity",
            "Return rows with missing refund amount"
        ],
        "Value": [
            len(sales),
            len(returns),
            len(returned),
            int(transactions.duplicated().sum()),
            int(returns.duplicated().sum()),
            int(sales[product_col].isna().sum()),
            (
                int(returned["_ReturnedQuantity"].isna().sum())
                if return_quantity_col else len(returned)
            ),
            int(returned["_RefundAmount"].isna().sum())
        ]
    })

    save_csv(
        quality_summary,
        "returns_data_quality_summary.csv"
    )

    # --------------------------------------------------------
    # STEP 5: PRODUCT-LEVEL RETURNS ANALYSIS
    # --------------------------------------------------------

    sales_by_product = (
        sales.groupby("_ProductKey", dropna=False)
        .agg(
            Units_Sold=("_SoldQuantity", "sum"),
            Sales_Revenue=("_SalesAmount", "sum"),
            Sales_Rows=("_ProductKey", "size")
        )
        .reset_index()
    )

    product_returns = returned[
        returned["_ProductKey"].notna()
    ].copy()

    returns_by_product = (
        product_returns.groupby("_ProductKey", dropna=False)
        .agg(
            Return_Records=("_ReturnRecord", "sum"),
            Returned_Units=("_ReturnedQuantity", "sum"),
            Refund_Amount=("_RefundAmount", "sum")
        )
        .reset_index()
    )

    product_analysis = sales_by_product.merge(
        returns_by_product,
        on="_ProductKey",
        how="left"
    )

    for column in [
        "Return_Records",
        "Returned_Units",
        "Refund_Amount"
    ]:
        product_analysis[column] = product_analysis[column].fillna(0)

    if return_quantity_col:
        product_analysis["Return_Rate_Percent"] = np.where(
            product_analysis["Units_Sold"] > 0,
            product_analysis["Returned_Units"]
            / product_analysis["Units_Sold"] * 100,
            np.nan
        )
    else:
        product_analysis["Return_Rate_Percent"] = np.nan

    product_analysis = product_analysis.rename(
        columns={"_ProductKey": "ProductID"}
    )

    # Attach category without duplicating product-level sales.
    if category_col:
        product_categories = (
            sales[[product_col, category_col]]
            .dropna(subset=[product_col])
            .drop_duplicates(subset=[product_col])
            .rename(columns={product_col: "ProductID"})
        )

        product_analysis = product_analysis.merge(
            product_categories,
            on="ProductID",
            how="left"
        )

    save_csv(
        product_analysis,
        "product_returns_analysis.csv"
    )

    # --------------------------------------------------------
    # STEP 6: CATEGORY-LEVEL RETURNS ANALYSIS
    # --------------------------------------------------------

    if category_col and category_col in product_analysis.columns:

        category_analysis = (
            product_analysis.groupby(category_col, dropna=False)
            .agg(
                Products=("ProductID", "nunique"),
                Units_Sold=("Units_Sold", "sum"),
                Returned_Units=("Returned_Units", "sum"),
                Return_Records=("Return_Records", "sum"),
                Sales_Revenue=("Sales_Revenue", "sum"),
                Refund_Amount=("Refund_Amount", "sum")
            )
            .reset_index()
        )

        category_analysis["Return_Rate_Percent"] = np.where(
            category_analysis["Units_Sold"] > 0,
            category_analysis["Returned_Units"]
            / category_analysis["Units_Sold"] * 100,
            np.nan
        )

        save_csv(
            category_analysis,
            "category_returns_analysis.csv"
        )

    else:
        logging.warning(
            "Category analysis skipped because no category column was found."
        )

    # --------------------------------------------------------
    # STEP 7: RETURN REASONS ANALYSIS
    # --------------------------------------------------------

    if reason_col:

        reason_analysis = (
            returned.groupby(reason_col, dropna=False)
            .agg(
                Return_Records=("_ReturnRecord", "sum"),
                Returned_Units=("_ReturnedQuantity", "sum"),
                Refund_Amount=("_RefundAmount", "sum")
            )
            .reset_index()
            .sort_values("Return_Records", ascending=False)
        )

        save_csv(
            reason_analysis,
            "return_reasons_analysis.csv"
        )

    else:
        logging.warning(
            "Return reason analysis skipped because no reason column was found."
        )

    # --------------------------------------------------------
    # STEP 8: SALES CHANNEL RETURNS ANALYSIS
    # --------------------------------------------------------

    if channel_col:

        if transaction_id_col and return_transaction_col:

            channel_lookup = (
                sales[sales["_TransactionKey"].notna()]
                .groupby("_TransactionKey", dropna=False)
                .agg(
                    Channel=(channel_col, "first"),
                    Units_Sold=("_SoldQuantity", "sum"),
                    Sales_Revenue=("_SalesAmount", "sum")
                )
                .reset_index()
            )

            returns_with_channel = returned[
                returned["_TransactionKey"].notna()
            ].merge(
                channel_lookup[["_TransactionKey", "Channel"]],
                on="_TransactionKey",
                how="left"
            )

            # Exclude returns whose transaction could not be matched.
            matched_returns = returns_with_channel[
                returns_with_channel["Channel"].notna()
            ].copy()

            channel_returns = (
                matched_returns.groupby("Channel", dropna=False)
                .agg(
                    Return_Records=("_ReturnRecord", "sum"),
                    Returned_Units=("_ReturnedQuantity", "sum"),
                    Refund_Amount=("_RefundAmount", "sum")
                )
                .reset_index()
            )

            channel_sales = (
                sales.groupby(channel_col, dropna=False)
                .agg(
                    Units_Sold=("_SoldQuantity", "sum"),
                    Sales_Revenue=("_SalesAmount", "sum")
                )
                .reset_index()
                .rename(columns={channel_col: "Channel"})
            )

            channel_analysis = channel_sales.merge(
                channel_returns,
                on="Channel",
                how="left"
            )

            for column in [
                "Return_Records",
                "Returned_Units",
                "Refund_Amount"
            ]:
                channel_analysis[column] = (
                    channel_analysis[column].fillna(0)
                )

            if return_quantity_col:
                channel_analysis["Return_Rate_Percent"] = np.where(
                    channel_analysis["Units_Sold"] > 0,
                    channel_analysis["Returned_Units"]
                    / channel_analysis["Units_Sold"] * 100,
                    np.nan
                )
            else:
                channel_analysis["Return_Rate_Percent"] = np.nan

            save_csv(
                channel_analysis,
                "channel_returns_analysis.csv"
            )

        else:
            channel_analysis = pd.DataFrame({
                "Status": [
                    "Channel return analysis requires transaction IDs "
                    "in both datasets."
                ]
            })

            save_csv(
                channel_analysis,
                "channel_returns_analysis.csv"
            )

    else:
        logging.warning(
            "Channel analysis skipped because no channel column was found."
        )

    # --------------------------------------------------------
    # STEP 9: QUANTIFY RETURN COSTS
    # --------------------------------------------------------

    if returned["_RefundAmount"].notna().any():
        total_refunds = float(returned["_RefundAmount"].sum())
        refund_note = "Sum of recorded refund amounts."
    else:
        total_refunds = np.nan
        refund_note = (
            "Unavailable: no usable refund amount values were found."
        )

    if returned["_ReturnedQuantity"].notna().any():
        total_returned_units = float(
            returned["_ReturnedQuantity"].sum(min_count=1)
        )
        quantity_note = "Sum of recorded returned quantities."
    else:
        total_returned_units = np.nan
        quantity_note = (
            "Unavailable: no usable return quantity values were found."
        )

    cost_summary = pd.DataFrame({
        "Metric": [
            "Total return records",
            "Total returned units",
            "Total recorded refunds"
        ],
        "Value": [
            len(returned),
            total_returned_units,
            total_refunds
        ],
        "Calculation_Method": [
            "Count of usable return records",
            quantity_note,
            refund_note
        ]
    })

    save_csv(
        cost_summary,
        "return_cost_summary.csv"
    )

    # --------------------------------------------------------
    # STEP 10: SAVE CLEANED RETURN DATA
    # --------------------------------------------------------

    cleaned_returns = returned.drop(
        columns=[
            "_ProductKey",
            "_TransactionKey",
            "_ReturnRecord"
        ],
        errors="ignore"
    )

    save_csv(
        cleaned_returns,
        "returns_cleaned_for_analysis.csv"
    )

    # --------------------------------------------------------
    # STEP 11: GENERATE BUSINESS REPORT
    # --------------------------------------------------------

    report_lines = [
        "NORTHSTAR RETAIL ANALYTICS PLATFORM",
        "MODULE 11: RETURNS ANALYSIS",
        "=" * 55,
        "",
        f"Transaction rows analysed: {len(sales):,}",
        f"Original return records: {len(returns):,}",
        f"Usable return records: {len(returned):,}",
        f"Returns source: {returns_path}",
        "",
        "ANALYSIS PERFORMED",
        "1. Product-level returns and return rates where quantities exist.",
        "2. Category-level returns analysis where category data exists.",
        "3. Return reasons analysis where reason data exists.",
        "4. Channel analysis where transactions can be matched.",
        "5. Recorded refund summary where financial values exist.",
        "6. Data-quality checks and cleaned return data.",
        "",
        "BUSINESS RECOMMENDATIONS",
        "Review products with high return rates for quality or description issues.",
        "Review frequent return reasons and prioritise corrective actions.",
        "Compare channel results only when transaction matching is reliable.",
        "Use verified refund and cost data to assess financial impact.",
        "",
        "DATA LIMITATIONS",
        "Return rates require comparable sold and returned quantities.",
        "Channel analysis requires matching transaction IDs.",
        "Refund totals do not necessarily represent the full cost of returns.",
        "Missing financial values are reported as unavailable.",
        "",
        f"Output folder: {OUTPUT_DIR}"
    ]

    report_path = REPORT_DIR / "module_11_returns_analysis.txt"

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # STEP 12: VERIFY OUTPUT FILES
    # --------------------------------------------------------

    required_files = [
        OUTPUT_DIR / "returns_data_quality_summary.csv",
        OUTPUT_DIR / "product_returns_analysis.csv",
        OUTPUT_DIR / "return_cost_summary.csv",
        OUTPUT_DIR / "returns_cleaned_for_analysis.csv",
        report_path
    ]

    missing_files = [
        str(file) for file in required_files if not file.exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Output verification failed:\n" + "\n".join(missing_files)
        )

    print("\n" + "=" * 60)
    print("MODULE 11: RETURNS ANALYSIS COMPLETED")
    print("=" * 60)
    print("Project: NorthStar Retail Analytics Platform")
    print(f"Transaction rows: {len(sales):,}")
    print(f"Original return records: {len(returns):,}")
    print(f"Usable return records: {len(returned):,}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Business report: {report_path}")
    print("Status: OUTPUT VERIFICATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()