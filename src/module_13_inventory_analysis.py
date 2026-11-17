# ============================================================
# NORTHSTAR RETAIL ANALYTICS PLATFORM
# MODULE 13: INVENTORY ANALYSIS
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

INVENTORY_FILE = ROOT / "DATA" / "raw" / "inventory.csv"
TRANSACTIONS_FILE = (
    ROOT / "outputs" / "integrated" / "integrated_transactions.csv"
)
PRODUCTS_FILE = ROOT / "DATA" / "raw" / "products.csv"

OUTPUT_DIR = ROOT / "outputs" / "module_13"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_csv(path, description):
    """Load a CSV file with basic validation."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} file not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"{description} dataset is empty.")

    print(f"{description} loaded: {len(df):,} rows")

    return df


def save_csv(df, path):
    """Save a DataFrame and verify the output."""

    df.to_csv(path, index=False)

    if not path.exists():
        raise RuntimeError(f"Output file was not created: {path}")

    check = pd.read_csv(path)

    if len(check) != len(df):
        raise RuntimeError(
            f"Output row count mismatch: {path}"
        )


# ============================================================
# 3. MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 70)
    print("NORTHSTAR RETAIL ANALYTICS PLATFORM")
    print("MODULE 13: INVENTORY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 3.1 Load inventory data
    # --------------------------------------------------------

    inventory = load_csv(INVENTORY_FILE, "Inventory")

    required_inventory_columns = [
        "SnapshotMonth",
        "StoreID",
        "ProductID",
        "OnHandUnits",
        "ReorderLevel",
        "StockoutFlag",
    ]

    missing_columns = [
        col for col in required_inventory_columns
        if col not in inventory.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing inventory columns: {missing_columns}"
        )

    # Transactions are used to identify slow-moving products.
    transactions = load_csv(
        TRANSACTIONS_FILE,
        "Integrated transactions",
    )

    required_transaction_columns = [
        "OrderDate",
        "ProductID",
        "StoreID",
        "Quantity",
    ]

    missing_transaction_columns = [
        col for col in required_transaction_columns
        if col not in transactions.columns
    ]

    if missing_transaction_columns:
        raise ValueError(
            "Missing transaction columns: "
            f"{missing_transaction_columns}"
        )

    # Product names are optional; ProductID is sufficient.
    products = None

    if PRODUCTS_FILE.exists():
        products = pd.read_csv(PRODUCTS_FILE)

    # --------------------------------------------------------
    # 3.2 Data cleaning and validation
    # --------------------------------------------------------

    inventory = inventory.copy()
    transactions = transactions.copy()

    inventory["SnapshotMonth"] = pd.to_datetime(
        inventory["SnapshotMonth"],
        errors="coerce",
    )

    inventory["OnHandUnits"] = pd.to_numeric(
        inventory["OnHandUnits"],
        errors="coerce",
    )

    inventory["ReorderLevel"] = pd.to_numeric(
        inventory["ReorderLevel"],
        errors="coerce",
    )

    inventory["StockoutFlag"] = pd.to_numeric(
        inventory["StockoutFlag"],
        errors="coerce",
    )

    transactions["OrderDate"] = pd.to_datetime(
        transactions["OrderDate"],
        errors="coerce",
    )

    transactions["Quantity"] = pd.to_numeric(
        transactions["Quantity"],
        errors="coerce",
    )

    inventory["StoreID"] = (
        inventory["StoreID"].astype("string").str.strip()
    )

    inventory["ProductID"] = (
        inventory["ProductID"].astype("string").str.strip()
    )

    transactions["StoreID"] = (
        transactions["StoreID"].astype("string").str.strip()
    )

    transactions["ProductID"] = (
        transactions["ProductID"].astype("string").str.strip()
    )

    missing_stock_before = int(
        inventory["OnHandUnits"].isna().sum()
    )

    invalid_negative_stock = int(
        (inventory["OnHandUnits"] < 0).sum()
    )

    invalid_negative_reorder = int(
        (inventory["ReorderLevel"] < 0).sum()
    )

    invalid_stockout_flags = int(
        (~inventory["StockoutFlag"].isin([0, 1])).sum()
    )

    # Do not replace unknown inventory quantities with zero.
    # Missing stock is unknown, not proof of zero stock.
    inventory["StockDataStatus"] = np.where(
        inventory["OnHandUnits"].isna(),
        "Missing Stock Quantity",
        "Available",
    )

    # Valid stockout flag is based on the supplied flag.
    inventory["StockoutFlag"] = (
        inventory["StockoutFlag"].fillna(0).astype(int)
    )

    # Reorder assessment requires known stock and reorder level.
    inventory["ReorderStatus"] = "Cannot Assess"

    known_stock = (
        inventory["OnHandUnits"].notna()
        & inventory["ReorderLevel"].notna()
    )

    inventory.loc[
        known_stock
        & (inventory["OnHandUnits"] == 0),
        "ReorderStatus",
    ] = "Stockout - Urgent Review"

    inventory.loc[
        known_stock
        & (inventory["OnHandUnits"] > 0)
        & (
            inventory["OnHandUnits"]
            <= inventory["ReorderLevel"]
        ),
        "ReorderStatus",
    ] = "Reorder Attention"

    inventory.loc[
        known_stock
        & (
            inventory["OnHandUnits"]
            > inventory["ReorderLevel"]
        ),
        "ReorderStatus",
    ] = "Stock Above Reorder Level"

    # --------------------------------------------------------
    # 3.3 Stockout frequency analysis
    # --------------------------------------------------------

    stockout_by_product = (
        inventory.groupby("ProductID", dropna=False)
        .agg(
            snapshot_count=("SnapshotMonth", "count"),
            stockout_snapshots=("StockoutFlag", "sum"),
            average_on_hand_units=("OnHandUnits", "mean"),
            latest_on_hand_units=("OnHandUnits", "last"),
            reorder_level=("ReorderLevel", "last"),
        )
        .reset_index()
    )

    stockout_by_product["stockout_frequency_percent"] = np.where(
        stockout_by_product["snapshot_count"] > 0,
        (
            stockout_by_product["stockout_snapshots"]
            / stockout_by_product["snapshot_count"]
        ) * 100,
        np.nan,
    )

    stockout_by_product = stockout_by_product.sort_values(
        ["stockout_frequency_percent", "stockout_snapshots"],
        ascending=[False, False],
        na_position="last",
    )

    # --------------------------------------------------------
    # 3.4 Stockout analysis by store
    # --------------------------------------------------------

    stockout_by_store = (
        inventory.groupby("StoreID", dropna=False)
        .agg(
            snapshot_count=("SnapshotMonth", "count"),
            stockout_snapshots=("StockoutFlag", "sum"),
            average_on_hand_units=("OnHandUnits", "mean"),
        )
        .reset_index()
    )

    stockout_by_store["stockout_frequency_percent"] = np.where(
        stockout_by_store["snapshot_count"] > 0,
        (
            stockout_by_store["stockout_snapshots"]
            / stockout_by_store["snapshot_count"]
        ) * 100,
        np.nan,
    )

    stockout_by_store = stockout_by_store.sort_values(
        "stockout_frequency_percent",
        ascending=False,
        na_position="last",
    )

    # --------------------------------------------------------
    # 3.5 Reorder attention analysis
    # --------------------------------------------------------

    reorder_attention = inventory[
        inventory["ReorderStatus"].isin(
            [
                "Stockout - Urgent Review",
                "Reorder Attention",
            ]
        )
    ].copy()

    reorder_attention = reorder_attention.sort_values(
        ["ReorderStatus", "OnHandUnits"],
        ascending=[True, True],
        na_position="last",
    )

    # --------------------------------------------------------
    # 3.6 Slow-moving product analysis
    # --------------------------------------------------------
    # Important:
    # Inventory data alone does not measure sales movement.
    # Transaction quantities are used as a movement proxy.
    # Products with no recorded sales in the available period
    # are identified separately.
    # A low sales quantity does not prove that a product is
    # slow-moving without a defined business threshold.

    valid_sales = transactions.dropna(
        subset=["OrderDate", "ProductID", "Quantity"]
    ).copy()

    if not valid_sales.empty:
        analysis_end_date = valid_sales["OrderDate"].max()

        analysis_start_date = valid_sales["OrderDate"].min()

        sales_window_days = max(
            1,
            (analysis_end_date - analysis_start_date).days + 1,
        )

        product_sales = (
            valid_sales.groupby("ProductID", dropna=False)
            .agg(
                units_sold=("Quantity", "sum"),
                transaction_lines=("Quantity", "count"),
                first_sale_date=("OrderDate", "min"),
                last_sale_date=("OrderDate", "max"),
            )
            .reset_index()
        )

        product_sales["sales_days_observed"] = (
            product_sales["last_sale_date"]
            - product_sales["first_sale_date"]
        ).dt.days + 1

        # Average daily sales across the full analysis window.
        product_sales["average_units_sold_per_day"] = (
            product_sales["units_sold"] / sales_window_days
        )

    else:
        analysis_start_date = pd.NaT
        analysis_end_date = pd.NaT
        sales_window_days = 1

        product_sales = pd.DataFrame(
            columns=[
                "ProductID",
                "units_sold",
                "transaction_lines",
                "first_sale_date",
                "last_sale_date",
                "sales_days_observed",
                "average_units_sold_per_day",
            ]
        )

    # Include all inventory products, even if sales are absent.
    inventory_products = (
        inventory[["ProductID"]]
        .dropna()
        .drop_duplicates()
    )

    slow_movers = inventory_products.merge(
        product_sales,
        on="ProductID",
        how="left",
        validate="one_to_one",
    )

    slow_movers["units_sold"] = (
        slow_movers["units_sold"].fillna(0)
    )

    slow_movers["transaction_lines"] = (
        slow_movers["transaction_lines"].fillna(0).astype(int)
    )

    slow_movers["movement_status"] = np.where(
        slow_movers["transaction_lines"] == 0,
        "No Recorded Sales",
        "Sales Recorded - Review Movement Rate",
    )

    # Add latest known stock information for each product.
    latest_inventory = (
        inventory.sort_values("SnapshotMonth")
        .drop_duplicates("ProductID", keep="last")
        [
            [
                "ProductID",
                "OnHandUnits",
                "ReorderLevel",
                "StockoutFlag",
                "ReorderStatus",
                "SnapshotMonth",
            ]
        ]
        .rename(
            columns={
                "OnHandUnits": "latest_on_hand_units",
                "ReorderLevel": "latest_reorder_level",
                "StockoutFlag": "latest_stockout_flag",
                "ReorderStatus": "latest_reorder_status",
                "SnapshotMonth": "latest_snapshot_month",
            }
        )
    )

    slow_movers = slow_movers.merge(
        latest_inventory,
        on="ProductID",
        how="left",
        validate="one_to_one",
    )

    # Sort by low observed sales; this is a review list,
    # not a claim that all listed products are slow-moving.
    slow_movers = slow_movers.sort_values(
        ["units_sold", "transaction_lines"],
        ascending=[True, True],
    )

    # --------------------------------------------------------
    # 3.7 Add product names if available
    # --------------------------------------------------------

    if products is not None and "ProductID" in products.columns:

        products = products.copy()

        products["ProductID"] = (
            products["ProductID"].astype("string").str.strip()
        )

        name_candidates = [
            "ProductName",
            "Product_Name",
            "Name",
        ]

        product_name_column = next(
            (
                col for col in name_candidates
                if col in products.columns
            ),
            None,
        )

        if product_name_column:
            product_names = (
                products[["ProductID", product_name_column]]
                .drop_duplicates("ProductID")
                .rename(
                    columns={
                        product_name_column: "ProductName"
                    }
                )
            )

            stockout_by_product = stockout_by_product.merge(
                product_names,
                on="ProductID",
                how="left",
                validate="many_to_one",
            )

            slow_movers = slow_movers.merge(
                product_names,
                on="ProductID",
                how="left",
                validate="many_to_one",
            )

            reorder_attention = reorder_attention.merge(
                product_names,
                on="ProductID",
                how="left",
                validate="many_to_one",
            )

    # --------------------------------------------------------
    # 4. SAVE OUTPUTS
    # --------------------------------------------------------

    stockout_product_file = (
        OUTPUT_DIR / "module_13_stockout_by_product.csv"
    )

    stockout_store_file = (
        OUTPUT_DIR / "module_13_stockout_by_store.csv"
    )

    reorder_file = (
        OUTPUT_DIR / "module_13_reorder_attention.csv"
    )

    slow_mover_file = (
        OUTPUT_DIR / "module_13_slow_movers.csv"
    )

    cleaned_inventory_file = (
        OUTPUT_DIR / "module_13_inventory_cleaned.csv"
    )

    save_csv(stockout_by_product, stockout_product_file)
    save_csv(stockout_by_store, stockout_store_file)
    save_csv(reorder_attention, reorder_file)
    save_csv(slow_movers, slow_mover_file)
    save_csv(inventory, cleaned_inventory_file)

    # --------------------------------------------------------
    # 5. SUMMARY METRICS
    # --------------------------------------------------------

    total_inventory_rows = len(inventory)

    known_stock_rows = int(
        inventory["OnHandUnits"].notna().sum()
    )

    total_stockout_rows = int(
        inventory["StockoutFlag"].sum()
    )

    total_products = int(
        inventory["ProductID"].nunique()
    )

    total_stores = int(
        inventory["StoreID"].nunique()
    )

    reorder_attention_rows = len(reorder_attention)

    no_recorded_sales_products = int(
        (slow_movers["transaction_lines"] == 0).sum()
    )

    stockout_rate = (
        total_stockout_rows / total_inventory_rows * 100
        if total_inventory_rows > 0
        else np.nan
    )

    # --------------------------------------------------------
    # 6. WRITE REPORT
    # --------------------------------------------------------

    report_file = REPORT_DIR / "module_13_inventory_analysis.txt"

    report_lines = [
        "=" * 70,
        "NORTHSTAR RETAIL ANALYTICS PLATFORM",
        "MODULE 13: INVENTORY ANALYSIS",
        "=" * 70,
        "",
        "DATASET SUMMARY",
        "-" * 70,
        f"Inventory snapshot rows: {total_inventory_rows:,}",
        f"Rows with known stock quantity: {known_stock_rows:,}",
        f"Missing stock quantities: {missing_stock_before:,}",
        f"Negative stock quantities: {invalid_negative_stock:,}",
        f"Negative reorder levels: {invalid_negative_reorder:,}",
        f"Invalid stockout flags: {invalid_stockout_flags:,}",
        f"Unique products: {total_products:,}",
        f"Unique stores: {total_stores:,}",
        "",
        "STOCKOUT SUMMARY",
        "-" * 70,
        f"Rows flagged as stockout: {total_stockout_rows:,}",
        f"Stockout snapshot rate: {stockout_rate:.2f}%",
        "",
        "REORDER ATTENTION",
        "-" * 70,
        f"Inventory rows requiring reorder/urgent review: "
        f"{reorder_attention_rows:,}",
        "",
        "SLOW-MOVING PRODUCT REVIEW",
        "-" * 70,
        f"Products with no recorded sales: "
        f"{no_recorded_sales_products:,}",
        f"Transaction analysis start: {analysis_start_date}",
        f"Transaction analysis end: {analysis_end_date}",
        "",
        "INTERPRETATION AND LIMITATIONS",
        "-" * 70,
        "Stockout frequency is based on the supplied StockoutFlag.",
        "Missing OnHandUnits values remain unknown; they are not",
        "automatically replaced with zero.",
        "Slow-mover screening uses transaction quantities as a",
        "movement proxy. No recorded sales does not necessarily",
        "prove a product is discontinued or obsolete.",
        "Inventory snapshots do not provide supplier lead times,",
        "safety stock, or forecast demand. Reorder suggestions are",
        "therefore based on OnHandUnits versus ReorderLevel only.",
        "",
        "OUTPUT FILES",
        "-" * 70,
        str(stockout_product_file),
        str(stockout_store_file),
        str(reorder_file),
        str(slow_mover_file),
        str(cleaned_inventory_file),
        str(report_file),
    ]

    report_file.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 7. OUTPUT VALIDATION
    # --------------------------------------------------------

    checks = {
        "Inventory data loaded":
            total_inventory_rows > 0,

        "Stockout product report created":
            stockout_product_file.exists(),

        "Stockout store report created":
            stockout_store_file.exists(),

        "Reorder report created":
            reorder_file.exists(),

        "Slow-mover report created":
            slow_mover_file.exists(),

        "Cleaned inventory report created":
            cleaned_inventory_file.exists(),

        "Summary report created":
            report_file.exists(),

        "Stockout frequency is within 0-100%":
            stockout_by_product[
                "stockout_frequency_percent"
            ].dropna().between(0, 100).all(),
    }

    print("\n" + "=" * 70)
    print("MODULE 13 OUTPUT VALIDATION")
    print("=" * 70)

    all_passed = True

    for description, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {description}")
        all_passed = all_passed and passed

    print("\nKEY RESULTS")
    print("-" * 70)
    print(f"Inventory rows: {total_inventory_rows:,}")
    print(f"Missing stock quantities: {missing_stock_before:,}")
    print(f"Stockout rows: {total_stockout_rows:,}")
    print(f"Stockout rate: {stockout_rate:.2f}%")
    print(f"Products needing reorder attention: {reorder_attention_rows:,}")
    print(f"Products with no recorded sales: {no_recorded_sales_products:,}")

    print("\nOUTPUT FILES")
    print(stockout_product_file)
    print(stockout_store_file)
    print(reorder_file)
    print(slow_mover_file)
    print(cleaned_inventory_file)
    print(report_file)

    if not all_passed:
        raise RuntimeError(
            "One or more Module 13 validation checks failed."
        )

    print("\nMODULE 13 COMPLETED SUCCESSFULLY!")


# ============================================================
# 8. EXECUTION
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)