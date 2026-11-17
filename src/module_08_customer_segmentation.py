
"""
NorthStar Retail Analytics Platform
Module 8: Cohort & Retention Analysis

Requirement:
- Group customers by join month.
- Track repeat purchase behaviour over time.
- Calculate monthly cohort retention.
- Produce CSV reports, a retention heatmap, and written interpretation.

Run from the project root:
    python src/module_08_cohort_retention.py
"""

from pathlib import Path
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. PROJECT PATHS AND LOGGING
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CLEANED_DIR = ROOT / "outputs" / "cleaned"
RAW_DIR = ROOT / "DATA" / "raw"
OUTPUT_DIR = ROOT / "outputs" / "module_08"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 2. SAFE DATA LOADING
# ---------------------------------------------------------

def load_dataset(filename):
    """Load cleaned data first; use raw data only if needed."""

    cleaned_path = CLEANED_DIR / filename
    raw_path = RAW_DIR / filename

    if cleaned_path.exists():
        path = cleaned_path
    elif raw_path.exists():
        logger.warning(
            "Cleaned file not found; using raw file: %s", filename
        )
        path = raw_path
    else:
        raise FileNotFoundError(
            f"Cannot find {filename} in {CLEANED_DIR} or {RAW_DIR}"
        )

    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %s: %s rows", filename, f"{len(df):,}")
    return df


# ---------------------------------------------------------
# 3. VALIDATE INPUT DATA
# ---------------------------------------------------------

def validate_data(transactions, customers):
    required_transactions = {
        "TransactionID",
        "CustomerID",
        "OrderDate",
        "LineAmount",
    }

    required_customers = {
        "CustomerID",
        "JoinDate",
    }

    missing_t = required_transactions - set(transactions.columns)
    missing_c = required_customers - set(customers.columns)

    if missing_t:
        raise ValueError(
            f"Transactions are missing required columns: {sorted(missing_t)}"
        )

    if missing_c:
        raise ValueError(
            f"Customers are missing required columns: {sorted(missing_c)}"
        )

    if transactions.empty or customers.empty:
        raise ValueError("Transactions or customers dataset is empty.")


# ---------------------------------------------------------
# 4. PREPARE CUSTOMER COHORTS
# ---------------------------------------------------------

def prepare_data(transactions, customers):
    transactions = transactions.copy()
    customers = customers.copy()

    # Parse dates safely.
    transactions["OrderDate"] = pd.to_datetime(
        transactions["OrderDate"],
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )

    customers["JoinDate"] = pd.to_datetime(
        customers["JoinDate"],
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )

    # Clean key fields.
    transactions["CustomerID"] = (
        transactions["CustomerID"].astype("string").str.strip()
    )
    customers["CustomerID"] = (
        customers["CustomerID"].astype("string").str.strip()
    )

    # Remove records that cannot support cohort calculations.
    transactions = transactions.dropna(
        subset=["CustomerID", "OrderDate", "TransactionID"]
    )
    customers = customers.dropna(
        subset=["CustomerID", "JoinDate"]
    )

    transactions = transactions[
        transactions["CustomerID"].ne("")
    ].copy()

    customers = customers[
        customers["CustomerID"].ne("")
    ].copy()

    # One customer must have one cohort assignment.
    # If duplicate customer records exist, retain the earliest valid join date.
    customers = (
        customers.sort_values("JoinDate")
        .drop_duplicates(subset=["CustomerID"], keep="first")
    )

    # Convert the join date to its monthly cohort.
    customers["CohortMonth"] = (
        customers["JoinDate"].dt.to_period("M")
    )

    # Convert order date to purchase activity month.
    transactions["OrderMonth"] = (
        transactions["OrderDate"].dt.to_period("M")
    )

    # Join the customer cohort to each transaction.
    activity = transactions.merge(
        customers[["CustomerID", "JoinDate", "CohortMonth"]],
        on="CustomerID",
        how="inner",
        validate="many_to_one",
    )

    # Exclude purchases that occur before the recorded customer join date.
    before_join = activity["OrderDate"] < activity["JoinDate"]

    logger.info(
        "Transactions before recorded join date: %s",
        f"{int(before_join.sum()):,}",
    )

    activity = activity.loc[~before_join].copy()

    # Calculate months elapsed since the join cohort.
    activity["CohortIndex"] = (
        (activity["OrderMonth"].dt.year
         - activity["CohortMonth"].dt.year) * 12
        + activity["OrderMonth"].dt.month
        - activity["CohortMonth"].dt.month
    )

    activity = activity[
        activity["CohortIndex"] >= 0
    ].copy()

    # Use one activity record per customer per month.
    # Multiple orders in one month still count as one active month.
    customer_months = activity[
        ["CustomerID", "CohortMonth", "OrderMonth", "CohortIndex"]
    ].drop_duplicates()

    return customers, activity, customer_months


# ---------------------------------------------------------
# 5. COHORT SIZE
# ---------------------------------------------------------

def create_cohort_sizes(customers):
    cohort_sizes = (
        customers.groupby("CohortMonth")["CustomerID"]
        .nunique()
        .rename("CohortSize")
        .reset_index()
    )

    cohort_sizes["CohortMonth"] = (
        cohort_sizes["CohortMonth"].astype(str)
    )

    cohort_sizes.to_csv(
        OUTPUT_DIR / "cohort_sizes.csv", index=False
    )

    return cohort_sizes


# ---------------------------------------------------------
# 6. MONTHLY COHORT RETENTION
# ---------------------------------------------------------

def create_retention_tables(customer_months, cohort_sizes):
    # Number of unique active customers in each cohort and activity month.
    retention_counts = (
        customer_months.groupby(
            ["CohortMonth", "CohortIndex"]
        )["CustomerID"]
        .nunique()
        .unstack("CohortIndex")
        .sort_index()
    )

    # Convert cohort labels to strings for readable output.
    retention_counts.index = retention_counts.index.astype(str)
    retention_counts.columns.name = "MonthsSinceJoin"

    sizes = cohort_sizes.set_index(
        "CohortMonth"
    )["CohortSize"]

    sizes.index = sizes.index.astype(str)

    # Retention = active customers / original cohort size.
    retention = retention_counts.div(sizes, axis=0) * 100

    retention_counts.to_csv(
        OUTPUT_DIR / "cohort_active_customers.csv"
    )

    retention.to_csv(
        OUTPUT_DIR / "cohort_retention_percent.csv",
        float_format="%.2f",
    )

    return retention_counts, retention


# ---------------------------------------------------------
# 7. REPEAT PURCHASE BEHAVIOUR
# ---------------------------------------------------------

def create_repeat_purchase_summary(activity, customers):
    # Distinct transaction IDs per customer.
    purchase_counts = (
        activity.groupby("CustomerID")["TransactionID"]
        .nunique()
        .rename("TotalTransactions")
    )

    # Distinct months in which a customer purchased.
    active_month_counts = (
        activity.groupby("CustomerID")["OrderMonth"]
        .nunique()
        .rename("ActivePurchaseMonths")
    )

    # Calculate customer-level sales where LineAmount is numeric.
    activity["LineAmount"] = pd.to_numeric(
        activity["LineAmount"], errors="coerce"
    )

    customer_sales = (
        activity.groupby("CustomerID")["LineAmount"]
        .sum(min_count=1)
        .rename("ObservedSales")
    )

    customer_summary = (
        customers[["CustomerID", "JoinDate", "CohortMonth"]]
        .merge(purchase_counts, on="CustomerID", how="left")
        .merge(active_month_counts, on="CustomerID", how="left")
        .merge(customer_sales, on="CustomerID", how="left")
    )

    customer_summary["TotalTransactions"] = (
        customer_summary["TotalTransactions"]
        .fillna(0)
        .astype(int)
    )

    customer_summary["ActivePurchaseMonths"] = (
        customer_summary["ActivePurchaseMonths"]
        .fillna(0)
        .astype(int)
    )

    customer_summary["RepeatPurchaser"] = (
        customer_summary["TotalTransactions"] >= 2
    )

    customer_summary["CohortMonth"] = (
        customer_summary["CohortMonth"].astype(str)
    )

    customer_summary.to_csv(
        OUTPUT_DIR / "customer_repeat_purchase_summary.csv",
        index=False,
    )

    cohort_repeat = (
        customer_summary.groupby("CohortMonth")
        .agg(
            CohortCustomers=("CustomerID", "nunique"),
            CustomersWithPurchases=(
                "TotalTransactions",
                lambda s: int((s >= 1).sum()),
            ),
            RepeatPurchasers=("RepeatPurchaser", "sum"),
            AverageTransactions=("TotalTransactions", "mean"),
            AverageActiveMonths=("ActivePurchaseMonths", "mean"),
            ObservedSales=("ObservedSales", "sum"),
        )
        .reset_index()
    )

    cohort_repeat["RepeatPurchaseRatePct"] = np.where(
        cohort_repeat["CustomersWithPurchases"] > 0,
        cohort_repeat["RepeatPurchasers"]
        / cohort_repeat["CustomersWithPurchases"] * 100,
        np.nan,
    )

    cohort_repeat.to_csv(
        OUTPUT_DIR / "cohort_repeat_purchase_summary.csv",
        index=False,
        float_format="%.2f",
    )

    return customer_summary, cohort_repeat


# ---------------------------------------------------------
# 8. MONTHLY PURCHASE ACTIVITY
# ---------------------------------------------------------

def create_monthly_activity(activity):
    monthly = (
        activity.groupby("OrderMonth")
        .agg(
            ActiveCustomers=("CustomerID", "nunique"),
            Transactions=("TransactionID", "nunique"),
            ObservedSales=("LineAmount", "sum"),
        )
        .reset_index()
    )

    monthly["OrderMonth"] = monthly["OrderMonth"].astype(str)

    monthly.to_csv(
        OUTPUT_DIR / "monthly_customer_activity.csv",
        index=False,
        float_format="%.2f",
    )

    return monthly


# ---------------------------------------------------------
# 9. RETENTION HEATMAP
# ---------------------------------------------------------

def create_heatmap(retention):
    if retention.empty:
        logger.warning("Retention table is empty; heatmap skipped.")
        return

    plot_data = retention.copy()

    # Keep a readable number of months if the history is long.
    plot_data = plot_data.iloc[:, :24]

    values = plot_data.to_numpy(dtype=float)

    fig_width = max(10, min(18, plot_data.shape[1] * 0.8))
    fig_height = max(5, min(14, plot_data.shape[0] * 0.35))

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        vmin=0,
        vmax=100,
    )

    ax.set_title("NorthStar Retail — Customer Cohort Retention (%)")
    ax.set_xlabel("Months Since Customer Join")
    ax.set_ylabel("Customer Join-Month Cohort")

    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_xticklabels(
        [str(c) for c in plot_data.columns],
        rotation=45,
    )

    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index.astype(str))

    fig.colorbar(image, ax=ax, label="Retention (%)")
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "cohort_retention_heatmap.png",
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ---------------------------------------------------------
# 10. WRITTEN BUSINESS INTERPRETATION
# ---------------------------------------------------------

def create_business_report(
    customers,
    activity,
    retention,
    customer_summary,
    cohort_repeat,
):
    lines = []

    lines.append("NORTHSTAR RETAIL ANALYTICS PLATFORM")
    lines.append("MODULE 8 — COHORT & RETENTION ANALYSIS")
    lines.append("=" * 55)

    lines.append(
        f"Valid customer records with join dates: {len(customers):,}"
    )
    lines.append(
        f"Valid post-join transaction rows: {len(activity):,}"
    )
    lines.append(
        f"Join-month cohorts analysed: {customers['CohortMonth'].nunique():,}"
    )

    if not customer_summary.empty:
        total_customers = customer_summary["CustomerID"].nunique()
        purchasers = int(
            (customer_summary["TotalTransactions"] >= 1).sum()
        )
        repeaters = int(
            customer_summary["RepeatPurchaser"].sum()
        )

        repeat_rate = (
            repeaters / purchasers * 100 if purchasers else np.nan
        )

        lines.append("")
        lines.append("REPEAT PURCHASE FINDINGS")
        lines.append("-" * 35)
        lines.append(f"Customers analysed: {total_customers:,}")
        lines.append(f"Customers with post-join purchases: {purchasers:,}")
        lines.append(f"Customers with 2+ transactions: {repeaters:,}")
        lines.append(
            f"Repeat purchase rate among purchasers: {repeat_rate:.2f}%"
            if pd.notna(repeat_rate)
            else "Repeat purchase rate: not available"
        )

    lines.append("")
    lines.append("COHORT RETENTION FINDINGS")
    lines.append("-" * 35)

    if not retention.empty:
        # Month 0 should be close to 100% for cohorts with valid activity.
        # It may be lower when registered customers have not purchased.
        month_zero = retention[0].dropna() if 0 in retention.columns else pd.Series(dtype=float)

        if not month_zero.empty:
            lines.append(
                "Month 0 retention ranges from "
                f"{month_zero.min():.2f}% to {month_zero.max():.2f}%."
            )
            lines.append(
                "Month 0 can be below 100% because the cohort includes "
                "registered customers who may not have purchased."
            )

        later_columns = [
            c for c in retention.columns
            if isinstance(c, (int, np.integer)) and c > 0
        ]

        if later_columns:
            latest_month = max(later_columns)
            available = retention[latest_month].dropna()

            if not available.empty:
                best_cohort = available.idxmax()
                worst_cohort = available.idxmin()

                lines.append(
                    f"At month {latest_month}, the highest observed "
                    f"retention is {available.max():.2f}% "
                    f"for cohort {best_cohort}."
                )
                lines.append(
                    f"The lowest observed retention at month {latest_month} "
                    f"is {available.min():.2f}% "
                    f"for cohort {worst_cohort}."
                )
        else:
            lines.append(
                "No later-month retention periods were available "
                "for comparison."
            )
    else:
        lines.append("No cohort retention data was available.")

    lines.append("")
    lines.append("BUSINESS RECOMMENDATIONS")
    lines.append("-" * 35)
    lines.append(
        "1. Compare retention between join-month cohorts before "
        "choosing customer engagement campaigns."
    )
    lines.append(
        "2. Investigate cohorts with weaker repeat purchase rates "
        "and test relevant re-engagement offers."
    )
    lines.append(
        "3. Evaluate cohort results alongside acquisition source, "
        "channel, discounting, and return behaviour in later modules."
    )
    lines.append(
        "4. Interpret the newest cohorts cautiously because they have "
        "had less time to make repeat purchases."
    )
    lines.append("")
    lines.append(
        "Method note: Cohorts are based on customer JoinDate, as required "
        "by the project brief. Retention means a customer has at least one "
        "recorded purchase in the relevant month. Repeat purchaser means "
        "at least two distinct transaction IDs after the recorded join date."
    )
    lines.append(
        "ObservedSales is the sum of LineAmount for valid post-join "
        "activity; confirm the business definition of LineAmount before "
        "interpreting it as net sales or profit."
    )

    report_path = REPORT_DIR / "module_08_cohort_retention_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------
# 11. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("NORTHSTAR RETAIL — MODULE 8")
    logger.info("COHORT & RETENTION ANALYSIS")
    logger.info("=" * 60)

    transactions = load_dataset("transactions.csv")
    customers = load_dataset("customers.csv")

    validate_data(transactions, customers)

    customers, activity, customer_months = prepare_data(
        transactions, customers
    )

    if activity.empty:
        raise ValueError(
            "No valid post-join customer purchases remain. "
            "Check date formats, CustomerID matches, and JoinDate values."
        )

    cohort_sizes = create_cohort_sizes(customers)

    retention_counts, retention = create_retention_tables(
        customer_months, cohort_sizes
    )

    customer_summary, cohort_repeat = create_repeat_purchase_summary(
        activity, customers
    )

    monthly = create_monthly_activity(activity)

    create_heatmap(retention)

    create_business_report(
        customers,
        activity,
        retention,
        customer_summary,
        cohort_repeat,
    )

    # Summary report for reproducibility.
    summary = pd.DataFrame([{
        "Project": "NorthStar Retail Analytics Platform",
        "Module": "Module 8 - Cohort & Retention",
        "CustomerRecords": len(customers),
        "ValidPostJoinTransactionRows": len(activity),
        "Cohorts": customers["CohortMonth"].nunique(),
        "RepeatPurchasers": int(
            customer_summary["RepeatPurchaser"].sum()
        ),
        "RetentionTableRows": retention.shape[0],
        "RetentionTableMonths": retention.shape[1],
        "Status": "Completed",
    }])

    summary.to_csv(
        REPORT_DIR / "module_08_summary.csv",
        index=False,
    )

    logger.info("-" * 60)
    logger.info("MODULE 8 COMPLETED")
    logger.info("Cohorts: %s", f"{customers['CohortMonth'].nunique():,}")
    logger.info("Valid post-join transaction rows: %s", f"{len(activity):,}")
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("Report directory: %s", REPORT_DIR)
    logger.info("Run completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("MODULE 8 FAILED: %s", exc)
        sys.exit(1)
