
from pathlib import Path
import json
import logging
import pandas as pd

# ============================================================
# NORTHSTAR RETAIL ANALYTICS PLATFORM
# MODULE 4: DATA INTEGRATION AND LEGACY RECONCILIATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "outputs" / "cleaned"
OUTPUT_DIR = BASE_DIR / "outputs" / "integrated"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)

FILES = {
    "transactions": "transactions.csv",
    "customers": "customers.csv",
    "products": "products.csv",
    "stores": "stores.csv",
    "returns": "returns.csv",
    "inventory": "inventory.csv",
    "campaigns": "campaigns.json",
    "store_targets": "store_targets_legacy.csv",
}

REQUIRED_COLUMNS = {
    "transactions": [
        "TransactionID", "OrderDate", "CustomerID", "StoreID",
        "ProductID", "Quantity", "Discount", "LineAmount",
        "Channel", "PaymentMode",
    ],
    "customers": [
        "CustomerID", "CustomerName", "Gender", "AgeBand",
        "City", "LoyaltyTier", "JoinDate", "EmailOptIn",
    ],
    "products": [
        "ProductID", "ProductName", "Category", "SubCategory",
        "Brand", "UnitCost", "MRP", "LaunchDate",
    ],
    "stores": [
        "StoreID", "StoreName", "City", "State",
        "Format", "OpenDate", "SqFt", "Region",
    ],
    "returns": [
        "ReturnID", "TransactionID", "ReturnDate",
        "Reason", "RefundAmount", "Condition",
    ],
    "inventory": [
        "SnapshotMonth", "StoreID", "ProductID",
        "OnHandUnits", "ReorderLevel", "StockoutFlag",
    ],
    "campaigns": [
        "campaign_id", "name", "channel", "start_date",
        "end_date", "spend_inr", "impressions", "clicks",
        "target_category",
    ],
    "store_targets": [
        "store_code", "period", "sales_target_inr",
        "footfall_target",
    ],
}


# ============================================================
# STEP 1: LOAD DATASETS
# ============================================================

def load_datasets():
    datasets = {}

    for name, filename in FILES.items():
        path = CLEANED_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing cleaned dataset: {path}\n"
                "Run Module 3 before Module 4."
            )

        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, dict):
                if isinstance(data.get("data"), list):
                    data = data["data"]
                else:
                    raise ValueError(
                        f"Unsupported JSON structure: {filename}"
                    )

            if not isinstance(data, list):
                raise ValueError(
                    f"Expected a JSON list in {filename}"
                )

            df = pd.DataFrame(data)
        else:
            df = pd.read_csv(path)

        datasets[name] = df

        logging.info(
            "%s loaded: %s rows, %s columns",
            name, len(df), len(df.columns),
        )

    return datasets


# ============================================================
# STEP 2: VALIDATE DATASET SCHEMAS
# ============================================================

def validate_columns(datasets):
    errors = []

    for name, required in REQUIRED_COLUMNS.items():
        missing = sorted(set(required) - set(datasets[name].columns))

        if missing:
            errors.append(f"{name}: missing columns {missing}")

    if errors:
        raise ValueError(
            "Dataset schema validation failed:\n- "
            + "\n- ".join(errors)
        )

    logging.info("Schema validation passed for all 8 datasets.")


# ============================================================
# STEP 3: STANDARDISE KEYS, TEXT AND DATES
# ============================================================

def standardise_datasets(datasets):
    key_columns = {
        "transactions": [
            "TransactionID", "CustomerID", "StoreID", "ProductID"
        ],
        "customers": ["CustomerID"],
        "products": ["ProductID"],
        "stores": ["StoreID"],
        "returns": ["ReturnID", "TransactionID"],
        "inventory": ["StoreID", "ProductID"],
        "store_targets": ["store_code"],
    }

    for dataset_name, columns in key_columns.items():
        df = datasets[dataset_name]

        for column in columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

    # Store identifiers are compared without case differences.
    datasets["transactions"]["StoreID"] = (
        datasets["transactions"]["StoreID"].str.upper()
    )
    datasets["stores"]["StoreID"] = (
        datasets["stores"]["StoreID"].str.upper()
    )
    datasets["store_targets"]["store_code"] = (
        datasets["store_targets"]["store_code"].str.upper()
    )

    date_columns = {
        "transactions": ["OrderDate"],
        "customers": ["JoinDate"],
        "products": ["LaunchDate"],
        "stores": ["OpenDate"],
        "returns": ["ReturnDate"],
        "inventory": ["SnapshotMonth"],
        "campaigns": ["start_date", "end_date"],
    }

    for name, columns in date_columns.items():
        for column in columns:
            datasets[name][column] = pd.to_datetime(
                datasets[name][column],
                errors="coerce",
                format="mixed",
                dayfirst=True,
            )

    # Convert common legacy period formats to YYYY-MM.
    # Unparseable values remain missing and are reported.
    targets = datasets["store_targets"]
    raw_period = targets["period"].astype("string").str.strip()

    parsed_period = pd.to_datetime(
        raw_period,
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )

    targets["period"] = parsed_period.dt.strftime("%Y-%m")

    return datasets


# ============================================================
# STEP 4: REPORT DIMENSION KEY QUALITY
# ============================================================

def dimension_key_report(df, dataset_name, key):
    non_null = df.loc[df[key].notna(), key]
    duplicates = non_null[non_null.duplicated(keep=False)]

    return {
        "dataset": dataset_name,
        "key": key,
        "row_count": len(df),
        "missing_key_rows": int(df[key].isna().sum()),
        "duplicate_key_rows": int(len(duplicates)),
        "distinct_duplicate_keys": int(duplicates.nunique()),
        "sample_duplicate_keys": ", ".join(
            duplicates.astype(str).drop_duplicates().head(10)
        ),
    }


def validate_dimension_keys(datasets):
    specifications = [
        ("customers", "CustomerID"),
        ("products", "ProductID"),
        ("stores", "StoreID"),
    ]

    reports = [
        dimension_key_report(datasets[name], name, key)
        for name, key in specifications
    ]

    pd.DataFrame(reports).to_csv(
        REPORT_DIR / "module_04_dimension_key_quality.csv",
        index=False,
    )

    for name, key in specifications:
        df = datasets[name]

        if df[key].isna().any():
            raise ValueError(
                f"{name} contains missing {key} values. "
                "Review module_04_dimension_key_quality.csv."
            )

        if df[key].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate {key} values. "
                "Review module_04_dimension_key_quality.csv. "
                "No records were automatically deleted."
            )

    logging.info("Dimension key validation passed.")


# ============================================================
# STEP 5: RECONCILIATION HELPER
# ============================================================

def reconcile_keys(left, right, left_key, right_key, check_name):
    left_values = left[left_key].dropna().astype(str)
    right_values = right[right_key].dropna().astype(str)

    left_keys = set(left_values)
    right_keys = set(right_values)

    unmatched = sorted(left_keys - right_keys)

    return {
        "check": check_name,
        "left_key": left_key,
        "right_key": right_key,
        "left_rows": len(left),
        "right_rows": len(right),
        "left_unique_keys": len(left_keys),
        "right_unique_keys": len(right_keys),
        "missing_left_key_rows": int(left[left_key].isna().sum()),
        "unmatched_unique_keys": len(unmatched),
        "sample_unmatched_keys": ", ".join(unmatched[:10]),
    }


# ============================================================
# STEP 6: RECONCILE LEGACY STORE TARGETS
# ============================================================

def reconcile_store_targets(datasets):
    targets = datasets["store_targets"].copy()
    stores = datasets["stores"].copy()

    # Rename the legacy key to match the store master key.
    # The source files themselves are not changed.
    targets_for_join = targets.rename(
        columns={"store_code": "StoreID"}
    )

    reconciliation = targets_for_join.merge(
        stores[
            ["StoreID", "StoreName", "City", "State", "Region"]
        ],
        on="StoreID",
        how="left",
        validate="many_to_one",
        indicator="_store_match",
        suffixes=("_target", "_store"),
    )

    reconciliation["StoreMatchStatus"] = (
        reconciliation["_store_match"]
        .map({
            "both": "MATCHED",
            "left_only": "UNMATCHED_STORE_CODE",
        })
    )

    reconciliation.drop(
        columns=["_store_match"]
    ).to_csv(
        OUTPUT_DIR / "reconciled_store_targets.csv",
        index=False,
    )

    unmatched = reconciliation[
        reconciliation["StoreMatchStatus"] == "UNMATCHED_STORE_CODE"
    ].copy()

    unmatched.drop(
        columns=["_store_match"]
    ).to_csv(
        REPORT_DIR / "module_04_unmatched_store_targets.csv",
        index=False,
    )

    # Flag duplicate store-period combinations rather than
    # silently adding potentially duplicated targets.
    duplicate_mask = targets.duplicated(
        subset=["store_code", "period"],
        keep=False,
    )

    targets[duplicate_mask].to_csv(
        REPORT_DIR / "module_04_duplicate_store_period_targets.csv",
        index=False,
    )

    # Aggregate only unique store-period target records.
    valid_targets = targets[~duplicate_mask].copy()

    valid_targets = valid_targets.merge(
        stores[
            ["StoreID", "StoreName", "City", "State", "Region"]
        ],
        left_on="store_code",
        right_on="StoreID",
        how="left",
        validate="many_to_one",
    )

    regional_summary = (
        valid_targets.groupby(
            ["period", "Region"],
            dropna=False,
        )
        .agg(
            StoreCount=("store_code", "nunique"),
            SalesTargetINR=("sales_target_inr", "sum"),
            FootfallTarget=("footfall_target", "sum"),
        )
        .reset_index()
    )

    regional_summary.to_csv(
        OUTPUT_DIR / "store_targets_by_period_region.csv",
        index=False,
    )

    return {
        "target_rows": len(targets),
        "matched_target_rows": int(
            (reconciliation["_store_match"] == "both").sum()
        ),
        "unmatched_target_rows": len(unmatched),
        "duplicate_store_period_rows": int(duplicate_mask.sum()),
        "invalid_period_rows": int(targets["period"].isna().sum()),
    }


# ============================================================
# STEP 7: INTEGRATE TRANSACTIONS
# ============================================================

def integrate_transactions(datasets):
    transactions = datasets["transactions"].copy()
    customers = datasets["customers"].copy()
    products = datasets["products"].copy()
    stores = datasets["stores"].copy()

    original_rows = len(transactions)

    # Keep online transactions; ONLINE is not a physical store key.
    online_mask = transactions["StoreID"].eq("ONLINE")

    integrated = transactions.merge(
        customers,
        on="CustomerID",
        how="left",
        validate="many_to_one",
        indicator="_customer_match",
        suffixes=("", "_customer"),
    )

    integrated = integrated.merge(
        products,
        on="ProductID",
        how="left",
        validate="many_to_one",
        indicator="_product_match",
        suffixes=("", "_product"),
    )

    integrated = integrated.merge(
        stores,
        on="StoreID",
        how="left",
        validate="many_to_one",
        indicator="_store_match",
        suffixes=("", "_store"),
    )

    if len(integrated) != original_rows:
        raise RuntimeError(
            f"Transaction rows changed during integration: "
            f"{original_rows} -> {len(integrated)}"
        )

    integrated["CustomerMatchStatus"] = (
        integrated["_customer_match"]
        .map({"both": "MATCHED", "left_only": "UNMATCHED"})
    )

    integrated["ProductMatchStatus"] = (
        integrated["_product_match"]
        .map({"both": "MATCHED", "left_only": "UNMATCHED"})
    )

    integrated["StoreMatchStatus"] = "MATCHED"

    integrated.loc[
        online_mask, "StoreMatchStatus"
    ] = "ONLINE"

    integrated.loc[
        ~online_mask & integrated["StoreID"].isna(),
        "StoreMatchStatus",
    ] = "MISSING_STORE_ID"

    integrated.loc[
        ~online_mask
        & integrated["StoreID"].notna()
        & integrated["_store_match"].eq("left_only"),
        "StoreMatchStatus",
    ] = "UNMATCHED_STORE_ID"

    # Save unmatched-reference exception reports.
    for status_column, status, filename in [
        (
            "CustomerMatchStatus",
            "UNMATCHED",
            "module_04_unmatched_customers.csv",
        ),
        (
            "ProductMatchStatus",
            "UNMATCHED",
            "module_04_unmatched_products.csv",
        ),
    ]:
        integrated[integrated[status_column] == status].copy().to_csv(
            REPORT_DIR / filename,
            index=False,
        )

    integrated[
        integrated["StoreMatchStatus"].isin(
            ["MISSING_STORE_ID", "UNMATCHED_STORE_ID"]
        )
    ].to_csv(
        REPORT_DIR / "module_04_unmatched_stores.csv",
        index=False,
    )

    integrated = integrated.drop(
        columns=[
            "_customer_match",
            "_product_match",
            "_store_match",
        ]
    )

    integrated.to_csv(
        OUTPUT_DIR / "integrated_transactions.csv",
        index=False,
    )

    return integrated


# ============================================================
# STEP 8: INTEGRATE RETURNS
# ============================================================

def integrate_returns(datasets):
    returns = datasets["returns"].copy()
    transactions = datasets["transactions"].copy()

    transaction_ids = set(
        transactions["TransactionID"].dropna().astype(str)
    )

    returns["ReturnTransactionMatch"] = (
        returns["TransactionID"].astype("string").isin(transaction_ids)
    )

    returns[
        ~returns["ReturnTransactionMatch"]
        & returns["TransactionID"].notna()
    ].to_csv(
        REPORT_DIR / "module_04_unmatched_returns.csv",
        index=False,
    )

    summary = (
        returns.groupby("TransactionID", dropna=False)
        .agg(
            ReturnRecordCount=("ReturnID", "count"),
            KnownRefundAmountINR=("RefundAmount", "sum"),
            MissingRefundAmountRows=(
                "RefundAmount",
                lambda series: int(series.isna().sum()),
            ),
            LatestReturnDate=("ReturnDate", "max"),
            ReturnReasons=(
                "Reason",
                lambda series: " | ".join(
                    series.dropna().astype(str).unique()
                ),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "return_summary_by_transaction.csv",
        index=False,
    )

    return summary


# ============================================================
# STEP 9: PREPARE INVENTORY AND CAMPAIGN OUTPUTS
# ============================================================

def save_supporting_datasets(datasets):
    inventory = datasets["inventory"].copy()
    stores = datasets["stores"][["StoreID"]]
    products = datasets["products"][["ProductID"]]

    inventory = inventory.merge(
        stores,
        on="StoreID",
        how="left",
        validate="many_to_one",
        indicator="_store_match",
    )

    inventory = inventory.merge(
        products,
        on="ProductID",
        how="left",
        validate="many_to_one",
        indicator="_product_match",
    )

    inventory["StoreReferenceStatus"] = (
        inventory["_store_match"]
        .map({"both": "MATCHED", "left_only": "UNMATCHED"})
    )

    inventory["ProductReferenceStatus"] = (
        inventory["_product_match"]
        .map({"both": "MATCHED", "left_only": "UNMATCHED"})
    )

    inventory.drop(
        columns=["_store_match", "_product_match"]
    ).to_csv(
        OUTPUT_DIR / "integrated_inventory.csv",
        index=False,
    )

    campaigns = datasets["campaigns"].copy()

    campaigns["CampaignDurationDays"] = (
        campaigns["end_date"] - campaigns["start_date"]
    ).dt.days + 1

    campaigns["CTR"] = campaigns["clicks"].div(
        campaigns["impressions"].where(
            campaigns["impressions"] > 0
        )
    )

    campaigns["CPC_INR"] = campaigns["spend_inr"].div(
        campaigns["clicks"].where(campaigns["clicks"] > 0)
    )

    campaigns.to_json(
        OUTPUT_DIR / "integrated_campaigns.json",
        orient="records",
        indent=2,
        date_format="iso",
    )


# ============================================================
# STEP 10: RECONCILIATION AND QUALITY REPORTS
# ============================================================

def create_quality_reports(
    datasets, integrated, target_metrics, return_summary
):
    transactions = datasets["transactions"]

    checks = [
        reconcile_keys(
            transactions, datasets["customers"],
            "CustomerID", "CustomerID",
            "Transactions to customers",
        ),
        reconcile_keys(
            transactions, datasets["products"],
            "ProductID", "ProductID",
            "Transactions to products",
        ),
        reconcile_keys(
            transactions[transactions["StoreID"].ne("ONLINE")],
            datasets["stores"],
            "StoreID", "StoreID",
            "Physical transactions to stores",
        ),
        reconcile_keys(
            datasets["returns"], transactions,
            "TransactionID", "TransactionID",
            "Returns to transactions",
        ),
        reconcile_keys(
            datasets["inventory"], datasets["stores"],
            "StoreID", "StoreID",
            "Inventory to stores",
        ),
        reconcile_keys(
            datasets["inventory"], datasets["products"],
            "ProductID", "ProductID",
            "Inventory to products",
        ),
        # Explicitly reconcile different source key names.
        reconcile_keys(
            datasets["store_targets"].rename(
                columns={"store_code": "StoreID"}
            ),
            datasets["stores"],
            "StoreID", "StoreID",
            "Legacy targets to stores",
        ),
    ]

    pd.DataFrame(checks).to_csv(
        REPORT_DIR / "module_04_key_reconciliation.csv",
        index=False,
    )

    quality = pd.DataFrame([{
        "source_transaction_rows": len(transactions),
        "integrated_transaction_rows": len(integrated),
        "integrated_transaction_columns": len(integrated.columns),
        "row_count_preserved": len(transactions) == len(integrated),
        "exact_duplicate_integrated_rows": int(
            integrated.duplicated().sum()
        ),
        "missing_transaction_ids": int(
            integrated["TransactionID"].isna().sum()
        ),
        "unmatched_customer_rows": int(
            (integrated["CustomerMatchStatus"] == "UNMATCHED").sum()
        ),
        "unmatched_product_rows": int(
            (integrated["ProductMatchStatus"] == "UNMATCHED").sum()
        ),
        "unmatched_physical_store_rows": int(
            integrated["StoreMatchStatus"].isin(
                ["MISSING_STORE_ID", "UNMATCHED_STORE_ID"]
            ).sum()
        ),
        "legacy_target_rows": target_metrics["target_rows"],
        "matched_legacy_target_rows": target_metrics[
            "matched_target_rows"
        ],
        "unmatched_legacy_target_rows": target_metrics[
            "unmatched_target_rows"
        ],
        "duplicate_store_period_target_rows": target_metrics[
            "duplicate_store_period_rows"
        ],
        "invalid_target_period_rows": target_metrics[
            "invalid_period_rows"
        ],
        "return_summary_rows": len(return_summary),
    }])

    quality.to_csv(
        REPORT_DIR / "module_04_integration_quality.csv",
        index=False,
    )


# ============================================================
# STEP 11: MAIN EXECUTION
# ============================================================

def main():
    logging.info("=" * 65)
    logging.info("NORTHSTAR RETAIL ANALYTICS - MODULE 4")
    logging.info("DATA INTEGRATION AND LEGACY RECONCILIATION")
    logging.info("=" * 65)

    datasets = load_datasets()
    validate_columns(datasets)
    datasets = standardise_datasets(datasets)

    validate_dimension_keys(datasets)

    target_metrics = reconcile_store_targets(datasets)

    integrated = integrate_transactions(datasets)

    return_summary = integrate_returns(datasets)

    save_supporting_datasets(datasets)

    create_quality_reports(
        datasets,
        integrated,
        target_metrics,
        return_summary,
    )

    logging.info("-" * 65)
    logging.info("MODULE 4 COMPLETED SUCCESSFULLY")
    logging.info("Integrated transaction rows: %s", len(integrated))
    logging.info(
        "Matched legacy target rows: %s",
        target_metrics["matched_target_rows"],
    )
    logging.info(
        "Unmatched legacy target rows: %s",
        target_metrics["unmatched_target_rows"],
    )
    logging.info(
        "Duplicate store-period target rows: %s",
        target_metrics["duplicate_store_period_rows"],
    )
    logging.info("Integrated outputs: %s", OUTPUT_DIR)
    logging.info("Reports: %s", REPORT_DIR)
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
