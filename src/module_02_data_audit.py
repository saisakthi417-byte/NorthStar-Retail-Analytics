
"""
NorthStar Retail Analytics Platform
Module 2: Data Audit
Reads all 8 raw datasets and creates audit reports.
Does not modify raw data.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "DATA" / "raw"
REPORTS = ROOT / "reports"

DATASETS = {
    "transactions": {
        "file": "transactions.csv",
        "required": [
            "TransactionID", "OrderDate", "CustomerID", "StoreID",
            "ProductID", "Quantity", "Discount", "LineAmount",
            "Channel", "PaymentMode"
        ],
        "key": ["TransactionID"],
        "dates": ["OrderDate"],
        "numeric": ["Quantity", "Discount", "LineAmount"],
    },
    "customers": {
        "file": "customers.csv",
        "required": [
            "CustomerID", "CustomerName", "Gender", "AgeBand",
            "City", "LoyaltyTier", "JoinDate", "EmailOptIn"
        ],
        "key": ["CustomerID"],
        "dates": ["JoinDate"],
        "numeric": [],
    },
    "products": {
        "file": "products.csv",
        "required": [
            "ProductID", "ProductName", "Category", "SubCategory",
            "Brand", "UnitCost", "MRP", "LaunchDate"
        ],
        "key": ["ProductID"],
        "dates": ["LaunchDate"],
        "numeric": ["UnitCost", "MRP"],
    },
    "stores": {
        "file": "stores.csv",
        "required": [
            "StoreID", "StoreName", "City", "State", "Format",
            "OpenDate", "SqFt", "Region"
        ],
        "key": ["StoreID"],
        "dates": ["OpenDate"],
        "numeric": ["SqFt"],
    },
    "returns": {
        "file": "returns.csv",
        "required": [
            "ReturnID", "TransactionID", "ReturnDate",
            "Reason", "RefundAmount", "Condition"
        ],
        "key": ["ReturnID"],
        "dates": ["ReturnDate"],
        "numeric": ["RefundAmount"],
    },
    "inventory": {
        "file": "inventory.csv",
        "required": [
            "SnapshotMonth", "StoreID", "ProductID",
            "OnHandUnits", "ReorderLevel", "StockoutFlag"
        ],
        "key": ["SnapshotMonth", "StoreID", "ProductID"],
        "dates": ["SnapshotMonth"],
        "numeric": ["OnHandUnits", "ReorderLevel", "StockoutFlag"],
    },
    "campaigns": {
        "file": "campaigns.json",
        "required": [
            "campaign_id", "name", "channel", "start_date", "end_date",
            "spend_inr", "impressions", "clicks", "target_category"
        ],
        "key": ["campaign_id"],
        "dates": ["start_date", "end_date"],
        "numeric": ["spend_inr", "impressions", "clicks"],
    },
    "store_targets": {
        "file": "store_targets_legacy.csv",
        "required": [
            "store_code", "period", "sales_target_inr", "footfall_target"
        ],
        "key": ["store_code", "period"],
        "dates": [],
        "numeric": ["sales_target_inr", "footfall_target"],
    },
}


def load_data(path):
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)

        if isinstance(data, dict):
            for field in ("data", "records", "campaigns"):
                if isinstance(data.get(field), list):
                    data = data[field]
                    break

        if not isinstance(data, list):
            raise ValueError("JSON must contain a list of records.")

        return pd.DataFrame(data)

    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def add_issue(issues, dataset, column, issue, count, examples="", severity="Review"):
    if count > 0:
        issues.append({
            "Dataset": dataset,
            "Column": column,
            "Issue": issue,
            "Count": int(count),
            "Examples": str(examples),
            "Severity": severity,
        })


def examples(values, limit=8):
    return ", ".join(
        pd.Series(values).dropna().astype(str).drop_duplicates().head(limit)
    )


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    issues = []
    summaries = []
    dataframes = {}

    # 1. Load and audit every source dataset.
    for name, config in DATASETS.items():
        path = RAW / config["file"]

        if not path.exists():
            # Also support source files stored in the project root.
            path = ROOT / config["file"]

        if not path.exists():
            print(f"ERROR: File not found: {config['file']}")
            summaries.append({
                "Dataset": name,
                "SourceFile": config["file"],
                "Status": "MISSING",
                "Rows": 0,
                "Columns": 0,
                "MissingCells": 0,
                "ExactDuplicateRows": 0,
                "MissingRequiredColumns": ", ".join(config["required"]),
            })
            continue

        try:
            df = load_data(path)
            dataframes[name] = df
            missing_columns = [
                col for col in config["required"] if col not in df.columns
            ]

            print(f"Loaded and audited {config['file']}: {df.shape}")

            if df.empty:
                add_issue(
                    issues, name, "Dataset", "Empty dataset",
                    1, "No data rows", "High"
                )

            add_issue(
                issues, name, "Dataset", "Exact duplicate rows",
                int(df.duplicated().sum()),
                "Rows identical across all columns", "Review"
            )

            for col in missing_columns:
                add_issue(
                    issues, name, col, "Required column missing",
                    1, "Column not present", "High"
                )

            # Missing values.
            for col in df.columns:
                missing = int(df[col].isna().sum())
                add_issue(
                    issues, name, col, "Missing values",
                    missing, "Blank or null values", "Review"
                )

            # Invalid dates.
            for col in config["dates"]:
                if col not in df.columns:
                    continue

                original = df[col]
                nonblank = (
                    original.notna()
                    & original.astype(str).str.strip().ne("")
                )
                parsed = pd.to_datetime(
                    original, errors="coerce", format="mixed"
                )
                invalid = nonblank & parsed.isna()

                add_issue(
                    issues, name, col, "Invalid date values",
                    int(invalid.sum()), examples(original[invalid]), "Review"
                )

            # Numeric validation.
            for col in config["numeric"]:
                if col not in df.columns:
                    continue

                original = df[col]
                nonblank = (
                    original.notna()
                    & original.astype(str).str.strip().ne("")
                )
                numeric = pd.to_numeric(original, errors="coerce")
                invalid = nonblank & numeric.isna()

                add_issue(
                    issues, name, col, "Non-numeric values",
                    int(invalid.sum()), examples(original[invalid]), "High"
                )

                if col != "Discount":
                    negative = numeric < 0
                    add_issue(
                        issues, name, col, "Negative numeric values",
                        int(negative.sum()), examples(numeric[negative]), "Review"
                    )

            # Discount range.
            if name == "transactions" and "Discount" in df.columns:
                discount = pd.to_numeric(df["Discount"], errors="coerce")
                invalid = discount.notna() & (
                    (discount < 0) | (discount > 100)
                )
                add_issue(
                    issues, name, "Discount", "Discount outside 0-100 range",
                    int(invalid.sum()), examples(discount[invalid]), "Review"
                )

            # Product selling price below unit cost.
            if name == "products" and {"UnitCost", "MRP"}.issubset(df.columns):
                cost = pd.to_numeric(df["UnitCost"], errors="coerce")
                mrp = pd.to_numeric(df["MRP"], errors="coerce")
                invalid = cost.notna() & mrp.notna() & (mrp < cost)
                ids = df.loc[invalid, "ProductID"] if "ProductID" in df else df.index[invalid]

                add_issue(
                    issues, name, "MRP", "MRP lower than UnitCost",
                    int(invalid.sum()), examples(ids), "Review"
                )

            # Campaign checks.
            if name == "campaigns":
                if {"start_date", "end_date"}.issubset(df.columns):
                    start = pd.to_datetime(
                        df["start_date"], errors="coerce", format="mixed"
                    )
                    end = pd.to_datetime(
                        df["end_date"], errors="coerce", format="mixed"
                    )
                    invalid = start.notna() & end.notna() & (end < start)
                    ids = df.loc[invalid, "campaign_id"]

                    add_issue(
                        issues, name, "end_date",
                        "Campaign ends before it starts",
                        int(invalid.sum()), examples(ids), "High"
                    )

                if {"clicks", "impressions"}.issubset(df.columns):
                    clicks = pd.to_numeric(df["clicks"], errors="coerce")
                    impressions = pd.to_numeric(
                        df["impressions"], errors="coerce"
                    )
                    invalid = (
                        clicks.notna() & impressions.notna()
                        & (clicks > impressions)
                    )
                    ids = df.loc[invalid, "campaign_id"]

                    add_issue(
                        issues, name, "clicks", "Clicks exceed impressions",
                        int(invalid.sum()), examples(ids), "Review"
                    )

            # Duplicate business keys.
            # Exact copies are reported separately. Flag a key only if rows
            # sharing that key contain conflicting information.
            key_cols = config["key"]

            if all(col in df.columns for col in key_cols):
                valid = df[key_cols].notna().all(axis=1)

                for col in key_cols:
                    valid &= df[col].astype(str).str.strip().ne("")

                duplicate_mask = valid & df.duplicated(
                    subset=key_cols, keep=False
                )
                conflicting_keys = []

                if duplicate_mask.any():
                    duplicate_rows = df.loc[duplicate_mask]

                    for key, group in duplicate_rows.groupby(
                        key_cols, dropna=False, sort=False
                    ):
                        if len(group.drop_duplicates()) > 1:
                            if not isinstance(key, tuple):
                                key = (key,)
                            conflicting_keys.append(key)

                examples_text = "; ".join(
                    " | ".join(map(str, key))
                    for key in conflicting_keys[:8]
                )

                add_issue(
                    issues, name, " + ".join(key_cols),
                    "Conflicting duplicate key values",
                    len(conflicting_keys), examples_text, "High"
                )

            summaries.append({
                "Dataset": name,
                "SourceFile": config["file"],
                "Status": "LOADED",
                "Rows": len(df),
                "Columns": len(df.columns),
                "MissingCells": int(df.isna().sum().sum()),
                "ExactDuplicateRows": int(df.duplicated().sum()),
                "MissingRequiredColumns": ", ".join(missing_columns),
            })

        except Exception as error:
            print(
                f"ERROR auditing {config['file']}: "
                f"{type(error).__name__}: {error}"
            )
            summaries.append({
                "Dataset": name,
                "SourceFile": config["file"],
                "Status": f"ERROR: {type(error).__name__}: {error}",
                "Rows": 0,
                "Columns": 0,
                "MissingCells": 0,
                "ExactDuplicateRows": 0,
                "MissingRequiredColumns": "",
            })

    # 2. Referential-integrity checks.
    def check_reference(source_name, source_col, target_name, target_col):
        if source_name not in dataframes or target_name not in dataframes:
            return

        source = dataframes[source_name]
        target = dataframes[target_name]

        if source_col not in source.columns or target_col not in target.columns:
            return

        target_values = set(
            target[target_col].dropna().astype(str).str.strip()
        )
        values = source[source_col].astype("string").str.strip()
        invalid = values.notna() & values.ne("") & ~values.isin(target_values)
        distinct = values[invalid].drop_duplicates()

        add_issue(
            issues, source_name, source_col,
            f"Unmatched references to {target_name}.{target_col}",
            int(invalid.sum()),
            f"Distinct unmatched IDs: {len(distinct)}; {examples(distinct)}",
            "Review"
        )

    check_reference("transactions", "CustomerID", "customers", "CustomerID")
    check_reference("transactions", "StoreID", "stores", "StoreID")
    check_reference("transactions", "ProductID", "products", "ProductID")
    check_reference("returns", "TransactionID", "transactions", "TransactionID")
    check_reference("inventory", "StoreID", "stores", "StoreID")
    check_reference("inventory", "ProductID", "products", "ProductID")
    check_reference("store_targets", "store_code", "stores", "StoreID")

    # 3. Save reports.
    issue_columns = [
        "Dataset", "Column", "Issue", "Count", "Examples", "Severity"
    ]
    issue_df = pd.DataFrame(issues, columns=issue_columns)
    summary_df = pd.DataFrame(summaries)

    issue_path = REPORTS / "module_02_data_audit_issues.csv"
    summary_path = REPORTS / "module_02_data_audit_summary.csv"

    issue_df.to_csv(issue_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    successful = int((summary_df["Status"] == "LOADED").sum())
    print("\n" + "=" * 55)
    print("MODULE 2: DATA AUDIT SUMMARY")
    print("=" * 55)
    print(f"Datasets loaded: {successful}/{len(DATASETS)}")
    print(f"Audit findings: {len(issue_df)}")
    print(f"Issue report: {issue_path}")
    print(f"Load summary: {summary_path}")

    if not issue_df.empty:
        print("\nAffected counts by severity (not finding totals):")
        print(issue_df.groupby("Severity")["Count"].sum().to_string())

    if successful != len(DATASETS):
        print("\nWARNING: One or more datasets failed to load.")
        raise SystemExit(1)

    print("\nAudit complete. Raw source files were not modified.")


if __name__ == "__main__":
    main()
