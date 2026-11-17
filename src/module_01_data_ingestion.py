
"""
Module 1: Data Ingestion
Project: NorthStar Retail Analytics
"""

from pathlib import Path
import json
import logging
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "DATA" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

REQUIRED_COLUMNS = {
    "transactions.csv": [
        "TransactionID", "OrderDate", "CustomerID", "StoreID",
        "ProductID", "Quantity", "Discount", "LineAmount",
        "Channel", "PaymentMode"
    ],
    "customers.csv": [
        "CustomerID", "CustomerName", "Gender", "AgeBand",
        "City", "LoyaltyTier", "JoinDate", "EmailOptIn"
    ],
    "products.csv": [
        "ProductID", "ProductName", "Category", "SubCategory",
        "Brand", "UnitCost", "MRP", "LaunchDate"
    ],
    "stores.csv": [
        "StoreID", "StoreName", "City", "State",
        "Format", "OpenDate", "SqFt", "Region"
    ],
    "returns.csv": [
        "ReturnID", "TransactionID", "ReturnDate",
        "Reason", "RefundAmount", "Condition"
    ],
    "inventory.csv": [
        "SnapshotMonth", "StoreID", "ProductID",
        "OnHandUnits", "ReorderLevel", "StockoutFlag"
    ],
    "campaigns.json": [
        "campaign_id", "name", "channel", "start_date",
        "end_date", "spend_inr", "impressions",
        "clicks", "target_category"
    ],
    "store_targets_legacy.csv": [
        "store_code", "period",
        "sales_target_inr", "footfall_target"
    ]
}


def load_dataset(filename):
    """Load a CSV or JSON file and validate its columns."""

    file_path = RAW_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Source file not found: {file_path}"
        )

    if filename.endswith(".csv"):
        df = pd.read_csv(file_path)

    elif filename.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as file:
            records = json.load(file)

        if isinstance(records, dict):
            records = records.get("data", records)

        if isinstance(records, dict):
            records = [records]

        if not isinstance(records, list):
            raise ValueError(
                "JSON must contain a list of records."
            )

        df = pd.json_normalize(records)

    else:
        raise ValueError(
            f"Unsupported file format: {filename}"
        )

    if df.empty:
        raise ValueError("Dataset contains no data rows.")

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS[filename]
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def main():
    validation_results = []

    for filename in REQUIRED_COLUMNS:
        try:
            df = load_dataset(filename)

            result = {
                "File": filename,
                "Status": "PASS",
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing_Values": int(df.isna().sum().sum()),
                "Duplicate_Rows": int(df.duplicated().sum()),
                "Message": "Loaded and validated successfully"
            }

            logging.info(
                "PASS: %s | Rows: %s | Columns: %s",
                filename, len(df), len(df.columns)
            )

        except (
            FileNotFoundError,
            json.JSONDecodeError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            ValueError,
            OSError
        ) as error:
            result = {
                "File": filename,
                "Status": "FAIL",
                "Rows": 0,
                "Columns": 0,
                "Missing_Values": 0,
                "Duplicate_Rows": 0,
                "Message": str(error)
            }

            logging.error("FAIL: %s | %s", filename, error)

        validation_results.append(result)

    report = pd.DataFrame(validation_results)

    report_path = (
        REPORT_DIR / "module_01_ingestion_validation.csv"
    )
    report.to_csv(report_path, index=False)

    print("\nMODULE 1: DATA INGESTION SUMMARY")
    print(report.to_string(index=False))

    passed = (report["Status"] == "PASS").sum()
    total = len(REQUIRED_COLUMNS)

    print(f"\nDatasets passed: {passed}/{total}")
    print(f"Validation report: {report_path}")

    if passed != total:
        raise SystemExit(
            "Some datasets failed. Check the report."
        )


if __name__ == "__main__":
    main()
