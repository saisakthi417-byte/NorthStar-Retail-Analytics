
from pathlib import Path
import json
import logging

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "DATA" / "raw"
CLEAN_DIR = PROJECT_ROOT / "outputs" / "cleaned"
REPORT_DIR = PROJECT_ROOT / "reports"

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# --------------------------------------------------
# 2. DATASET CONFIGURATION
# --------------------------------------------------

DATASETS = {
    "transactions": {
        "file": "transactions.csv",
        "dates": ["OrderDate"],
    },
    "customers": {
        "file": "customers.csv",
        "dates": ["JoinDate"],
    },
    "products": {
        "file": "products.csv",
        "dates": ["LaunchDate"],
    },
    "stores": {
        "file": "stores.csv",
        "dates": ["OpenDate"],
    },
    "returns": {
        "file": "returns.csv",
        "dates": ["ReturnDate"],
    },
    "inventory": {
        "file": "inventory.csv",
        "dates": ["SnapshotMonth"],
    },
    "campaigns": {
        "file": "campaigns.json",
        "dates": ["start_date", "end_date"],
    },
    "store_targets": {
        "file": "store_targets_legacy.csv",
        "dates": [],
    },
}


# --------------------------------------------------
# 3. LOAD ONE DATASET
# --------------------------------------------------

def load_dataset(filename):
    """Load CSV or JSON without modifying the raw source."""

    file_path = RAW_DIR / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Source file not found: {file_path}"
        )

    if file_path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(file_path)

    elif file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        dataframe = pd.json_normalize(data)

    else:
        raise ValueError(
            f"Unsupported file type: {filename}"
        )

    if len(dataframe.columns) == 0:
        raise ValueError(
            f"{filename} has no columns."
        )

    return dataframe


# --------------------------------------------------
# 4. CLEAN ONE DATASET
# --------------------------------------------------

def clean_dataset(name, filename, date_columns):
    """Clean one dataset and record every cleaning step."""

    dataframe = load_dataset(filename)

    original_rows = len(dataframe)
    original_missing = int(
        dataframe.isna().sum().sum()
    )

    changes = []

    # Step A: Normalize blank text and whitespace first.
    blank_count = 0
    whitespace_count = 0

    text_columns = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    for column in text_columns:
        values = dataframe[column].astype("string")
        blank_mask = values.str.strip().eq("").fillna(False)

        blank_count += int(blank_mask.sum())

        dataframe[column] = values.mask(
            blank_mask,
            pd.NA
        )

        original_text = dataframe[column].astype("string")

        cleaned_text = (
            original_text
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        changed_mask = (
            original_text.notna()
            & cleaned_text.notna()
            & original_text.ne(cleaned_text)
        )

        whitespace_count += int(changed_mask.sum())
        dataframe[column] = cleaned_text

    changes.append({
        "dataset": name,
        "step": "Blank text handling",
        "affected_rows_or_cells": blank_count,
        "reason": (
            "Convert blank and whitespace-only text "
            "to missing values."
        ),
    })

    changes.append({
        "dataset": name,
        "step": "Text whitespace standardisation",
        "affected_rows_or_cells": whitespace_count,
        "reason": (
            "Remove surrounding and repeated whitespace "
            "before duplicate detection."
        ),
    })

    # Step B: Standardize product category capitalization.
    if name == "products" and "Category" in dataframe.columns:
        original_categories = dataframe["Category"].copy()

        dataframe["Category"] = (
            dataframe["Category"]
            .astype("string")
            .str.strip()
            .str.title()
        )

        changed_count = int(
            (
                original_categories.notna()
                & original_categories.astype("string").ne(
                    dataframe["Category"]
                )
            ).sum()
        )

        changes.append({
            "dataset": name,
            "step": "Product category capitalization standardisation",
            "affected_rows_or_cells": changed_count,
            "reason": (
                "Standardize category capitalization, for example "
                "ACCESSORIES and Accessories become Accessories."
            ),
        })

    # Step C: Parse configured date columns BEFORE deduplication.
    # dayfirst=True supports source dates such as 13-01-2023.
    # ISO dates such as 2023-01-13 are also supported.
    for column in date_columns:
        if column not in dataframe.columns:
            changes.append({
                "dataset": name,
                "step": f"Date validation: {column}",
                "affected_rows_or_cells": 0,
                "reason": (
                    "Expected date column was not found; "
                    "review the dataset schema."
                ),
            })

            logging.warning(
                "%s: expected date column %s was not found.",
                name,
                column,
            )
            continue

        original_values = dataframe[column].copy()

        nonblank = (
            original_values.notna()
            & original_values.astype("string").str.strip().ne("")
        )

        parsed_dates = pd.to_datetime(
            original_values,
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        invalid_count = int(
            (nonblank & parsed_dates.isna()).sum()
        )

        dataframe[column] = parsed_dates

        changes.append({
            "dataset": name,
            "step": f"Date validation: {column}",
            "affected_rows_or_cells": invalid_count,
            "reason": (
                "Normalize parseable dates before duplicate detection. "
                "Unparseable nonblank values become missing "
                "and must be reviewed."
            ),
        })

        if invalid_count:
            logging.warning(
                "%s.%s: %s nonblank date values could not "
                "be parsed and need review.",
                name,
                column,
                invalid_count,
            )

    # Step D: Remove exact duplicates AFTER normalization.
    duplicate_count = int(
        dataframe.duplicated().sum()
    )

    dataframe = dataframe.drop_duplicates().copy()

    changes.append({
        "dataset": name,
        "step": "Exact duplicate removal after normalization",
        "affected_rows_or_cells": duplicate_count,
        "reason": (
            "Remove rows identical across all columns after "
            "normalizing text and configured dates. Records "
            "sharing only an ID are retained."
        ),
    })

    # Step E: Save cleaned dataset.
    output_path = CLEAN_DIR / filename

    if filename.lower().endswith(".csv"):
        dataframe.to_csv(output_path, index=False)

    else:
        dataframe.to_json(
            output_path,
            orient="records",
            date_format="iso",
            indent=2,
        )

    final_missing = int(
        dataframe.isna().sum().sum()
    )

    summary = {
        "dataset": name,
        "source_file": filename,
        "original_rows": original_rows,
        "cleaned_rows": len(dataframe),
        "rows_removed_as_exact_duplicates": duplicate_count,
        "original_missing_cells": original_missing,
        "final_missing_cells": final_missing,
        "columns": len(dataframe.columns),
        "output_file": str(output_path),
    }

    logging.info(
        "%s: %s -> %s rows; removed %s exact duplicates.",
        name,
        original_rows,
        len(dataframe),
        duplicate_count,
    )

    return summary, changes


# --------------------------------------------------
# 5. RUN ALL EIGHT DATASETS
# --------------------------------------------------

def main():
    summaries = []
    cleaning_log = []
    failed_datasets = []

    for name, config in DATASETS.items():
        try:
            summary, changes = clean_dataset(
                name,
                config["file"],
                config["dates"],
            )

            summaries.append(summary)
            cleaning_log.extend(changes)

        except (
            OSError,
            ValueError,
            TypeError,
            pd.errors.ParserError,
        ) as error:
            failed_datasets.append(name)

            logging.error(
                "%s could not be cleaned: %s",
                name,
                error,
            )

    # Save reports even if a dataset fails.
    summary_path = (
        REPORT_DIR / "module_03_cleaning_summary.csv"
    )
    log_path = (
        REPORT_DIR / "module_03_cleaning_log.csv"
    )

    pd.DataFrame(summaries).to_csv(
        summary_path,
        index=False,
    )

    pd.DataFrame(cleaning_log).to_csv(
        log_path,
        index=False,
    )

    print("\n" + "=" * 55)
    print("MODULE 3: DATA CLEANING SUMMARY")
    print("=" * 55)

    if summaries:
        print(
            pd.DataFrame(summaries).to_string(index=False)
        )

    print(f"\nSummary saved: {summary_path}")
    print(f"Cleaning log saved: {log_path}")
    print(f"Cleaned files saved in: {CLEAN_DIR}")

    print(
        f"\nSuccessfully processed: "
        f"{len(summaries)}/{len(DATASETS)} datasets"
    )

    if failed_datasets:
        print("Failed datasets:", ", ".join(failed_datasets))
        raise RuntimeError(
            "Module 3 requires attention. "
            "Check the errors above before proceeding."
        )


if __name__ == "__main__":
    main()
