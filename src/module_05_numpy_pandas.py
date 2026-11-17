
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# ============================================================
# MODULE 5: NUMPY AND PANDAS ANALYSIS
# Project: NorthStar Retail Analytics Platform
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "outputs" / "integrated" / "integrated_transactions.csv"
)
OUTPUT_DIR = BASE_DIR / "outputs" / "module_05"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)


def save_csv(dataframe, filename):
    """Save a DataFrame in the Module 5 output folder."""
    dataframe.to_csv(OUTPUT_DIR / filename, index=False)


def main():
    # STEP 1: LOAD AND VALIDATE DATA

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Integrated dataset not found: {INPUT_FILE}\n"
            "Run Module 4 before running Module 5."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The integrated dataset contains no rows.")

    required_columns = ["TransactionID", "LineAmount", "Quantity"]
    missing_required = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Required columns are missing: {missing_required}"
        )

    logging.info("Loaded integrated dataset: %s rows, %s columns",
                 *df.shape)

    # Convert relevant fields safely. Invalid values become missing;
    # no values are imputed.
    numeric_candidates = [
        "Quantity", "Discount", "LineAmount", "UnitCost", "MRP",
        "Age", "OnHandUnits", "ReorderLevel", "RefundAmount"
    ]

    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    original_missing = int(df.isna().sum().sum())

    # STEP 2: DATASET OVERVIEW AND MISSING VALUES

    overview = pd.DataFrame([
        {"Metric": "Total rows", "Value": len(df)},
        {"Metric": "Total columns", "Value": len(df.columns)},
        {"Metric": "Duplicate rows", "Value": int(df.duplicated().sum())},
        {"Metric": "Missing cells", "Value": original_missing},
        {
            "Metric": "Numeric columns",
            "Value": len(df.select_dtypes(include="number").columns)
        },
    ])
    save_csv(overview, "dataset_overview.csv")

    missing_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing_Values": df.isna().sum().values,
        "Missing_Percentage": (df.isna().mean().values * 100).round(2),
    }).sort_values("Missing_Values", ascending=False)

    save_csv(missing_summary, "missing_value_summary.csv")

    # STEP 3: NUMPY ARRAY OPERATIONS

    numeric_df = df.select_dtypes(include="number")
    numpy_results = []

    for column in numeric_df.columns:
        values = numeric_df[column].dropna().to_numpy(dtype=float)

        if values.size == 0:
            continue

        numpy_results.append({
            "Column": column,
            "Valid_Values": int(values.size),
            "Sum": float(np.sum(values)),
            "Mean": float(np.mean(values)),
            "Median": float(np.median(values)),
            "Minimum": float(np.min(values)),
            "Maximum": float(np.max(values)),
            "Standard_Deviation": float(np.std(values)),
            "25th_Percentile": float(np.percentile(values, 25)),
            "75th_Percentile": float(np.percentile(values, 75)),
        })

    save_csv(pd.DataFrame(numpy_results), "numpy_analysis.csv")

    numeric_summary = numeric_df.describe().T
    numeric_summary.index.name = "Column"
    numeric_summary.reset_index().to_csv(
        OUTPUT_DIR / "numeric_summary.csv", index=False
    )

    # STEP 4: CALCULATED COLUMNS

    calculated = df.copy()

    # Avoid division by zero. Undefined values remain missing.
    calculated["RevenuePerUnit"] = (
        calculated["LineAmount"]
        / calculated["Quantity"].where(calculated["Quantity"] != 0)
    )

    if "Discount" in calculated.columns:
        calculated["DiscountBand"] = pd.cut(
            calculated["Discount"],
            bins=[-np.inf, 0, 0.10, 0.25, np.inf],
            labels=[
                "Zero_or_Negative",
                "Low",
                "Medium",
                "High",
            ],
            include_lowest=True,
        )

    save_csv(
        calculated,
        "transactions_with_calculated_columns.csv"
    )

    # STEP 5: MULTI-LEVEL GROUPBY ANALYSIS

    group_columns = [
        column for column in ["Channel", "Category", "Region"]
        if column in df.columns
    ]

    if group_columns:
        groupby_summary = (
            df.groupby(group_columns, dropna=False, observed=True)
            .agg(
                Transaction_Rows=("TransactionID", "size"),
                Total_LineAmount=("LineAmount", "sum"),
                Average_LineAmount=("LineAmount", "mean"),
                Total_Quantity=("Quantity", "sum"),
            )
            .reset_index()
        )
    else:
        groupby_summary = pd.DataFrame()

    save_csv(groupby_summary, "multilevel_groupby_summary.csv")

    # STEP 6: PIVOT TABLES

    if "Channel" in df.columns and "Category" in df.columns:
        sales_pivot = pd.pivot_table(
            df,
            index="Channel",
            columns="Category",
            values="LineAmount",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        sales_pivot.to_csv(OUTPUT_DIR / "sales_pivot_table.csv")
    elif "Channel" in df.columns:
        sales_pivot = pd.pivot_table(
            df,
            index="Channel",
            values="LineAmount",
            aggfunc="sum",
            fill_value=0,
        )
        sales_pivot.to_csv(OUTPUT_DIR / "sales_pivot_table.csv")
    else:
        logging.warning(
            "Channel column unavailable; sales pivot cannot be grouped by channel."
        )
        pd.DataFrame().to_csv(
            OUTPUT_DIR / "sales_pivot_table.csv", index=False
        )

    if "StoreName" in df.columns and "Channel" in df.columns:
        store_pivot = pd.pivot_table(
            df,
            index="StoreName",
            columns="Channel",
            values="LineAmount",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        store_pivot.to_csv(OUTPUT_DIR / "store_channel_pivot.csv")
    else:
        logging.info(
            "StoreName or Channel unavailable; store/channel pivot skipped."
        )

    # STEP 7: IQR OUTLIER DETECTION
    # Flag outliers for review; do not automatically remove them.

    outlier_results = []

    for column in ["LineAmount", "Quantity", "Discount"]:
        if column not in df.columns:
            continue

        values = df[column].dropna()

        if len(values) < 4:
            continue

        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (values < lower_bound) | (values > upper_bound)

        outlier_results.append({
            "Column": column,
            "Valid_Values": int(len(values)),
            "Q1": q1,
            "Q3": q3,
            "IQR": float(iqr),
            "Lower_Bound": float(lower_bound),
            "Upper_Bound": float(upper_bound),
            "Outlier_Count": int(outlier_mask.sum()),
            "Outlier_Percentage": float(outlier_mask.mean() * 100),
            "Treatment": "Flagged for investigation; not removed",
        })

    save_csv(
        pd.DataFrame(outlier_results),
        "iqr_outlier_summary.csv"
    )

    # STEP 8: CORRELATION ANALYSIS

    correlation_candidates = [
        "Quantity", "Discount", "LineAmount", "UnitCost", "MRP", "Age"
    ]
    correlation_columns = [
        column for column in correlation_candidates
        if column in df.columns
        and df[column].notna().sum() > 1
        and df[column].nunique(dropna=True) > 1
    ]

    pairs = []
    correlation_file = OUTPUT_DIR / "correlation_matrix.csv"

    if len(correlation_columns) >= 2:
        correlation = df[correlation_columns].corr(method="pearson")
        correlation.to_csv(correlation_file)

        for i, col1 in enumerate(correlation.columns):
            for col2 in correlation.columns[i + 1:]:
                value = correlation.loc[col1, col2]
                pairs.append({
                    "Variable_1": col1,
                    "Variable_2": col2,
                    "Pearson_Correlation": value,
                    "Absolute_Correlation": (
                        abs(value) if pd.notna(value) else np.nan
                    ),
                })

        pairs_df = pd.DataFrame(pairs)

        if not pairs_df.empty:
            pairs_df = pairs_df.sort_values(
                "Absolute_Correlation", ascending=False,
                na_position="last"
            )
    else:
        logging.warning(
            "Not enough varying numeric variables for correlation analysis."
        )
        pd.DataFrame().to_csv(correlation_file)
        pairs_df = pd.DataFrame(columns=[
            "Variable_1", "Variable_2",
            "Pearson_Correlation", "Absolute_Correlation"
        ])

    save_csv(pairs_df, "correlation_pairs.csv")

    # STEP 9: BUSINESS INTERPRETATION

    interpretations = [
        "MODULE 5: NUMPY AND PANDAS BUSINESS INTERPRETATION",
        f"Dataset contains {len(df):,} rows and {len(df.columns)} columns.",
        f"Duplicate rows detected: {int(df.duplicated().sum()):,}.",
        f"Missing cells detected: {original_missing:,}.",
        "Missing values were not imputed. Investigate each field and "
        "document a business reason before any imputation.",
    ]

    valid_amounts = df["LineAmount"].dropna()

    if not valid_amounts.empty:
        interpretations.append(
            f"LineAmount has {len(valid_amounts):,} valid values; "
            f"total={valid_amounts.sum():,.2f}; "
            f"average={valid_amounts.mean():,.2f}. "
            "Confirm the field definition before interpreting it as "
            "net sales, revenue, or profit."
        )

    if not pairs_df.empty:
        valid_pairs = pairs_df.dropna(
            subset=["Pearson_Correlation"]
        )

        if not valid_pairs.empty:
            strongest = valid_pairs.iloc[0]
            r = float(strongest["Pearson_Correlation"])

            strength = (
                "strong" if abs(r) >= 0.7
                else "moderate" if abs(r) >= 0.4
                else "weak"
            )
            direction = (
                "positive" if r > 0
                else "negative" if r < 0
                else "approximately zero"
            )

            interpretations.append(
                f"The strongest absolute Pearson correlation is between "
                f"{strongest['Variable_1']} and {strongest['Variable_2']} "
                f"(r={r:.3f}), indicating a {strength} {direction} "
                "linear association. Correlation does not prove causation."
            )
        else:
            interpretations.append(
                "No valid correlation pairs were available for interpretation."
            )
    else:
        interpretations.append(
            "Correlation analysis could not produce variable pairs; "
            "check available numeric fields and variation."
        )

    if outlier_results:
        total_outliers = sum(
            row["Outlier_Count"] for row in outlier_results
        )
        interpretations.append(
            f"IQR analysis flagged {total_outliers:,} column-level "
            "outlier observations. These are candidates for investigation, "
            "not automatically errors; they were not removed."
        )
    else:
        interpretations.append(
            "IQR analysis did not have enough valid observations in the "
            "selected fields to report outliers."
        )

    if not groupby_summary.empty:
        interpretations.append(
            "The multi-level groupby report summarises transaction rows, "
            "LineAmount and Quantity by the available Channel, Category "
            "and Region dimensions."
        )

    interpretations.append(
        "Pivot tables summarise LineAmount across available business "
        "dimensions. Confirm field definitions and missing-data patterns "
        "before making business decisions."
    )

    interpretation_file = (
        REPORT_DIR / "module_05_business_interpretation.txt"
    )
    interpretation_file.write_text(
        "\n\n".join(interpretations), encoding="utf-8"
    )

    # STEP 10: SUMMARY AND OUTPUT VERIFICATION

    expected_files = [
        OUTPUT_DIR / "dataset_overview.csv",
        OUTPUT_DIR / "missing_value_summary.csv",
        OUTPUT_DIR / "numpy_analysis.csv",
        OUTPUT_DIR / "numeric_summary.csv",
        OUTPUT_DIR / "transactions_with_calculated_columns.csv",
        OUTPUT_DIR / "multilevel_groupby_summary.csv",
        OUTPUT_DIR / "sales_pivot_table.csv",
        OUTPUT_DIR / "iqr_outlier_summary.csv",
        OUTPUT_DIR / "correlation_matrix.csv",
        OUTPUT_DIR / "correlation_pairs.csv",
        REPORT_DIR / "module_05_business_interpretation.txt",
    ]

    missing_outputs = [
        str(path) for path in expected_files if not path.exists()
    ]

    summary = pd.DataFrame([{
        "Project": "NorthStar Retail Analytics Platform",
        "Module": "Module 5 - NumPy and Pandas",
        "Input_Rows": len(df),
        "Input_Columns": len(df.columns),
        "Numeric_Columns_Analysed": len(numeric_df.columns),
        "Duplicate_Rows": int(df.duplicated().sum()),
        "Missing_Cells": original_missing,
        "NumPy_Analysis_Created": (
            OUTPUT_DIR / "numpy_analysis.csv"
        ).exists(),
        "Groupby_Created": (
            OUTPUT_DIR / "multilevel_groupby_summary.csv"
        ).exists(),
        "Sales_Pivot_Created": (
            OUTPUT_DIR / "sales_pivot_table.csv"
        ).exists(),
        "IQR_Report_Created": (
            OUTPUT_DIR / "iqr_outlier_summary.csv"
        ).exists(),
        "Correlation_Matrix_Created": correlation_file.exists(),
        "Calculated_Columns_Created": (
            OUTPUT_DIR / "transactions_with_calculated_columns.csv"
        ).exists(),
        "Missing_Expected_Outputs": len(missing_outputs),
        "Status": (
            "OUTPUT VERIFICATION PASSED"
            if not missing_outputs else "OUTPUT VERIFICATION FAILED"
        ),
    }])

    summary.to_csv(
        REPORT_DIR / "module_05_summary.csv", index=False
    )

    if missing_outputs:
        raise RuntimeError(
            "Expected outputs are missing:\n" + "\n".join(missing_outputs)
        )

    print("\n" + "=" * 60)
    print("MODULE 5 ANALYSIS COMPLETED")
    print("=" * 60)
    print("Project: NorthStar Retail Analytics Platform")
    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {len(df.columns)}")
    print(f"Numeric columns: {len(numeric_df.columns)}")
    print(f"Duplicate rows: {int(df.duplicated().sum()):,}")
    print(f"Missing cells: {original_missing:,}")
    print(f"Verified output files: {len(expected_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary report: {REPORT_DIR / 'module_05_summary.csv'}")
    print(f"Interpretation: {interpretation_file}")
    print("Status: OUTPUT VERIFICATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
