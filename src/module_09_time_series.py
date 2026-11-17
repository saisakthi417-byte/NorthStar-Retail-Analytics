
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# ============================================================
# MODULE 9: TIME SERIES ANALYSIS AND FORECASTING
# Project: NorthStar Retail Analytics Platform
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "outputs" / "integrated" / "integrated_transactions.csv"
)
OUTPUT_DIR = BASE_DIR / "outputs" / "module_09"
REPORT_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    force=True,
)

# The script checks these common date column names in order.
DATE_CANDIDATES = [
    "OrderDate",
    "TransactionDate",
    "Transaction_Date",
    "Date",
]

SALES_COLUMN = "LineAmount"
FORECAST_MONTHS = 3


def find_date_column(df):
    """Find a recognised transaction date column."""
    for column in DATE_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "No recognised date column was found. "
        f"Available columns: {list(df.columns)}. "
        f"Expected one of: {DATE_CANDIDATES}"
    )


def save_csv(dataframe, filename):
    dataframe.to_csv(OUTPUT_DIR / filename, index=False)


def main():
    # STEP 1: LOAD AND VALIDATE INPUT

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Integrated dataset not found: {INPUT_FILE}\n"
            "Run Module 4 first."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The integrated dataset is empty.")

    if SALES_COLUMN not in df.columns:
        raise ValueError(
            f"Required sales column '{SALES_COLUMN}' is missing. "
            f"Available columns: {list(df.columns)}"
        )

    date_column = find_date_column(df)

    df[date_column] = pd.to_datetime(
        df[date_column], errors="coerce", dayfirst=True
    )
    df[SALES_COLUMN] = pd.to_numeric(
        df[SALES_COLUMN], errors="coerce"
    )

    valid = df.dropna(subset=[date_column, SALES_COLUMN]).copy()

    if valid.empty:
        raise ValueError(
            "No rows have both a valid transaction date and LineAmount."
        )

    # A transaction's LineAmount is assumed to be the sales measure.
    # Confirm its business definition before interpreting totals as revenue.
    valid["Month"] = (
        valid[date_column].dt.to_period("M").dt.to_timestamp()
    )

    monthly = (
        valid.groupby("Month", as_index=False)
        .agg(
            Monthly_Sales=(SALES_COLUMN, "sum"),
            Transaction_Rows=(SALES_COLUMN, "size"),
        )
        .sort_values("Month")
        .reset_index(drop=True)
    )

    if len(monthly) < 12:
        raise ValueError(
            f"Only {len(monthly)} months of valid sales data are available. "
            "At least 12 monthly observations are required for this "
            "annual-seasonality analysis. Review date parsing and source data."
        )

    # Include missing calendar months as missing, not zero sales.
    # Missing months must be investigated before relying on the forecast.
    full_month_index = pd.date_range(
        monthly["Month"].min(),
        monthly["Month"].max(),
        freq="MS",
    )

    monthly = (
        monthly.set_index("Month")
        .reindex(full_month_index)
        .rename_axis("Month")
        .reset_index()
    )

    monthly["Month_Number"] = monthly["Month"].dt.month
    monthly["Month_Label"] = monthly["Month"].dt.strftime("%Y-%m")

    missing_months = int(monthly["Monthly_Sales"].isna().sum())

    if missing_months:
        logging.warning(
            "%s calendar months have no usable sales observations. "
            "They remain missing and are excluded from model fitting.",
            missing_months,
        )

    model_data = monthly.dropna(subset=["Monthly_Sales"]).copy()
    model_data = model_data.sort_values("Month").reset_index(drop=True)

    if len(model_data) < 12:
        raise ValueError(
            "Fewer than 12 valid monthly sales observations remain "
            "after validation."
        )

    save_csv(monthly, "monthly_sales.csv")

    # STEP 2: FIT A LINEAR TREND

    # Time index is based on the actual calendar month position.
    first_month = monthly["Month"].min()
    model_data["Time_Index"] = (
        (model_data["Month"].dt.year - first_month.year) * 12
        + (model_data["Month"].dt.month - first_month.month)
    )

    x = model_data["Time_Index"].to_numpy(dtype=float)
    y = model_data["Monthly_Sales"].to_numpy(dtype=float)

    trend_slope, trend_intercept = np.polyfit(x, y, 1)
    model_data["Trend"] = trend_intercept + trend_slope * x

    # STEP 3: ESTIMATE MONTH-OF-YEAR SEASONALITY

    # Additive decomposition:
    # observed sales = fitted trend + seasonal effect + residual.
    model_data["Detrended_Sales"] = (
        model_data["Monthly_Sales"] - model_data["Trend"]
    )

    seasonal_effects = (
        model_data.groupby(model_data["Month"].dt.month)[
            "Detrended_Sales"
        ]
        .mean()
        .reindex(range(1, 13))
    )

    # If a month of year is not observed, its seasonal effect is unknown.
    if seasonal_effects.isna().any():
        missing_seasonal_months = seasonal_effects[
            seasonal_effects.isna()
        ].index.tolist()

        raise ValueError(
            "Seasonal effects cannot be estimated for every calendar month. "
            f"Missing month numbers: {missing_seasonal_months}. "
            "More complete seasonal data is needed."
        )

    # Center the additive seasonal effects so their annual mean is zero.
    seasonal_effects = seasonal_effects - seasonal_effects.mean()

    model_data["Seasonal"] = model_data["Month"].dt.month.map(
        seasonal_effects
    )
    model_data["Residual"] = (
        model_data["Monthly_Sales"]
        - model_data["Trend"]
        - model_data["Seasonal"]
    )

    # STEP 4: SAVE DECOMPOSITION

    decomposition = monthly.merge(
        model_data[
            ["Month", "Trend", "Seasonal", "Residual"]
        ],
        on="Month",
        how="left",
    )

    decomposition["Decomposition_Method"] = (
        "Additive linear trend plus average month-of-year seasonal effect"
    )

    save_csv(decomposition, "sales_decomposition.csv")

    # STEP 5: FORECAST THE NEXT THREE CALENDAR MONTHS

    last_month = monthly["Month"].max()
    future_months = pd.date_range(
        last_month + pd.offsets.MonthBegin(1),
        periods=FORECAST_MONTHS,
        freq="MS",
    )

    future_index = (
        (future_months.year - first_month.year) * 12
        + (future_months.month - first_month.month)
    ).astype(float)

    future_trend = trend_intercept + trend_slope * future_index
    future_seasonal = np.array([
        seasonal_effects.loc[month]
        for month in future_months.month
    ])

    forecasts = future_trend + future_seasonal

    # Approximate prediction range based on residual variability.
    # This is not a formal model-based confidence interval.
    residuals = model_data["Residual"].dropna().to_numpy(dtype=float)

    if len(residuals) >= 2:
        residual_std = float(np.std(residuals, ddof=1))
    else:
        residual_std = 0.0

    lower_bound = forecasts - 1.96 * residual_std
    upper_bound = forecasts + 1.96 * residual_std

    forecast_df = pd.DataFrame({
        "Forecast_Month": future_months,
        "Forecast_Sales": forecasts,
        "Trend_Component": future_trend,
        "Seasonal_Component": future_seasonal,
        "Approx_95pct_Lower_Bound": lower_bound,
        "Approx_95pct_Upper_Bound": upper_bound,
        "Uncertainty_Method": (
            "Forecast plus/minus 1.96 times in-sample residual standard "
            "deviation; approximate prediction range, not a formal "
            "confidence interval"
        ),
    })

    save_csv(forecast_df, "next_quarter_sales_forecast.csv")

    # STEP 6: MODEL QUALITY AND DIRECTIONAL METRICS

    fitted = model_data["Trend"] + model_data["Seasonal"]
    actual = model_data["Monthly_Sales"].to_numpy(dtype=float)
    errors = actual - fitted.to_numpy(dtype=float)

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Percentage errors are omitted when actual monthly sales are zero.
    nonzero_mask = actual != 0

    if nonzero_mask.any():
        mape = float(
            np.mean(
                np.abs(errors[nonzero_mask] / actual[nonzero_mask])
            ) * 100
        )
    else:
        mape = np.nan

    first_actual = float(model_data["Monthly_Sales"].iloc[0])
    last_actual = float(model_data["Monthly_Sales"].iloc[-1])

    if first_actual != 0:
        observed_change_pct = (
            (last_actual - first_actual) / abs(first_actual)
        ) * 100
    else:
        observed_change_pct = np.nan

    # STEP 7: BUSINESS INTERPRETATION REPORT

    report_lines = [
        "NORTHSTAR RETAIL ANALYTICS PLATFORM",
        "MODULE 9: TIME SERIES ANALYSIS AND FORECASTING",
        "",
        f"Input dataset: {INPUT_FILE}",
        f"Date column used: {date_column}",
        f"Sales measure used: {SALES_COLUMN}",
        f"Valid transaction rows: {len(valid):,}",
        f"Monthly observations in calendar range: {len(monthly):,}",
        f"Usable monthly observations: {len(model_data):,}",
        f"Calendar months without usable sales: {missing_months:,}",
        "",
        "METHOD",
        "Monthly LineAmount totals were calculated from valid dates and "
        "numeric sales values. An additive model was used: fitted linear "
        "trend plus average month-of-year seasonal effect plus residual.",
        "",
        "TREND",
        f"Estimated linear trend slope: {trend_slope:,.2f} sales units per month.",
        f"Observed first-to-last monthly change: {observed_change_pct:,.2f}%."
        if pd.notna(observed_change_pct)
        else "Observed first-to-last percentage change was not calculated "
             "because the first monthly total was zero.",
        (
            "The fitted trend is upward."
            if trend_slope > 0
            else "The fitted trend is downward."
            if trend_slope < 0
            else "The fitted trend is approximately flat."
        ),
        "",
        "SEASONALITY",
        "Seasonal components represent average deviations for each "
        "calendar month after removing the fitted linear trend. They are "
        "descriptive estimates and may be unreliable if historical data "
        "covers only one annual cycle.",
        "",
        "FORECAST",
        "The next three calendar months were forecast using the fitted "
        "linear trend plus the estimated seasonal effect for each month.",
        "",
        "APPROXIMATE UNCERTAINTY",
        f"Residual standard deviation: {residual_std:,.2f}.",
        "The reported lower and upper bounds equal the forecast plus/minus "
        "1.96 times the in-sample residual standard deviation. This is a "
        "simplified uncertainty range, not a formal statistical confidence "
        "interval and not a guarantee of future outcomes.",
        "",
        "IN-SAMPLE FIT",
        f"Mean absolute error (MAE): {mae:,.2f}.",
        f"Root mean squared error (RMSE): {rmse:,.2f}.",
        f"Mean absolute percentage error (MAPE): {mape:,.2f}%."
        if pd.notna(mape)
        else "MAPE was not calculated because all actual monthly totals were zero.",
        "",
        "LIMITATIONS AND BUSINESS ACTION",
        "This baseline model does not explicitly account for promotions, "
        "price changes, holidays, stockouts, store openings, returns, or "
        "other external drivers. Review the historical data, missing months, "
        "residuals and business events before using the forecast for staffing, "
        "inventory or financial commitments. Confirm that LineAmount is the "
        "correct sales measure for the intended business decision.",
        "Missing monthly observations were not imputed. Investigate them "
        "before relying on the forecast.",
    ]

    report_file = REPORT_DIR / "module_09_forecast_interpretation.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    # STEP 8: SUMMARY AND OUTPUT VALIDATION

    expected_files = [
        OUTPUT_DIR / "monthly_sales.csv",
        OUTPUT_DIR / "sales_decomposition.csv",
        OUTPUT_DIR / "next_quarter_sales_forecast.csv",
        REPORT_DIR / "module_09_forecast_interpretation.txt",
    ]

    missing_outputs = [
        str(path) for path in expected_files if not path.exists()
    ]

    summary = pd.DataFrame([{
        "Project": "NorthStar Retail Analytics Platform",
        "Module": "Module 9 - Time Series Analysis and Forecasting",
        "Date_Column": date_column,
        "Sales_Column": SALES_COLUMN,
        "Valid_Transaction_Rows": len(valid),
        "Monthly_Observations": len(model_data),
        "Missing_Calendar_Months": missing_months,
        "Forecast_Months": FORECAST_MONTHS,
        "Trend_Slope_Per_Month": trend_slope,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape,
        "Verified_Output_Files": len(expected_files) - len(missing_outputs),
        "Status": (
            "OUTPUT VERIFICATION PASSED"
            if not missing_outputs
            else "OUTPUT VERIFICATION FAILED"
        ),
    }])

    summary.to_csv(
        REPORT_DIR / "module_09_summary.csv", index=False
    )

    if missing_outputs:
        raise RuntimeError(
            "Expected outputs are missing:\n"
            + "\n".join(missing_outputs)
        )

    print("\n" + "=" * 60)
    print("MODULE 9 ANALYSIS COMPLETED")
    print("=" * 60)
    print("Project: NorthStar Retail Analytics Platform")
    print(f"Date column: {date_column}")
    print(f"Sales column: {SALES_COLUMN}")
    print(f"Valid transaction rows: {len(valid):,}")
    print(f"Usable monthly observations: {len(model_data):,}")
    print(f"Missing calendar months: {missing_months:,}")
    print(f"Forecast months: {FORECAST_MONTHS}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary: {REPORT_DIR / 'module_09_summary.csv'}")
    print(f"Interpretation: {report_file}")
    print("Status: OUTPUT VERIFICATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
