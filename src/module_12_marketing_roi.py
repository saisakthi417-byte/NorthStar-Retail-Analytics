# ============================================================
# NORTHSTAR RETAIL ANALYTICS PLATFORM
# MODULE 12: MARKETING ROI ANALYSIS
# ============================================================

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CAMPAIGNS_FILE = ROOT / "DATA" / "raw" / "campaigns.json"

TRANSACTIONS_FILE = (
    ROOT / "outputs" / "integrated" /
    "integrated_transactions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "module_12"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_campaigns(file_path):
    """Load marketing campaigns from JSON."""

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("campaigns"), list):
        records = data["campaigns"]
    elif isinstance(data, dict) and isinstance(data.get("data"), list):
        records = data["data"]
    elif isinstance(data, dict):
        records = [data]
    else:
        raise ValueError("Unsupported campaigns JSON format.")

    return pd.json_normalize(records)


def main():

    print("=" * 65)
    print("NORTHSTAR RETAIL ANALYTICS PLATFORM")
    print("MODULE 12: MARKETING ROI ANALYSIS")
    print("=" * 65)

    # --------------------------------------------------------
    # 2.1 Check input files
    # --------------------------------------------------------

    if not CAMPAIGNS_FILE.exists():
        raise FileNotFoundError(
            f"Campaign file not found: {CAMPAIGNS_FILE}"
        )

    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Integrated transactions file not found: "
            f"{TRANSACTIONS_FILE}"
        )

    # --------------------------------------------------------
    # 2.2 Load datasets
    # --------------------------------------------------------

    campaigns = load_campaigns(CAMPAIGNS_FILE)

    transactions = pd.read_csv(TRANSACTIONS_FILE)

    print(f"\nCampaign records loaded: {len(campaigns):,}")
    print(f"Transaction records loaded: {len(transactions):,}")

    # --------------------------------------------------------
    # 3. DATA VALIDATION
    # --------------------------------------------------------

    required_campaign_columns = [
        "campaign_id",
        "name",
        "channel",
        "start_date",
        "end_date",
        "spend_inr",
        "impressions",
        "clicks",
        "target_category",
    ]

    required_transaction_columns = [
        "TransactionID",
        "OrderDate",
        "Category",
        "LineAmount",
    ]

    missing_campaign_columns = [
        col for col in required_campaign_columns
        if col not in campaigns.columns
    ]

    missing_transaction_columns = [
        col for col in required_transaction_columns
        if col not in transactions.columns
    ]

    if missing_campaign_columns:
        raise ValueError(
            f"Missing campaign columns: {missing_campaign_columns}"
        )

    if missing_transaction_columns:
        raise ValueError(
            f"Missing transaction columns: "
            f"{missing_transaction_columns}"
        )

    print("\nData validation completed.")

    # --------------------------------------------------------
    # 4. DATA CLEANING
    # --------------------------------------------------------

    campaigns = campaigns.copy()
    transactions = transactions.copy()

    campaigns["start_date"] = pd.to_datetime(
        campaigns["start_date"], errors="coerce"
    )

    campaigns["end_date"] = pd.to_datetime(
        campaigns["end_date"], errors="coerce"
    )

    transactions["OrderDate"] = pd.to_datetime(
        transactions["OrderDate"], errors="coerce"
    )

    for col in ["spend_inr", "impressions", "clicks"]:
        campaigns[col] = pd.to_numeric(
            campaigns[col], errors="coerce"
        )

    campaigns["spend_inr"] = campaigns["spend_inr"].fillna(0)
    campaigns["impressions"] = campaigns["impressions"].fillna(0)
    campaigns["clicks"] = campaigns["clicks"].fillna(0)

    transactions["LineAmount"] = pd.to_numeric(
        transactions["LineAmount"], errors="coerce"
    )

    transactions = transactions.dropna(
        subset=["OrderDate", "LineAmount", "Category"]
    )

    campaigns["target_category_clean"] = (
        campaigns["target_category"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    transactions["category_clean"] = (
        transactions["Category"]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    print("Data cleaning completed.")

    # --------------------------------------------------------
    # 5. CAMPAIGN-LEVEL ANALYSIS
    # --------------------------------------------------------

    campaign_results = []
    matched_transactions = []

    for _, campaign in campaigns.iterrows():

        start_date = campaign["start_date"]
        end_date = campaign["end_date"]

        target_category = campaign["target_category_clean"]

        spend = float(campaign["spend_inr"])
        clicks = float(campaign["clicks"])
        impressions = float(campaign["impressions"])

        # Calculate CTR.
        if impressions > 0:
            ctr = (clicks / impressions) * 100
        else:
            ctr = np.nan

        # Match transactions to the campaign date window.
        if (
            pd.notna(start_date)
            and pd.notna(end_date)
            and end_date >= start_date
        ):

            date_mask = transactions["OrderDate"].between(
                start_date,
                end_date,
                inclusive="both",
            )

            # "All" campaigns include every product category.
            if target_category == "all":
                category_mask = pd.Series(
                    True,
                    index=transactions.index,
                )
            else:
                category_mask = (
                    transactions["category_clean"]
                    == target_category
                )

            matched = transactions.loc[
                date_mask & category_mask
            ].copy()

        else:
            matched = transactions.iloc[0:0].copy()

        estimated_revenue = float(
            matched["LineAmount"].sum()
        )

        # Calculate exploratory ROI estimate.
        if spend > 0:
            roi = (
                (estimated_revenue - spend)
                / spend
            ) * 100
        else:
            roi = np.nan

        # CPA cannot be calculated without acquisition counts.
        cpa = np.nan

        campaign_results.append({
            "campaign_id": campaign["campaign_id"],
            "campaign_name": campaign["name"],
            "marketing_channel": campaign["channel"],
            "target_category": campaign["target_category"],
            "start_date": start_date,
            "end_date": end_date,
            "marketing_spend_inr": spend,
            "clicks": clicks,
            "impressions": impressions,
            "ctr_percent": ctr,
            "estimated_period_category_revenue_inr":
                estimated_revenue,
            "roi_proxy_percent": roi,
            "cpa_inr": cpa,
            "matched_transaction_count": len(matched),
            "attribution_note":
                "Estimated date/category revenue; not confirmed "
                "campaign attribution",
        })

        # Preserve matched transaction details for audit.
        if not matched.empty:

            matched_export = matched[
                [
                    "TransactionID",
                    "OrderDate",
                    "Category",
                    "LineAmount",
                ]
            ].copy()

            matched_export.insert(
                0, "campaign_id", campaign["campaign_id"]
            )

            matched_export.insert(
                1, "campaign_name", campaign["name"]
            )

            matched_export.insert(
                2, "marketing_channel", campaign["channel"]
            )

            matched_transactions.append(matched_export)

    campaign_output = pd.DataFrame(campaign_results)

    # --------------------------------------------------------
    # 6. CHANNEL-LEVEL ANALYSIS
    # --------------------------------------------------------

    channel_output = (
        campaigns.groupby("channel", dropna=False)
        .agg(
            campaign_count=("campaign_id", "nunique"),
            marketing_spend_inr=("spend_inr", "sum"),
            total_clicks=("clicks", "sum"),
            total_impressions=("impressions", "sum"),
        )
        .reset_index()
        .rename(columns={"channel": "marketing_channel"})
    )

    # CTR must be calculated from channel totals.
    channel_output["ctr_percent"] = np.where(
        channel_output["total_impressions"] > 0,
        (
            channel_output["total_clicks"]
            / channel_output["total_impressions"]
        ) * 100,
        np.nan,
    )

    revenue_by_channel = (
        campaign_output.groupby(
            "marketing_channel",
            dropna=False,
        )
        .agg(
            estimated_revenue_inr=(
                "estimated_period_category_revenue_inr",
                "sum",
            ),
        )
        .reset_index()
    )

    channel_output = channel_output.merge(
        revenue_by_channel,
        on="marketing_channel",
        how="left",
        validate="one_to_one",
    )

    # Exploratory ROI estimate, not verified attribution.
    channel_output["roi_proxy_percent"] = np.where(
        channel_output["marketing_spend_inr"] > 0,
        (
            (
                channel_output["estimated_revenue_inr"]
                - channel_output["marketing_spend_inr"]
            )
            / channel_output["marketing_spend_inr"]
        ) * 100,
        np.nan,
    )

    # CPA requires acquisition counts, which are not supplied.
    channel_output["acquisition_count"] = np.nan
    channel_output["cpa_inr"] = np.nan

    channel_output["roi_interpretation"] = (
        "Exploratory estimate; not confirmed campaign attribution"
    )

    channel_output["cpa_interpretation"] = (
        "Unavailable: acquisition counts not supplied"
    )

    # --------------------------------------------------------
    # 7. SAVE OUTPUT FILES
    # --------------------------------------------------------

    campaign_file = (
        OUTPUT_DIR / "module_12_campaign_roi.csv"
    )

    channel_file = (
        OUTPUT_DIR / "module_12_channel_roi.csv"
    )

    matched_file = (
        OUTPUT_DIR / "module_12_matched_transactions.csv"
    )

    report_file = (
        REPORT_DIR / "module_12_marketing_roi.txt"
    )

    campaign_output.to_csv(
        campaign_file, index=False
    )

    channel_output.to_csv(
        channel_file, index=False
    )

    if matched_transactions:
        matched_output = pd.concat(
            matched_transactions,
            ignore_index=True,
        )
    else:
        matched_output = pd.DataFrame(
            columns=[
                "campaign_id",
                "campaign_name",
                "marketing_channel",
                "TransactionID",
                "OrderDate",
                "Category",
                "LineAmount",
            ]
        )

    matched_output.to_csv(
        matched_file, index=False
    )

    # --------------------------------------------------------
    # 8. SUMMARY REPORT
    # --------------------------------------------------------

    total_spend = campaigns["spend_inr"].sum()
    total_clicks = campaigns["clicks"].sum()
    total_impressions = campaigns["impressions"].sum()

    if total_impressions > 0:
        overall_ctr = (
            total_clicks / total_impressions
        ) * 100
    else:
        overall_ctr = np.nan

    report_lines = [
        "=" * 65,
        "NORTHSTAR RETAIL ANALYTICS PLATFORM",
        "MODULE 12: MARKETING ROI ANALYSIS",
        "=" * 65,
        "",
        f"Campaigns analysed: {len(campaigns):,}",
        f"Transactions available: {len(transactions):,}",
        f"Total marketing spend (INR): {total_spend:,.2f}",
        f"Total clicks: {total_clicks:,.0f}",
        f"Total impressions: {total_impressions:,.0f}",
        f"Overall CTR: {overall_ctr:.2f}%",
        "",
        "CHANNEL RESULTS",
        "-" * 65,
        channel_output.to_string(index=False),
        "",
        "FORMULAS",
        "-" * 65,
        "CTR (%) = Clicks / Impressions * 100",
        "ROI (%) = (Revenue - Marketing Spend) / Marketing Spend * 100",
        "CPA = Marketing Spend / Acquisition Count",
        "",
        "DATA LIMITATIONS",
        "-" * 65,
        "Marketing channels differ from transaction sales channels.",
        "Transactions do not contain a campaign ID.",
        "Revenue is estimated using campaign dates and target categories.",
        "Campaign date windows may overlap and double-count transactions.",
        "ROI is an exploratory estimate, not verified campaign ROI.",
        "CPA is unavailable because acquisition counts were not supplied.",
        "Revenue minus spend is not net profit.",
        "",
        "OUTPUT FILES",
        "-" * 65,
        str(campaign_file),
        str(channel_file),
        str(matched_file),
        str(report_file),
        "",
        "Module 12 processing completed.",
    ]

    report_file.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 9. OUTPUT VALIDATION
    # --------------------------------------------------------

    checks = {
        "Campaign output has one row per campaign":
            len(campaign_output) == len(campaigns),

        "Channel output was generated":
            len(channel_output) > 0,

        "Campaign CTR values are within 0-100%":
            campaign_output["ctr_percent"]
            .dropna()
            .between(0, 100)
            .all(),

        "Campaign CSV exists":
            campaign_file.exists(),

        "Channel CSV exists":
            channel_file.exists(),

        "Matched transactions CSV exists":
            matched_file.exists(),

        "Summary report exists":
            report_file.exists(),
    }

    print("\n" + "=" * 65)
    print("MODULE 12 OUTPUT VALIDATION")
    print("=" * 65)

    all_passed = True

    for description, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {description}")
        all_passed = all_passed and passed

    print("\nCHANNEL PERFORMANCE")
    print("-" * 65)
    print(channel_output.to_string(index=False))

    print("\nOUTPUT FILES CREATED")
    print(campaign_file)
    print(channel_file)
    print(matched_file)
    print(report_file)

    if not all_passed:
        raise RuntimeError(
            "One or more output validation checks failed."
        )

    print("\nMODULE 12 COMPLETED SUCCESSFULLY!")
    print("Note: ROI is estimated; CPA is unavailable from supplied data.")


# ============================================================
# 10. EXECUTION
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)