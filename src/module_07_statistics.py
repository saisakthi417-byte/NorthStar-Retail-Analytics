
from pathlib import Path
import pandas as pd
from scipy import stats

# Project paths
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_FOLDER / "outputs" / "integrated" / "integrated_transactions.csv"
REPORT_FOLDER = PROJECT_FOLDER / "reports"
REPORT_FILE = REPORT_FOLDER / "module_07_statistics_summary.txt"

# Load data
df = pd.read_csv(DATA_FILE)

# Prepare columns
df["LineAmount"] = pd.to_numeric(df["LineAmount"], errors="coerce")
df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce")
df["Channel"] = df["Channel"].astype(str).str.strip().str.title()

sales = df["LineAmount"].dropna()

print("=" * 50)
print("NORTHSTAR RETAIL ANALYTICS PLATFORM")
print("MODULE 7 - STATISTICS AND HYPOTHESIS TESTING")
print("=" * 50)

# Descriptive statistics
print("\n1. DESCRIPTIVE STATISTICS")
print("Valid sales rows:", len(sales))
print("Mean:", round(sales.mean(), 2))
print("Median:", round(sales.median(), 2))
print("Standard deviation:", round(sales.std(), 2))
print("Minimum:", round(sales.min(), 2))
print("Maximum:", round(sales.max(), 2))

# Hypothesis test 1: Discount vs sales amount
discounted = df.loc[df["Discount"] > 0, "LineAmount"].dropna()
non_discounted = df.loc[df["Discount"] == 0, "LineAmount"].dropna()

print("\n2. HYPOTHESIS TEST: DISCOUNT VS SALES AMOUNT")
print("Discounted rows:", len(discounted))
print("Non-discounted rows:", len(non_discounted))

if len(discounted) >= 2 and len(non_discounted) >= 2:
    test1 = stats.ttest_ind(
        discounted,
        non_discounted,
        equal_var=False
    )

    print("Discounted average:", round(discounted.mean(), 2))
    print("Non-discounted average:", round(non_discounted.mean(), 2))
    print("T-statistic:", round(test1.statistic, 4))
    print("P-value:", f"{test1.pvalue:.8g}")

    if test1.pvalue < 0.05:
        test1_result = (
            "Statistically significant difference detected "
            "between discounted and non-discounted sales amounts."
        )
    else:
        test1_result = (
            "No statistically significant difference detected "
            "between discounted and non-discounted sales amounts."
        )

    print("Result:", test1_result)
else:
    test1_result = "Insufficient data to perform the discount test."
    print(test1_result)

# Hypothesis test 2: Sales amount across channels
print("\n3. HYPOTHESIS TEST: SALES AMOUNT BY CHANNEL")

channel_groups = {}
for channel in ["Store", "Online", "App"]:
    values = df.loc[df["Channel"] == channel, "LineAmount"].dropna()
    channel_groups[channel] = values
    print(channel, "rows:", len(values), "| average:", round(values.mean(), 2))

if all(len(values) >= 2 for values in channel_groups.values()):
    test2 = stats.f_oneway(
        channel_groups["Store"],
        channel_groups["Online"],
        channel_groups["App"]
    )

    print("ANOVA F-statistic:", round(test2.statistic, 4))
    print("ANOVA p-value:", f"{test2.pvalue:.8g}")

    if test2.pvalue < 0.05:
        test2_result = (
            "At least one channel mean differs significantly "
            "at the 5% significance level."
        )
    else:
        test2_result = (
            "No statistically significant difference detected "
            "between channel means at the 5% significance level."
        )

    print("Result:", test2_result)
else:
    test2_result = "Insufficient data to perform the channel test."
    print(test2_result)

# Save report
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

report = [
    "NORTHSTAR RETAIL ANALYTICS PLATFORM",
    "MODULE 7 - STATISTICS AND HYPOTHESIS TESTING",
    "",
    "DESCRIPTIVE STATISTICS",
    f"Valid sales rows: {len(sales)}",
    f"Mean LineAmount: {sales.mean():.2f}",
    f"Median LineAmount: {sales.median():.2f}",
    f"Standard deviation: {sales.std():.2f}",
    f"Minimum LineAmount: {sales.min():.2f}",
    f"Maximum LineAmount: {sales.max():.2f}",
    "",
    "HYPOTHESIS TEST 1: DISCOUNT VS SALES AMOUNT",
    f"Discounted rows: {len(discounted)}",
    f"Non-discounted rows: {len(non_discounted)}",
    f"Discounted average: {discounted.mean():.2f}",
    f"Non-discounted average: {non_discounted.mean():.2f}",
    f"P-value: {test1.pvalue:.8g}" if len(discounted) >= 2 and len(non_discounted) >= 2 else "P-value: Not calculated",
    f"Conclusion: {test1_result}",
    "",
    "HYPOTHESIS TEST 2: SALES AMOUNT BY CHANNEL",
    f"Store average: {channel_groups['Store'].mean():.2f}",
    f"Online average: {channel_groups['Online'].mean():.2f}",
    f"App average: {channel_groups['App'].mean():.2f}",
    f"Conclusion: {test2_result}",
    "",
    "INTERPRETATION AND LIMITATIONS",
    "A p-value below 0.05 is treated as statistically significant.",
    "Statistical association does not establish causation.",
    "The tests compare recorded LineAmount values as supplied.",
    "Results do not establish profitability.",
    "The tests assume observations are sufficiently independent.",
    "The channel ANOVA identifies an overall difference only; it does not identify which pair of channels differs.",
]

REPORT_FILE.write_text("\n".join(report) + "\n", encoding="utf-8")

print("\n" + "=" * 50)
print("MODULE 7 ANALYSIS COMPLETED")
print("Summary report saved to:", REPORT_FILE)
print("=" * 50)



# Module 7: Random sampling analysis

print("\n4. RANDOM SAMPLING ANALYSIS")

sample = sales.sample(
    n=min(1000, len(sales)),
    random_state=42
)

print("Full dataset rows:", len(sales))
print("Sample rows:", len(sample))
print("Full dataset mean:", round(sales.mean(), 2))
print("Sample mean:", round(sample.mean(), 2))
print("Difference:", round(sample.mean() - sales.mean(), 2))

sampling_report = REPORT_FOLDER / "module_07_sampling_summary.txt"

sampling_report.write_text(
    "NORTHSTAR RETAIL ANALYTICS PLATFORM\n"
    "MODULE 7 - RANDOM SAMPLING\n\n"
    f"Full dataset rows: {len(sales)}\n"
    f"Sample rows: {len(sample)}\n"
    f"Full dataset mean: {sales.mean():.2f}\n"
    f"Sample mean: {sample.mean():.2f}\n"
    f"Difference: {sample.mean() - sales.mean():.2f}\n\n"
    "A reproducible random sample was selected using random_state=42.\n"
    "The sample mean is compared with the full dataset mean.\n"
    "A sample may differ from the population; one sample does not prove representativeness.\n",
    encoding="utf-8"
)

print("Sampling report saved to:", sampling_report)
