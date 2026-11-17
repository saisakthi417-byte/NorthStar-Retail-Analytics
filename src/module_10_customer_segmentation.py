
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 1. Set project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "outputs" / "integrated" / "integrated_transactions.csv"
OUTPUT_DIR = ROOT / "outputs" / "module_10"
REPORT_DIR = ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Find the required columns
def find_column(data, possible_names):
    for name in possible_names:
        for column in data.columns:
            cleaned_column = (
                str(column).lower().replace("_", "").replace(" ", "")
            )
            cleaned_name = name.lower().replace("_", "").replace(" ", "")

            if cleaned_column == cleaned_name:
                return column

    return None


# 3. Load the integrated dataset
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Please run the previous modules first."
    )

data = pd.read_csv(INPUT_FILE)

customer_col = find_column(
    data, ["CustomerID", "Customer_ID", "Customer Id"]
)
date_col = find_column(
    data, ["OrderDate", "TransactionDate", "InvoiceDate", "Date"]
)
sales_col = find_column(
    data, ["LineAmount", "Sales", "TotalAmount", "Revenue", "Amount"]
)
order_col = find_column(
    data, ["OrderID", "TransactionID", "InvoiceNo", "InvoiceID"]
)

if not customer_col or not date_col or not sales_col:
    raise ValueError(
        "Required columns not found. Available columns:\n"
        + ", ".join(data.columns)
    )

data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
data[sales_col] = pd.to_numeric(data[sales_col], errors="coerce")

data = data.dropna(subset=[customer_col, date_col, sales_col])
data = data[data[sales_col] > 0].copy()

if data.empty:
    raise ValueError("No valid customer transaction records were found.")


# 4. Calculate RFM values
# Recency: days since the customer's last purchase
# Frequency: number of orders, where an order ID is available
# Monetary: total customer spending

analysis_date = data[date_col].max().normalize() + pd.Timedelta(days=1)

if order_col:
    frequency = data.groupby(customer_col)[order_col].nunique()
    frequency_name = "Unique orders"
else:
    frequency = data.groupby(customer_col).size()
    frequency_name = "Transaction rows (order ID unavailable)"

rfm = data.groupby(customer_col).agg(
    LastPurchase=(date_col, "max"),
    Monetary=(sales_col, "sum")
)

rfm["Frequency"] = frequency
rfm["Recency"] = (
    analysis_date - rfm["LastPurchase"]
).dt.days

rfm = rfm.reset_index()
rfm = rfm.rename(columns={customer_col: "CustomerID"})

rfm = rfm[
    (rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)
].copy()

if len(rfm) < 4:
    raise ValueError(
        "At least four customers are needed for four-cluster analysis."
    )


# 5. Calculate RFM scores from 1 to 5
# A lower recency value receives a higher score.

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["RFM_Score"] = (
    rfm["R_Score"].astype(str)
    + rfm["F_Score"].astype(str)
    + rfm["M_Score"].astype(str)
)


# 6. Prepare data for K-Means clustering
# Log transformation reduces the effect of very large values.

features = rfm[["Recency", "Frequency", "Monetary"]].copy()
features = features.apply(lambda column: __import__("numpy").log1p(column))

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(scaled_features)


# 7. Understand each cluster and name the segments
cluster_profile = rfm.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
).round(2)

recency_mid = rfm["Recency"].median()
frequency_mid = rfm["Frequency"].median()
monetary_mid = rfm["Monetary"].median()


def name_segment(row):
    recent = row["Avg_Recency"] <= recency_mid
    frequent = row["Avg_Frequency"] >= frequency_mid
    valuable = row["Avg_Monetary"] >= monetary_mid

    if recent and frequent and valuable:
        return "Champions"
    elif recent and frequent:
        return "Loyal Customers"
    elif recent and valuable:
        return "Recent High Spenders"
    elif not recent and frequent:
        return "Needs Attention"
    elif not recent and not frequent and not valuable:
        return "At Risk / Low Engagement"
    else:
        return "Potential Customers"


cluster_profile["Segment"] = cluster_profile.apply(
    name_segment, axis=1
)

segment_names = cluster_profile["Segment"].to_dict()
rfm["Segment"] = rfm["Cluster"].map(segment_names)


# 8. Add business recommendations
recommendations = {
    "Champions": "Reward loyalty and offer exclusive benefits.",
    "Loyal Customers": "Use loyalty points and repeat-purchase offers.",
    "Recent High Spenders": "Encourage repeat purchases with personalised offers.",
    "Needs Attention": "Send re-engagement campaigns and relevant offers.",
    "At Risk / Low Engagement": "Run win-back campaigns and investigate inactivity.",
    "Potential Customers": "Use targeted promotions to encourage more purchases."
}

rfm["Recommendation"] = rfm["Segment"].map(recommendations)

cluster_profile["Recommendation"] = (
    cluster_profile["Segment"].map(recommendations)
)


# 9. Save results
rfm.to_csv(OUTPUT_DIR / "customer_rfm_analysis.csv", index=False)
rfm.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

cluster_profile.to_csv(
    OUTPUT_DIR / "cluster_profiles.csv"
)

segment_summary = rfm.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Average_Recency=("Recency", "mean"),
    Average_Frequency=("Frequency", "mean"),
    Average_Monetary=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum")
).round(2)

segment_summary.to_csv(
    OUTPUT_DIR / "segment_summary.csv"
)


# 10. Create a short business report
report_file = REPORT_DIR / "module_10_customer_segmentation.txt"

with open(report_file, "w", encoding="utf-8") as report:
    report.write("NORTHSTAR RETAIL ANALYTICS PLATFORM\n")
    report.write("MODULE 10: CUSTOMER SEGMENTATION\n\n")

    report.write(f"Valid transaction rows: {len(data)}\n")
    report.write(f"Customers analysed: {len(rfm)}\n")
    report.write(f"Frequency calculation: {frequency_name}\n")
    report.write("Clustering method: K-Means (4 clusters)\n\n")

    report.write("CUSTOMER SEGMENT SUMMARY\n")
    report.write(segment_summary.to_string())
    report.write("\n\nCLUSTER PROFILES\n")
    report.write(cluster_profile.to_string())
    report.write("\n\nBUSINESS RECOMMENDATIONS\n")

    for segment, recommendation in recommendations.items():
        report.write(f"\n{segment}: {recommendation}\n")


# 11. Display completion details
print("\nMODULE 10 COMPLETED")
print("-------------------")
print(f"Valid transactions: {len(data):,}")
print(f"Customers analysed: {len(rfm):,}")
print(f"Frequency calculation: {frequency_name}")

print("\nCustomer segment summary:")
print(segment_summary)

print("\nFiles saved in:")
print(OUTPUT_DIR)
print(REPORT_DIR)
