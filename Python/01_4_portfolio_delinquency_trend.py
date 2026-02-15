import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------

script_dir  = os.path.dirname(os.path.abspath(__file__))
data_dir    = os.path.join(script_dir, "..", "Data_Generated")

delinquency_path = os.path.normpath(
    os.path.join(data_dir, "01_4d_portfolio_delinquency_trend.csv")
)

df_trend                    = pd.read_csv(delinquency_path)
df_trend["year_month"]      = pd.to_datetime(df_trend["year_month"])

# -----------------------------------------------------------
# Clean + index time
# -----------------------------------------------------------

# Clean columns first (so you can safely reference them)
df_trend.columns = (
    df_trend.columns
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

# Ensure year_month is datetime, then use it as the time index
df_trend["year_month"] = pd.to_datetime(df_trend["year_month"], errors="coerce")
df_trend = (
    df_trend
    .dropna(subset=["year_month"])
    .set_index("year_month")
    .sort_index()
    .asfreq("MS")  # month start frequency
)

# -----------------------------------------------------------
# Identify the DPD 30+ rate column
# (Your SQL likely produced a column literally named "round")
# -----------------------------------------------------------

# Prefer a clean name if you later alias it in SQL
if "dpd_30_plus_rate" in df_trend.columns:
    col_dpd_rate = "dpd_30_plus_rate"
elif "dpd_30_plus_rate_pct" in df_trend.columns:
    col_dpd_rate = "dpd_30_plus_rate_pct"
elif "round" in df_trend.columns:
    col_dpd_rate = "round"
else:
    raise KeyError(
        "Cannot find DPD 30+ rate column. Expected one of: "
        "'dpd_30_plus_rate', 'dpd_30_plus_rate_pct', or 'round'."
    )

# -----------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------

# Convert DPD 30+ to percent for readability
# If it's already percent (e.g., 12.3), keep it.
srs_rate = df_trend[col_dpd_rate]

if srs_rate.dropna().max() <= 1.5:
    df_trend["dpd_30_plus_rate_pct"] = srs_rate * 100.0
else:
    df_trend["dpd_30_plus_rate_pct"] = srs_rate

# 3-month moving averages (trend smoothing)
df_trend["dpd_30_plus_rate_pct_ma3"] = df_trend["dpd_30_plus_rate_pct"].rolling(3, min_periods=1).mean()

if "defaulted_loans" not in df_trend.columns:
    raise KeyError("Missing required column: 'defaulted_loans'.")

df_trend["defaulted_loans_ma3"] = df_trend["defaulted_loans"].rolling(3, min_periods=1).mean()

# Bucket shares (optional but useful)
bucket_cols = ["current_loans", "dpd_1_29_loans", "dpd_30_59_loans", "dpd_60_89_loans", "dpd_90_plus_loans"]
if "active_loans" in df_trend.columns and all(c in df_trend.columns for c in bucket_cols):
    for col in bucket_cols:
        df_trend[f"{col}_share_pct"] = np.where(
            df_trend["active_loans"] > 0,
            (df_trend[col] / df_trend["active_loans"]) * 100.0,
            np.nan
        )

# Lead/lag correlation: delinquency now vs defaults 0–6 months later
corr_rows = []
for lag in range(0, 7):
    srs_x = df_trend["dpd_30_plus_rate_pct"]
    srs_y = df_trend["defaulted_loans"].shift(-lag)  # defaults happen after delinquency
    df_tmp = pd.concat([srs_x, srs_y], axis=1).dropna()
    corr = df_tmp.corr().iloc[0, 1] if len(df_tmp) >= 3 else np.nan
    corr_rows.append({"defaults_lag_months": lag, "corr_dpd30plus_vs_future_defaults": corr})

df_corr = pd.DataFrame(corr_rows)

# -----------------------------------------------------------
# Visualizations
# -----------------------------------------------------------

# Chart 1: DPD 30+ rate (%) vs defaults (count)
fig, ax1 = plt.subplots(figsize=(10, 5))

line1, = ax1.plot(
    df_trend.index,
    df_trend["dpd_30_plus_rate_pct"],
    marker="o",
    linewidth=1.5,
    label="DPD 30+ Rate (%)"
)

line2, = ax1.plot(
    df_trend.index,
    df_trend["dpd_30_plus_rate_pct_ma3"],
    linewidth=2.5,
    label="DPD 30+ Rate (3M MA)"
)

ax1.set_ylabel("DPD 30+ Rate (%)")
ax1.set_xlabel("Month")

ax2 = ax1.twinx()

line3, = ax2.plot(
    df_trend.index,
    df_trend["defaulted_loans"],
    marker="o",
    linewidth=1.5,
    label="Defaulted Loans"
)

line4, = ax2.plot(
    df_trend.index,
    df_trend["defaulted_loans_ma3"],
    linewidth=2.5,
    label="Defaulted Loans (3M MA)"
)

ax2.set_ylabel("Defaulted Loans")

lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

plt.title("Delinquency (DPD 30+) vs Defaults Over Time")
plt.tight_layout()
plt.show()




# Chart 2: Bucket shares over time (if available)

# Build share_cols list (these are the columns we created above)
share_cols = [f"{col}_share_pct" for col in bucket_cols if f"{col}_share_pct" in df_trend.columns]

# Optional: for plotting, skip months where all shares are missing
df_chart2 = df_trend.copy()
df_chart2 = df_chart2.dropna(subset=share_cols, how="all")


plt.figure(figsize=(10, 5))

for col in share_cols:

    label = col.replace("_share_pct", "")

    if col == "current_loans_share_pct":
        plt.plot(
            df_chart2.index,
            df_chart2[col],
            linewidth=4,
            marker="o",        # ← add dots
            markersize=6,
            label="Current"
        )
    else:
        plt.plot(
            df_chart2.index,
            df_chart2[col],
            linewidth=1.5,
            marker="o",        # ← add dots
            markersize=4,
            alpha=0.8,
            label=label
        )

plt.title("DPD Bucket Shares Over Time")
plt.xlabel("Month")
plt.ylabel("Share of Active Loans (%)")
plt.legend()
plt.tight_layout()
plt.show()

