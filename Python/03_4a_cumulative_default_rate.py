import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Pandas display settings
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 2000)

# find the path to current active Python path, ground it here
script_dir                  = os.path.dirname(os.path.abspath(__file__))

data_dir                    = os.path.join(script_dir, "..", "Data_Generated")

# Attach the file name into the path
cdr_filename                = "03_4a_cumulative_default_rate.csv"
cdr_path_local              = os.path.join(data_dir, cdr_filename)
cdr_path_uploaded           = "/mnt/data/03_4a_cumulative_default_rate.csv"

# Load dataset

df_cdr12m                   = pd.read_csv(cdr_path_local)


# Ensure date columns are parsed correctly
df_cdr12m["origination_month"] = pd.to_datetime(df_cdr12m["origination_month"])


# Sort for correct time order (plotting only)
df_cdr12m                   = df_cdr12m.sort_values("origination_month")

# Basic sanity checks (no cleaning)
print(df_cdr12m.columns)
print(df_cdr12m.dtypes)
print(df_cdr12m.head())

# ----------------------------
# CHART
# ----------------------------
# Sort for correct time order
df_cdr12m                   = df_cdr12m.sort_values("origination_month")

srs_x                       = df_cdr12m["origination_month"]
srs_n_loans                 = df_cdr12m["n_loans_in_vintage"]
srs_n_defaults              = df_cdr12m["n_default_12m_loans"]
srs_cdr_12m                 = df_cdr12m["cdr_12m"]

fig                         = plt.figure(figsize=(14,6))
ax1                         = plt.gca()

bar_loans                   = ax1.bar(
    srs_x,
    srs_n_loans,
    width=20,
    label="Loans in Vintage (N)"
)

bar_defaults                = ax1.bar(
    srs_x,
    srs_n_defaults,
    width=10,
    label="Defaults within 12M (N)"
)

# Enlarge axis labels (2x)
ax1.set_xlabel("Origination Month", fontsize=20)
ax1.set_ylabel("Loan Counts", fontsize=20)

ax2                         = ax1.twinx()

ax2.plot(
    srs_x,
    srs_cdr_12m,
    label="12M Cumulative Default Rate (CDR)",
    color="black",
    linewidth=3,
    marker="o",
    markersize=9,
    markeredgewidth=2.5
)

ax2.set_ylabel("12M Cumulative Default Rate", fontsize=20)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

# Enlarge title (2x)
plt.title(
    "12M Cumulative Default Rate and Loan Counts by Vintage",
    fontsize=24,
    fontweight="bold"
)

# Larger bar value labels
for rect in bar_loans:
    x_pos                   = rect.get_x() + rect.get_width() / 2
    y_val                   = rect.get_height()
    if pd.notna(y_val) and y_val != 0:
        ax1.text(
            x_pos,
            y_val,
            f"{int(y_val)}",
            ha="center",
            va="bottom",
            color=rect.get_facecolor(),
            fontsize=12,
            fontweight="bold"
        )

for rect in bar_defaults:
    x_pos                   = rect.get_x() + rect.get_width() / 2
    y_val                   = rect.get_height()
    if pd.notna(y_val) and y_val != 0:
        ax1.text(
            x_pos,
            y_val,
            f"{int(y_val)}",
            ha="center",
            va="bottom",
            color=rect.get_facecolor(),
            fontsize=12,
            fontweight="bold"
        )

# Larger line value labels
y_max                       = float(np.nanmax(srs_cdr_12m)) if len(srs_cdr_12m) else 0.0
y_offset                    = max(0.15, y_max * 0.03)

for x_val, y_val in zip(srs_x, srs_cdr_12m):
    if pd.notna(y_val):
        ax2.text(
            x_val,
            y_val + y_offset,
            f"{y_val:.2f}%",
            ha="center",
            va="bottom",
            color="black",
            fontsize=12,
            fontweight="bold"
        )

# Force every bar to have its own 45° label
ax1.set_xticks(srs_x)
ax1.set_xticklabels(
    [d.strftime("%Y-%m") for d in srs_x],
    rotation=45,
    ha="right",
    fontsize=14
)

handles1, labels1           = ax1.get_legend_handles_labels()
handles2, labels2           = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=12)

plt.tight_layout()
plt.show()

