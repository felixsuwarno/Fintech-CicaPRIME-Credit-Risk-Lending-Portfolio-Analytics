import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Pandas display settings
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 2000)

# find the path to current active Python path, ground it here
script_dir                          = os.path.dirname(os.path.abspath(__file__))

# Project structure
# .py files live in /Python
# CSV files live in /Data_Generated
data_dir                            = os.path.join(script_dir, "..", "Data_Generated")

# Attach the file name into the path
cashflowgap_path                    = os.path.join(data_dir, "01_2_scheduled_vs_actual_cash_flow.csv")

# Normalize path
cashflowgap_path                    = os.path.normpath(cashflowgap_path)

# Load CSV
df_cashflowgap                      = pd.read_csv(cashflowgap_path)

# Convert date
df_cashflowgap["year_month"]        = pd.to_datetime(df_cashflowgap["year_month"])



# -----------------------------------------------------------
# Business Answer Starts here
# -----------------------------------------------------------


# Compute percentage gap
df_cashflowgap["cashflow_gap_pct"]  = np.where  (
                                                    df_cashflowgap["scheduled_cash_flow"] > 0,
                                                    (df_cashflowgap["cashflow_gap"] / df_cashflowgap["scheduled_cash_flow"]) * 100,
                                                    np.nan
                                                )

df_cashflowgap["cashflow_gap_pct"]  = df_cashflowgap["cashflow_gap_pct"].round(2)

print(df_cashflowgap)

print()

avg_gap_pct                 =  df_cashflowgap["cashflow_gap_pct"].mean()
std_gap_pct                 =  df_cashflowgap["cashflow_gap_pct"].std()
total_months                =  df_cashflowgap["cashflow_gap_pct"].notna().sum()
under_collection_months     = (df_cashflowgap["cashflow_gap_pct"] < 0).sum()
freq_under_collection       =  under_collection_months / total_months


# -----------------------------------------------------------
# Visualization — Monthly Cashflow Gap (%) + Summary Table
# -----------------------------------------------------------

fig, (ax_chart, ax_table) = plt.subplots(
                                            2,
                                            1,
                                            figsize=(14, 9),
                                            gridspec_kw={"height_ratios": [4, 1]}
                                        )

# -----------------------------
# Top chart
# -----------------------------

ax_chart.axhline(0, linewidth=1)

ax_chart.plot   (
                    df_cashflowgap["year_month"],
                    df_cashflowgap["cashflow_gap_pct"],
                    marker="o"
                )

# annotate each point
for x, y in zip(df_cashflowgap["year_month"], df_cashflowgap["cashflow_gap_pct"]):
    if pd.isna(y):
        continue

    offset = 0.9 if y >= 0 else -1.1

    ax_chart.text   (
                        x,
                        y + offset,
                        f"{y:.1f}%",
                        ha = "center",
                        va = "bottom" if y >= 0 else "top",
                        fontsize=12,
                        fontweight="bold"
                    )

ax_chart.set_title  (
                        "CICA Prime — Monthly Cashflow Gap (%)\nActual vs Scheduled Payments",
                        fontsize=20,
                        fontweight="bold"
                    )

ax_chart.set_ylabel ("Cashflow Gap (%)" , fontsize  = 14)
ax_chart.set_xlabel ("Month"            , fontsize  = 14)
ax_chart.tick_params(axis="x"           , rotation  = 45   , labelsize=14)
ax_chart.tick_params(axis="y"           , labelsize = 14)

# -----------------------------
# Bottom table (summary stats)
# -----------------------------

ax_table.axis("off")

stats_rows = [
    ["Average cashflow gap",            f"{avg_gap_pct:.2f}%"],
    ["Standard deviation",              f"{std_gap_pct:.2f}%"],
    ["Under-collection months",         f"{under_collection_months} out of {total_months}"],
    ["Frequency of under-collection",   f"{freq_under_collection:.2%}"],
]

table = ax_table.table(
    cellText    = stats_rows,
    colWidths   = [0.35, 0.15],   # ⬅️ roughly half-width table
    cellLoc     = "center",
    loc         = "center",
    bbox        = [((1-0.25)/2), 0.50, 0.25, 0.70]  # ⬅️ centered + moved UP
)

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.0, 1.6)

for cell in table.get_celld().values():
    cell.set_edgecolor("#B0B0B0")   # light gray borders
    cell.set_linewidth(0.8)
    cell.set_facecolor("#F2F2F2")   # soft gray background
    cell.PAD = 0.18

# -----------------------------
# Final layout
# -----------------------------

plt.tight_layout()
plt.show()
