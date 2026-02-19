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
script_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(script_dir, "..", "Data_Generated")

# Attach the file name into the path
pd_path                     = os.path.join(data_dir, "03_2_exposure_at_default.csv")
df_ead                      = pd.read_csv(pd_path)

# Ensure date columns are parsed correctly
df_ead["origination_date"]  = pd.to_datetime(df_ead["origination_date"])
df_ead["origination_month"] = pd.to_datetime(df_ead["origination_month"])
df_ead["default_date"]      = pd.to_datetime(df_ead["default_date"])



# Summarize EAD by vintage: 
# Group by origination_month then:
# compute defaulted_loan_count, 
# total principal_unpaid_on_default, 
# average principal_unpaid_on_default.

df_ead  = df_ead.sort_values(by='origination_month', ascending=True)

df_ead_by_vintage           =   ( df_ead
                                        .groupby("origination_month", as_index=False)
                                        .agg
                                        (
                                            defaulted_loan_count                = ("loan_id"                        , "nunique" ),
                                            total_principal_unpaid_on_default   = ("principal_unpaid_on_default"    , "sum"     ),
                                            avg_principal_unpaid_on_default     = ("principal_unpaid_on_default"    , "mean"    )
                                        )
                                )

# Sort result chronologically
df_ead_by_vintage           =   df_ead_by_vintage.sort_values("origination_month")


# Summarize EAD by risk tier: 
# Group by risk_tier_at_signup then :
# compute defaulted_loan_count, 
# total principal_unpaid_on_default, 
# average principal_unpaid_on_default.

df_ead_by_risk_tier         =   ( df_ead
                                        .groupby("risk_tier_at_signup", as_index=False)
                                        .agg
                                        (
                                            defaulted_loan_count                = ("loan_id"                     , "nunique"),
                                            total_principal_unpaid_on_default   = ("principal_unpaid_on_default" , "sum"),
                                            avg_principal_unpaid_on_default     = ("principal_unpaid_on_default" , "mean")
                                        )
                                )

df_ead_by_risk_tier         =   df_ead_by_risk_tier.sort_values("risk_tier_at_signup")

print(df_ead_by_vintage.head(100))

# -----------------------------
# Chart 1: Avg EAD by Risk Tier
# -----------------------------

df_plot_risk = df_ead_by_risk_tier.copy()

plt.figure(figsize=(18, 10))

bars = plt.bar(
    df_plot_risk["risk_tier_at_signup"].astype(str),
    df_plot_risk["avg_principal_unpaid_on_default"]
)

plt.title("Average EAD (Unpaid Principal) by Risk Tier at Signup", fontsize=24)
plt.xlabel("Risk Tier", fontsize=20)
plt.ylabel("Average Principal Unpaid at Default ($)", fontsize=20)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

# PD-style note box: top-left, tight to edge, framed
plt.text(
    0.02, 0.98,
    "Numbers above bars = Defaulted loan count (N)",
    transform=plt.gca().transAxes,
    ha="left",
    va="top",
    fontsize=18,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.35")
)

# Bar labels (N) above bars
for bar, n in zip(bars, df_plot_risk["defaulted_loan_count"]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + (0.01 * df_plot_risk["avg_principal_unpaid_on_default"].max()),
        f"{int(n)}",
        ha="center",
        va="bottom",
        fontsize=18
    )

plt.tight_layout()
plt.show()


# -----------------------------------
# Chart 2: Avg EAD by Origination Month
# -----------------------------------

df_plot_vintage = df_ead_by_vintage.copy()
df_plot_vintage["origination_month_label"] = df_plot_vintage["origination_month"].dt.strftime("%Y-%m")

plt.figure(figsize=(24, 10))

bars = plt.bar(
    df_plot_vintage["origination_month_label"],
    df_plot_vintage["avg_principal_unpaid_on_default"]
)

plt.title("Average EAD (Unpaid Principal) by Origination Month", fontsize=24)
plt.xlabel("Origination Month", fontsize=20)
plt.ylabel("Average Principal Unpaid at Default ($)", fontsize=20)

plt.xticks(rotation=45, ha="right", fontsize=16)
plt.yticks(fontsize=18)

plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))

# ---- Add vertical headroom so bars do not hit legend ----
max_y = df_plot_vintage["avg_principal_unpaid_on_default"].max()
plt.ylim(0, max_y * 1.20)   # 20% extra vertical space

# PD-style note box (top-right)
plt.text(
    0.98, 0.98,
    "Numbers above bars = Defaulted loan count (N)",
    transform=plt.gca().transAxes,
    ha="right",
    va="top",
    fontsize=18,
    bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.35")
)

# Bar labels
for bar, n in zip(bars, df_plot_vintage["defaulted_loan_count"]):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + (0.02 * max_y),   # small offset above bar
        f"{int(n)}",
        ha="center",
        va="bottom",
        fontsize=16
    )

plt.tight_layout()
plt.show()
