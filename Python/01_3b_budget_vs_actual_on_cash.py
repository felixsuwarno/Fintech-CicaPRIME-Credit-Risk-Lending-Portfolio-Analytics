import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------

script_dir              = os.path.dirname(os.path.abspath(__file__))
data_generated_dir      = os.path.join(script_dir, "..", "Data_Generated")
data_raw_dir            = os.path.join(script_dir, "..", "Data_RAW")

actual_cash_path        = os.path.normpath(os.path.join(data_generated_dir, "01_3b_actual_cash.csv"))
budget_plan_path        = os.path.normpath(os.path.join(data_raw_dir, "budget_plan_monthly.csv"))

df_actual               = pd.read_csv(actual_cash_path)
df_budget_raw           = pd.read_csv(budget_plan_path)

# -----------------------------------------------------------
# Prepare actual cash (monthly)
# -----------------------------------------------------------

df_actual["year_month"]     = pd.to_datetime(df_actual["year_month"])
df_actual                   = df_actual.sort_values("year_month")
df_actual                   = df_actual.set_index("year_month")
df_actual                   = df_actual.asfreq("MS")

df_actual["actual_cash"]    = df_actual["actual_cash"].fillna(0)

# -----------------------------------------------------------
# Prepare budget cash (monthly) - NO pivot_table
# -----------------------------------------------------------

df_budget_raw["year_month"] = pd.to_datetime(df_budget_raw["month"])

df_budget_base              = df_budget_raw[df_budget_raw["scenario_name"] == "base"].copy()
df_budget_stretch           = df_budget_raw[df_budget_raw["scenario_name"] == "stretch"].copy()

df_budget_base              = df_budget_base.groupby("year_month", as_index=False).agg(planned_cash_inflow=("planned_cash_inflow", "sum"))
df_budget_stretch           = df_budget_stretch.groupby("year_month", as_index=False).agg(planned_cash_inflow=("planned_cash_inflow", "sum"))

df_budget_base              = df_budget_base.rename(columns={"planned_cash_inflow": "budget_base_for_cash"})
df_budget_stretch           = df_budget_stretch.rename(columns={"planned_cash_inflow": "budget_stretch_for_cash"})

df_budget                   = df_budget_base.merge(df_budget_stretch, on="year_month", how="outer")
df_budget                   = df_budget.sort_values("year_month")
df_budget                   = df_budget.set_index("year_month")
df_budget                   = df_budget.asfreq("MS")

# -----------------------------------------------------------
# Merge
# -----------------------------------------------------------

df_bva                      = df_actual.merge(df_budget, left_index=True, right_index=True, how="left")

# -----------------------------------------------------------
# Variance (%)
# -----------------------------------------------------------

df_bva["var_cash_base"]         = df_bva["actual_cash"] - df_bva["budget_base_for_cash"]
df_bva["var_cash_stretch"]      = df_bva["actual_cash"] - df_bva["budget_stretch_for_cash"]

df_bva["var_cash_base_pct"]     = df_bva["var_cash_base"] / df_bva["budget_base_for_cash"].replace(0, np.nan)
df_bva["var_cash_stretch_pct"]  = df_bva["var_cash_stretch"] / df_bva["budget_stretch_for_cash"].replace(0, np.nan)

# -----------------------------------------------------------
# Plot settings
# -----------------------------------------------------------

plt.rcParams.update({
    "font.size":        14,
    "axes.titlesize":   18,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12
})

# -----------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------

def _plot_by_year_segments(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    label: str,
    marker: str = "o",
    markersize: int = 7,
    color: str | None = None
) -> None:

    years   = df.index.to_series().dt.year.unique()
    first   = True

    for year in years:

        mask    = df.index.to_series().dt.year == year
        x       = df.index[mask]
        y       = df.loc[x, col]

        ax.plot(
            x,
            y,
            label       = label if first else None,
            marker      = marker,
            markersize  = markersize,
            color       = color
        )

        first = False


def _add_vertical_guides_every_4_months(ax: plt.Axes, df: pd.DataFrame) -> None:

    for dt in df.index[::4]:
        ax.axvline(dt, linestyle=":", linewidth=0.8, alpha=0.6)


# -----------------------------------------------------------
# Chart — Cash
# -----------------------------------------------------------

COLOR_ACTUAL         = "C0"
COLOR_BASE           = "C1"
COLOR_STRETCH        = "C2"
COLOR_VAR_BASE       = "C0"
COLOR_VAR_STRETCH    = "C1"

fig, (ax_top, ax_bottom) = plt.subplots(
    nrows       = 2,
    ncols       = 1,
    figsize     = (14, 9),
    sharex      = True,
    gridspec_kw = {"height_ratios": [2.2, 1.2]}
)

# -------------------------
# Top: Actual vs Budget ($)
# -------------------------

_plot_by_year_segments(ax_top, df_bva, "actual_cash",           "Actual",            color=COLOR_ACTUAL)
_plot_by_year_segments(ax_top, df_bva, "budget_base_for_cash",  "Budget (Base)",     color=COLOR_BASE)
_plot_by_year_segments(ax_top, df_bva, "budget_stretch_for_cash","Budget (Stretch)", color=COLOR_STRETCH)

ax_top.set_title("CICA Prime — Actual vs Budget (Cash)")
ax_top.set_ylabel("Dollars")
ax_top.legend()

_add_vertical_guides_every_4_months(ax_top, df_bva)

# -------------------------
# Bottom: Variance (%)
# -------------------------

df_pct                                   = df_bva.copy()
df_pct["var_cash_base_pct"]              = df_pct["var_cash_base_pct"] * 100
df_pct["var_cash_stretch_pct"]           = df_pct["var_cash_stretch_pct"] * 100

ax_bottom.axhline(0, linewidth=1)

_plot_by_year_segments(ax_bottom, df_pct, "var_cash_base_pct",     "Variance % vs Base",    color=COLOR_VAR_BASE)
_plot_by_year_segments(ax_bottom, df_pct, "var_cash_stretch_pct",  "Variance % vs Stretch", color=COLOR_VAR_STRETCH)

ax_bottom.set_title("CICA Prime — Monthly Variance % (Cash)")
ax_bottom.set_ylabel("Percent")
ax_bottom.set_xlabel("Month")
ax_bottom.legend()

_add_vertical_guides_every_4_months(ax_bottom, df_pct)

plt.tight_layout()
plt.show()
