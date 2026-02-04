import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Load data
# -----------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "Data_Generated")

budget_vs_actual_path = os.path.normpath(
    os.path.join(data_dir, "01_3_budget_vs_actual_performance.csv")
)

df_bva = pd.read_csv(budget_vs_actual_path)
df_bva["year_month"] = pd.to_datetime(df_bva["year_month"])
df_bva = df_bva.set_index("year_month").sort_index().asfreq("MS")

# -----------------------------------------------------------
# Variance columns (Actual - Budget)
# -----------------------------------------------------------

df_bva["var_revenue_base"]     = df_bva["actual_revenue"] - df_bva["budget_base_for_revenue"]
df_bva["var_revenue_stretch"]  = df_bva["actual_revenue"] - df_bva["budget_stretch_for_revenue"]

df_bva["var_cash_base"]        = df_bva["actual_cash"] - df_bva["budget_base_for_cash"]
df_bva["var_cash_stretch"]     = df_bva["actual_cash"] - df_bva["budget_stretch_for_cash"]

# Loss variances: positive means losses worse than plan
df_bva["var_loss_base"]        = df_bva["actual_loss"] - df_bva["budget_base_for_loss"]
df_bva["var_loss_stretch"]     = df_bva["actual_loss"] - df_bva["budget_stretch_for_loss"]

# -----------------------------------------------------------
# Percent variances (per month): (Actual - Budget) / Budget
# -----------------------------------------------------------

df_bva["var_revenue_base_pct"]     = df_bva["var_revenue_base"] / df_bva["budget_base_for_revenue"].replace(0, np.nan)
df_bva["var_revenue_stretch_pct"]  = df_bva["var_revenue_stretch"] / df_bva["budget_stretch_for_revenue"].replace(0, np.nan)

df_bva["var_cash_base_pct"]        = df_bva["var_cash_base"] / df_bva["budget_base_for_cash"].replace(0, np.nan)
df_bva["var_cash_stretch_pct"]     = df_bva["var_cash_stretch"] / df_bva["budget_stretch_for_cash"].replace(0, np.nan)

df_bva["var_loss_base_pct"]        = df_bva["var_loss_base"] / df_bva["budget_base_for_loss"].replace(0, np.nan)
df_bva["var_loss_stretch_pct"]     = df_bva["var_loss_stretch"] / df_bva["budget_stretch_for_loss"].replace(0, np.nan)

# -----------------------------------------------------------
# Plot settings
# -----------------------------------------------------------

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
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
    markersize: int = 7
) -> None:
    """
    Plot one series as separate line segments per calendar year.
    This breaks the line between Dec->Jan while keeping January points visible.
    """
    years = df.index.to_series().dt.year.unique()
    first = True

    for y in years:
        mask = df.index.to_series().dt.year == y
        x = df.index[mask]
        yvals = df.loc[x, col]

        ax.plot(
            x,
            yvals,
            label=label if first else None,
            marker=marker,
            markersize=markersize
        )
        first = False


def _add_vertical_guides_every_4_months(ax: plt.Axes, df: pd.DataFrame) -> None:
    """
    Add vertical dotted lines aligned to every 4 months starting from the first month.
    """
    for dt in df.index[::4]:
        ax.axvline(dt, linestyle=":", linewidth=0.8, alpha=0.6)


def plot_actual_vs_budget_with_variance_pct(
    df: pd.DataFrame,
    actual_col: str,
    base_budget_col: str,
    stretch_budget_col: str,
    var_base_pct_col: str,
    var_stretch_pct_col: str,
    title_top: str,
    title_bottom: str,
    show_stretch_variance: bool = True
) -> None:
    fig, (ax_top, ax_bottom) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]}
    )

    # ---- Top: levels ----
    _plot_by_year_segments(ax_top, df, actual_col, "Actual")
    _plot_by_year_segments(ax_top, df, base_budget_col, "Budget (Base)")
    _plot_by_year_segments(ax_top, df, stretch_budget_col, "Budget (Stretch)")

    ax_top.set_title(title_top)
    ax_top.set_ylabel("Dollars")
    ax_top.legend()

    _add_vertical_guides_every_4_months(ax_top, df)

    # ---- Bottom: variance % ----
    ax_bottom.axhline(0, linewidth=1)
    _plot_by_year_segments(ax_bottom, df, var_base_pct_col, "Variance % vs Base")

    if show_stretch_variance:
        _plot_by_year_segments(ax_bottom, df, var_stretch_pct_col, "Variance % vs Stretch")

    # Convert y-axis from fraction to percent by multiplying plotted data (do it by scaling the data)
    # We'll re-plot with scaled values so labels and markers stay consistent.
    ax_bottom.cla()
    ax_bottom.axhline(0, linewidth=1)

    # Plot scaled variance % (x100)
    years = df.index.to_series().dt.year.unique()
    first_base = True
    first_stretch = True

    for y in years:
        mask = df.index.to_series().dt.year == y
        x = df.index[mask]

        y_base = (df.loc[x, var_base_pct_col] * 100)
        ax_bottom.plot(
            x, y_base,
            label="Variance % vs Base" if first_base else None,
            marker="o",
            markersize=7
        )
        first_base = False

        if show_stretch_variance:
            y_stretch = (df.loc[x, var_stretch_pct_col] * 100)
            ax_bottom.plot(
                x, y_stretch,
                label="Variance % vs Stretch" if first_stretch else None,
                marker="o",
                markersize=7
            )
            first_stretch = False

    ax_bottom.set_title(title_bottom)
    ax_bottom.set_ylabel("Percent")
    ax_bottom.set_xlabel("Month")
    ax_bottom.legend()

    _add_vertical_guides_every_4_months(ax_bottom, df)

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# Charts
# -----------------------------------------------------------

plot_actual_vs_budget_with_variance_pct(
    df=df_bva,
    actual_col="actual_revenue",
    base_budget_col="budget_base_for_revenue",
    stretch_budget_col="budget_stretch_for_revenue",
    var_base_pct_col="var_revenue_base_pct",
    var_stretch_pct_col="var_revenue_stretch_pct",
    title_top="CICA Prime — Actual vs Budget (Revenue)",
    title_bottom="CICA Prime — Monthly Variance % (Revenue)",
    show_stretch_variance=True
)

plot_actual_vs_budget_with_variance_pct(
    df=df_bva,
    actual_col="actual_cash",
    base_budget_col="budget_base_for_cash",
    stretch_budget_col="budget_stretch_for_cash",
    var_base_pct_col="var_cash_base_pct",
    var_stretch_pct_col="var_cash_stretch_pct",
    title_top="CICA Prime — Actual vs Budget (Cash)",
    title_bottom="CICA Prime — Monthly Variance % (Cash)",
    show_stretch_variance=True
)

plot_actual_vs_budget_with_variance_pct(
    df=df_bva,
    actual_col="actual_loss",
    base_budget_col="budget_base_for_loss",
    stretch_budget_col="budget_stretch_for_loss",
    var_base_pct_col="var_loss_base_pct",
    var_stretch_pct_col="var_loss_stretch_pct",
    title_top="CICA Prime — Actual vs Budget (Net Credit Loss)",
    title_bottom="CICA Prime — Monthly Variance % (Net Credit Loss)",
    show_stretch_variance=True
)
