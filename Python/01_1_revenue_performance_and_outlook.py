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

# Project 06 structure:
# .py files live in /Python
# CSV files live in /Data_Generated
# find the directory for the CSV file
data_dir = os.path.join(script_dir, "..", "Data_Generated")

# Attach the file name into the path
revenue_path                = os.path.join(data_dir, "01_1_revenue_performance_and_outlook.csv")

# (Optional) normalize the path string
revenue_path                = os.path.normpath(revenue_path)

# Load CSV files to dataframe
df_revenue                  = pd.read_csv(revenue_path)

# Convert all dates to datetime objects (robust for mixed formats)
df_revenue["year_month"]    = pd.to_datetime(df_revenue["year_month"])

# make a series for timeseries modeling purpose. index is year_month, and the data is taken from gross_revenue
# tell python that the date is in year_month form using .asfreq("MS")
srs_gross_revenue = df_revenue.set_index("year_month")["gross_revenue"].asfreq("MS")

# ----------------------------------------------------------------------------------------------------------
# draw first graph (STL)

fig = STL(srs_gross_revenue, period=12).fit().plot()

# enlarge the figure canvas (fix "small chart" + excess whitespace)
fig.set_size_inches(16, 10)


# remove default STL title
fig.axes[0].set_title("")

# add business title
fig.suptitle(
    "CICA Prime — Gross Revenue STL Decomposition\n"
    "Trend • Seasonality • Residuals",
    fontsize=24,
    fontweight="bold",
    ha="center",
    x=0.5,
    y=0.94   # moved DOWN to center it in the header space
)

for ax in fig.axes:
    ax.set_ylabel(ax.get_ylabel(), fontsize=16)

# rotate x-axis labels (all subplots)
for ax in fig.axes:
    ax.tick_params(axis="x", rotation = 45, labelsize = 16)
    ax.tick_params(axis="y", labelsize = 16)

# compute January boundaries from the data index
start       = srs_gross_revenue.index.min()
end         = srs_gross_revenue.index.max()
jan_dates   = pd.date_range(start=start, end=end, freq="YS")  # Year Start = Jan 1

# draw dotted year lines on ALL subplots
for ax in fig.axes:
    for d in jan_dates:
        ax.axvline(d, linestyle=":", linewidth=0.8)

# bold January tick labels wherever ticks exist
for ax in fig.axes:
    for label, tick in zip(ax.get_xticklabels(), ax.get_xticks()):
        dt = mdates.num2date(tick)
        if dt.month == 1:
            label.set_fontweight("bold")

# spacing (tight layout tuned to keep big title + rotated ticks, while reducing white space)
fig.tight_layout(rect=[0.03, 0.10, 0.99, 0.90])

plt.show()


# ----------------------------------------------------------------------------------------------------------
# draw second graph, use SARIMA for time series data which has seasonality ( yearly )
# ----------------------------------------------------------------------------------------------------------

sarima_result   = SARIMAX( srs_gross_revenue, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12) ).fit(disp=False)
# srs_gross_revenue is the time series data you feed into SARIMAX.

# order=( 1, 1, 1) is telling the model how to handle short term behavior. 
# The first 1    - look at the last month.
# The second 1   - Remove the trend so the data is easier to learn
# The third 1    - Learn from recent mistakes

# .fit(disp=False)
# It looks at all the data, Tries many parameter combinations, Chooses the ones that explain history best.
# disp = False says “Don’t spam my screen with math messages.”


forecast_obj    = sarima_result.get_forecast(steps=12)
# predict the next 12 months. Check happens in month 1, based on what, check what happens on the next month...
# do all that 12 times. 
# the result is predicted revenue, uncertainty ranges, statistical details


df_forecast     = forecast_obj.summary_frame()
# this dataframe now has 
# mean              - Best guess of revenue)
# mean_se           - how unsure the model is about its prediction — 
#                     0 is more confident, a bigger number less confident.
#                     mean_se is the same unit as your data.
#                     There is no maximum; it depends on how noisy and unstable the history is
#                     If mean_se is small compared to the mean, the forecast is stable
#                     If mean_se is large compared to the mean, the forecast is risky and unreliable
# mean_ci_lower     - Conservative (bad case)
# mean_ci_upper     - Optimistic (good case)
# ci is confidence interval

print(df_forecast)

plt.figure(figsize=(16, 8))

plt.title(
    "CICA Prime — Monthly Gross Revenue (Interest + Fees)\n"
    "Actual vs 12-Month SARIMA Forecast",
    fontsize=24,          # doubled
    fontweight="bold",    # bold
    loc="center"          # force centering
)

plt.plot(srs_gross_revenue, label="Actual")
plt.plot(df_forecast["mean"], label="Forecast")

plt.fill_between(
    df_forecast.index,
    df_forecast["mean_ci_lower"],
    df_forecast["mean_ci_upper"],
    alpha=0.2,
    label="95% CI"
)

plt.legend()
plt.xticks(rotation=45)

# bold January ticks + draw year boundary lines (forecast chart)
ax = plt.gca()

ax.tick_params(axis="both", labelsize=16)
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)

for label in ax.get_xticklabels():
    tick_text = label.get_text()
    if tick_text == "":
        continue

    tick_dt = pd.to_datetime(tick_text, errors="coerce")

    if pd.notna(tick_dt) and tick_dt.month == 1:
        label.set_fontweight("bold")
        ax.axvline(
            tick_dt,
            linestyle=":",
            linewidth=0.8
        )

plt.subplots_adjust(left=0.06, right=0.99, bottom=0.18, top=0.88)
plt.show()
