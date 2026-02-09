# WIP ( Work in Progress ) Feb 8th 2026

# Consumer Lending “Cica PRIME” Credit Risk and Portfolio Analytics  
**Risk Management • Portfolio Analytics • Forecasting • Stress Testing**

This project analyzes synthetic but realistic consumer installment lending data from CICA (Consumer Installment Credit Account) Prime, a simulated consumer credit product designed to resemble real-world installment lending / BNPL-style portfolios offered by fintechs and traditional banks. The product extends unsecured credit to consumers and generates revenue through interest and fees, while incurring credit losses through delinquency and default.

The dataset spans January 2023 to December 2025 and includes loan-level, customer-level, and payment-level data, along with contractual payment schedules, macroeconomic scenarios, and internal budget plans. The data captures the full credit lifecycle: customer acquisition, underwriting decisions, loan origination, repayment behavior, delinquency, default, recoveries, and portfolio-level financial outcomes.

The analysis focuses on risk-aware, finance-aligned analytics, modeling how portfolio growth, repayment behavior, and credit losses are expected to evolve over time under different conditions. The goal is to evaluate portfolio health, revenue sustainability, and downside risk using predictive modeling, time-series forecasting, and scenario-based stress testing, consistent with how consumer lending analytics teams operate in practice.

<br><br>

➤ **Project Goal / Purpose:**  

The goal of this project is to understand how CICA Prime’s consumer lending portfolio grows over time, how stable that growth is, and how exposed the business is to downside risk. The analysis uses forecasting, predictive modeling, and standard credit risk techniques commonly applied in consumer lending.

The project looks at how revenue, cash inflows, delinquency, and defaults are expected to change over time, and compares actual results to internal plans to evaluate how accurate and resilient the business’s financial planning is. At the borrower and loan level, the analysis estimates customer lifetime value, the likelihood that borrowers stop using the product, and the likelihood that loans default, with the goal of using past behavior to make informed predictions about future outcomes.

At the portfolio level, the project measures expected credit losses, evaluates how concentrated revenue and risk are across borrowers, and tests how the portfolio would perform under different economic conditions. The goal is to determine whether portfolio growth is supported by sustainable cash flows and manageable credit risk, or whether it relies on elevated risk, loose underwriting, or a small group of borrowers—especially when economic conditions worsen.

<br><br>

➤ **Skills Demonstrated:**  

Consumer lending analytics and KPI design, credit portfolio performance analysis, delinquency and default analysis (DPD, roll rates, vintage curves), cohort-based customer lifetime value (LTV) modeling, probability of default (PD) modeling, loss analysis using EAD and LGD concepts, revenue and cash-flow analysis, budget vs actual variance analysis, and macro-driven stress testing and scenario analysis

**(SQL • Python • Pandas • Time-Series Forecasting • Credit Risk Metrics • Executive-Ready Analysis)**

<br><br>

➤ **Core Business Questions:**

**1 — Forecasting & Financial Planning**

1. **Revenue performance & outlook** <br> How did monthly interest and fee revenue perform from 2023–2025, and what is the expected revenue trajectory over the next 12 months?
2. **Cash inflows vs contractual expectations** <br> How do actual monthly cash collections compare to scheduled cash collections from 2023–2025, and what is the resulting monthly collection gap? How large is the monthly collection gap relative to scheduled cash, and how stable is this gap over time?
3. **Delinquency & default trends** <br> How have delinquency and default rates evolved, and what do these trends indicate about portfolio credit health?
4. **Budget vs actual performance** <br> How reliable is financial planning given deviations between budgeted and realized revenue, cash inflows, and losses?

**2 — Borrower Activation, Churn & Value**

1. **Customer activation timing** <br> How long does it take newly acquired customers to originate their first loan?
2. **Borrower inactivity & churn risk** <br> Which customers are most likely to stop borrowing after their initial loan?
3. **Customer lifetime value (LTV)** <br> Which customers are expected to generate the highest lifetime value after accounting for credit losses?
4. **Value concentration** <br> How concentrated is customer value, and how dependent is portfolio performance on top-value segments?

**3 — Credit Risk Modeling & Portfolio Loss Dynamics**

1. **Probability of default (PD)** <br> Which individual loans are most likely to default based on borrower, loan, and early behavior signals?
2. **Exposure at default (EAD)** <br> How much exposure remains outstanding at the time loans default?
3. **Loss given default (LGD)** <br> How severe are losses after recoveries, and how do they vary across segments?
4. **Vintage risk performance** <br> How do cumulative defaults and losses evolve by origination vintage?
5. **Credit policy & decision thresholds** <br> Where should approval, review, and rejection thresholds be set based on predicted default risk?
6. **Model monitoring & governance** <br> How stable and well-calibrated are credit risk models over time?

**4 — Portfolio Fragility & Stress Testing**

1. **Macro stress & portfolio survival** <br> Under adverse macroeconomic scenarios, can portfolio revenue absorb stressed credit losses without threatening business viability?

<br><br>

**➤ Executive Summary:**
WORK IN PROGRESS !!!
FOR LATER.

<br><br>

**➤ The Dataset**

The raw dataset spans January 2023 through December 2025, and all reporting and conclusions in this project are intentionally scoped to this full three-year period to evaluate portfolio growth, seasonality, underwriting changes, and macro-driven risk cycles, while supporting time-series forecasting, predictive modeling, portfolio risk analytics, and stress testing in a consistent analytical framework.

The analysis uses seven core tables representing a consumer installment lending product (“CICA Prime”):

**customers**
- One row per customer
- Contains signup_date, acquisition channel, risk_tier_at_signup, and demographic buckets (income_band, region, age_band)
- Used to define customer cohorts, segment performance, and build customer-level features for LTV / churn / PD modeling

**applications**
- One row per credit application
- Contains application date, approval/decline decision, approved amount, and a synthetic decision_score with reason codes for declines
- Used to analyze conversion funnels, approval rates, underwriting shifts over time, and credit policy tightening/loosening

**loans**
- One row per booked loan
- Contains origination details (principal, term_months, apr, origination fees), merchant category, and lifecycle outcomes (loan_status, default_date)
- Used as the central table for portfolio analytics, risk segmentation, PD/LGD/EAD construction, and cohort/vintage analysis

**payment_schedule**
- One row per contractual installment
- Contains due_date and the contractual split between due_principal and due_fee_interest, plus scheduled remaining balance
- Used to measure scheduled cash flows, delinquency timing, and to establish the “what should have happened” baseline for forecasting and variance analysis

**payments**
- One row per cash event
- Contains payment_date, payment_amount, split between paid_principal and paid_fee_interest, plus payment_type (scheduled, partial, refund, recovery)
- Used to measure actual cash collections, revenue realized through fees/interest, delinquency behavior, recoveries after default, and net loss outcomes

**macro_monthly**
- One row per month per scenario
- Contains monthly macro indexes (unemployment_index, rates_index, consumer_stress_index) for baseline, adverse, and severe scenarios
- Used for stress testing, scenario comparison, and macro-driven sensitivity analysis of delinquency/default behavior and portfolio losses

**budget_plan_monthly**
- One row per month per plan scenario
- Contains planned originations, cash inflow, revenue, and net losses under base vs stretch plans
- Used for budget vs actual variance modeling, planning accuracy evaluation, and “what management expected vs what happened” analysis

**dim_month**
- Calendar spine to help with time series modeling.
- This is an additional "tool" to help with the calculation. Not data in itself.

<br><br>

## The Main Report - Key Questions Answered

### 1 — Forecasting & Financial Planning

<br>

**1.1. Revenue Performance & Outlook**

How did total portfolio interest and fee revenue perform on a monthly basis from 2023 through 2025, and what is the expected monthly revenue performance over the next 12 months?

**Tables used**
- payments
- dim_month ( the calendar spine )

**SQL Method**
- Filter gross revenue ( **paid_fee_interest** ) for all **payment_type** labeled **scheduled** or **partial**.
- Aggregate monthly realized interest and fee revenue using **payment_date**, converted to a **year_month** configuration.
- Left join to a calendar spine to ensure zero-revenue months are included.
- SQL Output : a simple table with two columns, **year_month** and **gross_revenue**.

**Python Method**
- Load the monthly revenue output from SQL and index it by `year_month` as a monthly time series (`.asfreq("MS")`).
- Run STL decomposition (`period=12`) and plot the trend, seasonal, and residual components to diagnose revenue structure.
- Fit the historical monthly revenue series into a seasonal SARIMA model to capture short-term dynamics and yearly seasonality.
- Generate a 12-month forecast and extract the forecast table (`mean`, `mean_ci_lower`, `mean_ci_upper`).
- Plot actual vs forecast and shade the confidence interval to visually communicate expected trajectory and uncertainty.

<br>

**Charts**

<p align="center">
  <img src="Charts/01_1_revenue_performance_and_outlook_a_STL.png" width="100%">
</p>

<p align="center">
  <img src="Charts/01_1_revenue_performance_and_outlook_b_SARIMAX.png" width="100%">
</p>

**Key Insights**

TO BE WRITTEN

<br><br>

**1.2. Cash Inflows vs Contractual Expectations**

How do actual cash collections compare to scheduled payments on a monthly basis from 2023 through 2025?
How stable / reliable are monthly collection gaps across this period?

**Tables used**
- **payment_schedule** , contractual principal and interest due
- **payments** , actual cash collected
- **dim_month** ( the calendar spine )

**SQL Method**
- Aggregate monthly scheduled cash flows from **payment_schedule**.
- Aggregate monthly actual cash flows from payments.
- Align both series on the same monthly calendar spine.
- Compute monthly cash collection gaps -> gap amount = actual − scheduled
- SQL output: one clean monthly table with columns(**year_month**, **scheduled_cash**, **actual_cash**, **cash_flow_gap**)

**Python Method**
- Load the monthly table that shows how much cash was expected and how much cash was actually collected for each month.
- For each month, calculate the cash gap percentage by dividing the difference between actual and expected cash by the expected cash.
- Summarize the results by reporting the average gap percentage, how much the gap percentage varies from month to month, and how often monthly cash collection is below expectations.
- Plot the monthly cash gap percentage over time and label each month with its exact value so the chart is easy to read.

<br>

**Charts**

<p align="center">
  <img src="Charts/01_2_scheduled_vs_actual_cash_flow.png" width="100%">
</p>

<br>

**Key Insights**
TO BE WRITTEN

<br><br>

**1.3. Budget vs Actual Performance**

Did actual revenue earned, cash collected, and credit losses differ from what management planned?

<br>

**Tables used**
- budget_plan_monthly — planned revenue ($), cash inflow ($), net credit loss ($) by month + scenario
- payments — actual cash received and realized interest+fees
- loans — default event timing (only for labeling / cohorting if needed)
- calendar spine

<br>

**SQL Method :** <br>
This work produces two monthly tables that will eventually be joined on year_month, Monthly actuals table and Monthly budget table.

**Monthly Actuals** (what really happened) 
-> it has three columns, "**actual_revenue**" , "**actual_cash**" , "**actual_loss**", all in year_month.

For the "**actual_revenue**":
- Use the **payments** table.
- Keep only payments where **payment_type** IN ('scheduled','partial').
- Sum **paid_fee_interest** (interest + fees actually collected).
- Group results by **payment_date** month (year_month).
- Left join to **dim_month** so every month exists, and fill missing months with 0 revenue.

For the "**actual_cash**":
- Use the **payments** table.
- Sum **payment_amount** (all cash collected, regardless of type).
- Group results by **payment_date** month (**year_month**).
- Left join to **dim_month** so every month exists, and fill missing months with 0 cash.

For the "**actual_loss**":
- Use the **loans** table and keep only defaulted loans where **default_date** IS NOT NULL.
- Join those defaulted loans to payments by **loan_id**, keeping only payment rows where **payment_date** <= **default_date**.
- For each defaulted loan, sum **paid_principal** only for **payment_type** IN ('**scheduled**','**partial**') to get principal paid before default.
- Compute unpaid principal at default as: **principal** − **principal_paid_pre_default**.
- Group unpaid principal by the month of **default_date** to get monthly unpaid principal loss.
- Separately, from payments, sum **payment_amount** by month where **payment_type** = '**recovery**' to get monthly recoveries.
- Align both monthly series to **dim_month** so every month exists, and fill missing months with 0.
- Compute monthly net loss as: **unpaid_principal_loss** − **recovered_principal**.

**Monthly Budget** (what management had planned ) 
- Use the budget_plan_monthly source table at monthly grain (year_month).
- Budget values are already provided in a pivoted format with fixed scenarios and metrics.
- Select one row per year_month containing:
	**base_budget_for_revenue**
	**stretch_budget_for_revenue**
	**base_budget_for_cash**
	**stretch_budget_for_cash**
	**base_budget_for_loss**
	**stretch_budget_for_loss**
- Join to the calendar spine to ensure all months are present.
- Fill missing budget values with 0 to align with monthly actuals.
- Output a single monthly budget table to be joined to monthly actuals on year_month.

**Result** :
A table with these columns

**year_month**,

**actual_revenue**,
**budget_base_for_revenue**,
**budget_stretch_for_revenue**,

**actual_cash**,
**budget_base_for_cash**,
**budget_stretch_for_cash**,

**actual_loss**,
**budget_base_for_loss**,
**budget_stretch_for_loss**
  
<br>

**Python Method :**
- Read the pre-aggregated monthly budget vs actual CSV
- Compute monthly variance by subtracting each budget value (Base and Stretch) from the actual result, so each variance column directly measures how far real performance deviated from plan in dollars.
- Convert each monthly dollar variance into a percent of the budget, so the size of the gap is comparable across months; for losses, a positive percent means losses exceeded plan.
- This delivers 3 charts ( each for Revenue, Cash, Net Credit Loss ), each chart is a 2-row stacked figure. Top chart is the Budget VS Actual, bottom chart is the Monthly Variance % vs Base + Stretch with a zero line.

<br>

**Charts**

<p align="center">
  <img src="Charts/01_3a_budget_vs_actual_on_revenue.png" width="100%">
</p>

**Key Insights**
TO BE WRITTEN

<br>
<br>

<p align="center">
  <img src="Charts/01_3b_budget_vs_actual_on_cash.png" width="100%">
</p>

**Key Insights**
TO BE WRITTEN

<br><br>

<p align="center">
  <img src="Charts/01_3c_budget_vs_actual_on_credit_loss.png" width="100%">
</p>

<br><br>

**Key Insights**
TO BE WRITTEN

<br><br>

### 2 — Borrower Activation, Churn & Value

<br>

**2.1. Customer Activation Timing**

How long does it take customers to activate into credit usage by originating their first loan?

**Tables used**
- customers — signup date and segmentation attributes
- loans — first loan origination date
- dim_month ( the calendar spine )

**SQL Method**
- **Identify first credit usage**: Join loans to customers and, for each customer, find the earliest loan origination date that occurs on or after the signup date. This ensures activation reflects the first valid use of credit.
- **Measure activation delay**: Calculate activation time as the number of days between customer signup and first loan origination.
- **Summarize by cohort**: Group customers by signup month and compute the average and median activation days for each cohort to compare how quickly different signup cohorts activate into borrowing.
- SQL output : a table with these columns : year_month, n_customers, avg_activation_days, median_activation_days

**Python Method**
- Compute and visualize activation-time metrics by signup month, applying a cutoff defined as last month in the data minus 18 months so that only fully observable cohorts are included in trend analysis.
  
<br>

**Charts**

<p align="center">
  <img src="Charts/02_1_customer_activation_timing.png" width="100%">
</p>

**Key Insights**
TO BE WRITTEN

<br>
<br>

**2.2. Borrower Inactivity & Churn Risk**

Which customers are likely to stop borrowing or become inactive after their initial loan?

**Tables used**
- customers — customer attributes
- loans — loan timing and frequency

**SQL Method**
- The initial loan is defined as the earliest loan origination for each customer, ordered by origination date and loan ID.
- **Order loans and set end date**: Assign a fixed portfolio end date (2025-12-31) and number each customer’s loans by origination date (and **loan_id** for ties) so we can consistently identify “first” and “second” loans.
- **Isolate the first loan**: Keep only **loan_number** = 1 per customer to define the **first_loan_id** and **first_loan_date**, which anchors the customer’s borrowing start.
- **Isolate the second loan**: Keep only **loan_number** = 2 per customer to capture the earliest “return borrowing” event (**second_loan_id**, **second_loan_date**) if it exists.
- **Create the 180-day return window**: Join first and second loans and compute **daydate_180** = **first_loan_date** + 180 days to define the return window boundary.
- **Apply an observability cutoff** (include_flag): Mark customers as included only if **daydate_180** is on or before 2025-12-31, so every included customer has a fully observable 180-day return window.
- **Calculate inactivity score** :
	- For customers who are not included the score is set to NULL
	- For included customers who has no second loan, assign score = 1
	- For included customers who has second loan which is within 180 days after the first loan, calculate score = days between loans ÷ 180
	- For included customers who has second loan which is outside 180 days after the first loan, assign score = 1
- **Output the modeling table** : Return one row per customer containing first/second loan timing, the 180-day window boundary, the include_flag, and the final inactive_score target for downstream Python work.

**Python Method**
- **Load the modeling table**: Read the SQL output (one row per borrowing customer) and keep only rows where inactive_score is not null so the target is fully observable.
- **Validate the target**: Convert inactive_score to numeric and keep only valid values so summaries and plots do not break or misread the target.
- **Measure portfolio-level full inactivity**: Compute the share of observable customers with inactive_score = 1.00 to quantify customers who did not return within 180 days.
- **Summarize return timing distribution**: Describe and plot the distribution of inactive_score to separate the “fully inactive” mass at 1.00 from the return-timing spread among customers who came back within 180 days.
- **Create outcome buckets for communication**: Bucket customers into Low/Medium/High groups using fixed thresholds on inactive_score so results are easy to explain and compare across segments.

<br>

**Charts**

<p align="center">
  <img src="Charts/02_2a_borrower_inactivity_and_churn_risk.png" width="100%">
</p>

<br>

<p align="center">
  <img src="Charts/02_2b_borrower_inactivity_and_churn_risk.png" width="100%">
</p>

<br>

<p align="center">
  <img src="Charts/02_2c_borrower_inactivity_and_churn_risk.png" width="100%">
</p>

<br>

<p align="center">
  <img src="Charts/02_2d_borrower_inactivity_and_churn_risk.png" width="100%">
</p>

<br>

<p align="center">
  <img src="Charts/02_2e_borrower_inactivity_and_churn_risk.png" width="100%">
</p>

**Key Insights**
TO BE WRITTEN

<br>
<br>
