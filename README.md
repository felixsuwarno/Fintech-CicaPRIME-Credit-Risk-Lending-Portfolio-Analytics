WIP ( Work in Progress ) Jan 30th 2026

# Consumer Credit “Cica PRIME” Risk & Portfolio Analytics  
**Risk Management • Portfolio Analytics • Forecasting • Stress Testing**

This project analyzes synthetic but realistic consumer installment lending data from CICA (Consumer Installment Credit Account) Prime, a simulated consumer credit product designed to resemble real-world installment lending / BNPL-style portfolios offered by fintechs and traditional banks. The product extends unsecured credit to consumers and generates revenue through interest and fees, while incurring credit losses through delinquency and default.

The dataset spans January 2023 to December 2025 and includes loan-level, customer-level, and payment-level data, along with contractual payment schedules, macroeconomic scenarios, and internal budget plans. The data captures the full credit lifecycle: customer acquisition, underwriting decisions, loan origination, repayment behavior, delinquency, default, recoveries, and portfolio-level financial outcomes.

The analysis focuses on risk-aware, finance-aligned analytics, modeling how portfolio growth, repayment behavior, and credit losses are expected to evolve over time under different conditions. The goal is to evaluate portfolio health, revenue sustainability, and downside risk using predictive modeling, time-series forecasting, and scenario-based stress testing, consistent with how consumer lending analytics teams operate in practice.

<br>

➤ **Project Goal / Purpose:**  

The goal of this project is to understand how CICA Prime’s consumer lending portfolio grows over time, how stable that growth is, and how exposed the business is to downside risk. The analysis uses forecasting, predictive modeling, and standard credit risk techniques commonly applied in consumer lending.

The project looks at how revenue, cash inflows, delinquency, and defaults are expected to change over time, and compares actual results to internal plans to evaluate how accurate and resilient the business’s financial planning is. At the borrower and loan level, the analysis estimates customer lifetime value, the likelihood that borrowers stop using the product, and the likelihood that loans default, with the goal of using past behavior to make informed predictions about future outcomes.

At the portfolio level, the project measures expected credit losses, evaluates how concentrated revenue and risk are across borrowers, and tests how the portfolio would perform under different economic conditions. The goal is to determine whether portfolio growth is supported by sustainable cash flows and manageable credit risk, or whether it relies on elevated risk, loose underwriting, or a small group of borrowers—especially when economic conditions worsen.

<br>

➤ **Skills Demonstrated:**  

Consumer lending analytics and KPI design, credit portfolio performance analysis, delinquency and default analysis (DPD, roll rates, vintage curves), cohort-based customer lifetime value (LTV) modeling, probability of default (PD) modeling, loss analysis using EAD and LGD concepts, revenue and cash-flow analysis, budget vs actual variance analysis, and macro-driven stress testing and scenario analysis

**(SQL • Python • Pandas • Time-Series Forecasting • Credit Risk Metrics • Executive-Ready Analysis)**

<br>

➤ **Core Business Questions:**

**1 — Forecasting & Financial Planning**

1. **Revenue performance & outlook** <br> How did monthly interest and fee revenue perform from 2023–2025, and what is the expected revenue trajectory over the next 12 months?
2. **Cash inflows vs contractual expectations** <br> How do actual monthly cash collections compare to scheduled payments from 2023–2025, and how reliable are collection gaps across these 36 months?
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























