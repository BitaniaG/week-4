# week-4
Credit Risk Probability Model for Alternative Data. An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Model
Credit Scoring Business Understanding

1. How Basel II Influences the Need for Interpretability

The Basel II Accord emphasizes risk measurement, capital adequacy, and model transparency.
Because financial institutions must justify how credit decisions are made, credit risk models must be:
Interpretable (easy to explain to regulators)
Documented (clear logic, assumptions, limitations)
Stable and auditable

This means that even if complex machine learning models deliver higher performance, banks must ensure the model behaves consistently and can be explained to auditors. Basel II prioritizes clarity and fairness over complexity.

2. Why We Need a Proxy Target Variable & Business Risks

The dataset does not include a true “default” label.
To train a credit risk model, we must engineer a proxy target that estimates which customers behave like high-risk borrowers.

We do this using RFM behavioral patterns, assuming that disengaged customers (low activity, low spending, long inactivity period) resemble high-risk borrowers.

However, this creates business risks:
The proxy may misclassify good customers as risky.
Behavioral patterns may not perfectly represent real repayment behavior.
Bias may be introduced if a segment is consistently labeled as risky.
Model predictions are limited by the quality of this artificial target.

Therefore, the proxy must be used carefully and improved once real repayment data becomes available.

3. Trade-offs: Simple vs Complex Models in a Regulated Context

Simple models like Logistic Regression with WoE are easy to explain, transparent, and regulator-friendly.
They show exactly how each feature affects risk, which makes them easy to audit and monitor.
However, they may miss complex patterns, so their accuracy can be slightly lower.

Complex models like Gradient Boosting or XGBoost usually predict better because they capture nonlinear relationships.
But they are harder to interpret, behave more like “black boxes,” and require explainability tools to justify decisions.
This reduced transparency makes them riskier to use in highly regulated environments.

In short:
Simple = more interpretable, safer for regulation
Complex = more accurate, but harder to explain

Top Insights from EDA: 
The exploratory data analysis revealed several important patterns in the transaction dataset. First, transaction amounts are highly skewed: most transactions are small, with a few very large transactions acting as outliers. This indicates that features like Amount may benefit from log transformation or outlier capping before modeling. Second, certain categorical features such as CustomerId, AccountId, and ProductId have very high cardinality, making one-hot encoding impractical; alternative encoding techniques like frequency encoding may be more suitable. Third, while missing values are generally minimal, some columns such as ProductCategory and PricingStrategy contain gaps that should be addressed with simple imputation or selective removal. Fourth, the numeric features Amount and Value are highly correlated, suggesting that one may be redundant for modeling. Finally, customer behavior varies across channels and product categories: some channels and product categories exhibit higher transaction volumes and amounts, indicating that these features could be strong predictors for credit risk. Additionally, patterns in customer frequency and monetary value hint at the usefulness of RFM analysis for identifying high-risk customers.