# Telecom Customer Churn Prediction

Predicting customer churn using machine learning to identify high-risk accounts and design proactive retention strategies.

---

## 📌 Project Overview & Objective

Customer churn (customer attrition) is one of the most critical metrics for subscription-based businesses, particularly in telecommunications. Acquiring new customers is often 5 to 25 times more expensive than retaining existing ones. 

The objective of this project is to:
1. **Analyze a telecom customer dataset** of approximately 7,043 customers to identify drivers of customer attrition.
2. **Build and compare classification models** (Logistic Regression & Random Forest) to predict churn with high accuracy.
3. **Formulate data-driven retention strategies** to prevent churn, focusing specifically on high-risk customer segments.

---

## 📁 Repository Structure

```directory
├── data/                            # Raw dataset (downloaded programmatically)
├── plots/                           # Generated EDA and Model Evaluation plots
│   ├── churn_distribution.png
│   ├── churn_vs_contract.png
│   ├── churn_vs_tenure.png
│   ├── churn_vs_monthly_charges.png
│   ├── churn_vs_payment_method.png
│   ├── correlation_heatmap.png
│   ├── numerical_distributions.png
│   ├── retention_patterns.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── feature_importances.png
├── src/                             # Source code modules
│   ├── data_loader.py               # Data retrieval and loading
│   ├── eda.py                       # Exploratory Data Analysis & visual generation
│   ├── preprocessing.py             # Cleaning, encoding, and scaling
│   ├── modeling.py                  # Model training and cross-validation
│   └── evaluation.py                # Model evaluation and metrics plotting
├── churn_analysis_notebook.ipynb    # Clean Jupyter notebook containing the pipeline
├── main.py                          # Main execution orchestrator
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## 📊 Dataset Description

The dataset used is the IBM Telco Customer Churn dataset, representing 7,043 customers and 21 features.
* **Target Variable**: `Churn` (Yes or No)
* **Demographics**: Gender, Senior Citizen status, Partner, and Dependents.
* **Services Subscribed**: Phone service, Multiple lines, Internet service (DSL, Fiber optic, None), Online security, Online backup, Device protection, Tech support, Streaming TV, and Streaming Movies.
* **Account Info**: Tenure (months), Contract type (Month-to-month, One year, Two year), Paperless billing, Payment method, Monthly charges, and Total charges.

---

## 🛠️ Data Cleaning & Preprocessing

* **ID Removal**: Dropped the non-predictive `customerID` column.
* **DataType Correction**: Converted `TotalCharges` from string/object to float.
* **Missing Value Imputation**: Handled 11 blank spaces in `TotalCharges` (originating from new customers with a tenure of 0) by filling them with `0.0`.
* **Categorical Encoding**:
  - One-hot encoded multi-class variables (Contract type, Payment method, Internet service) with `drop_first=True` to prevent collinearity.
  - Mapped binary categories (e.g. `Churn`) to `0`/`1`.
* **Feature Scaling**: Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using Scikit-Learn's `StandardScaler` fitted exclusively on the training split to avoid data leakage.
* **Validation Split**: Used an 80/20 train-test split with **stratified sampling** to maintain the target variable's distribution.

---

## 📈 Key Exploratory Data Analysis (EDA) Insights

Detailed analyses are saved as high-resolution figures in the `plots/` directory:

1. **Churn Distribution**: 26.5% of the customer base has churned, signifying a class imbalance.
2. **Contract Type**: Month-to-month contract customers have a massive **42.71% churn rate**, compared to just **6.76%** for long-term contract (One/Two Year) customers. **Month-to-month customers are 6.3x more likely to churn!**
3. **Tenure**: Customer churn is heavily front-loaded. A significant portion of churn occurs within the **first 12 months** of service (average tenure of churned customers is 18.0 months vs. 37.6 months for retained).
4. **Monthly Charges**: Churned customers pay higher monthly fees on average ($74.44 vs. $61.27 for retained, a **21.5% cost premium**). High bills are directly linked to attrition.
5. **Payment Method**: Electronic Check users experience the highest churn rate (45.29%). Those on credit card or bank transfer auto-pay exhibit the highest retention rates (~15-16% churn).
6. **Value Added Services**: Lack of Tech Support and Online Security correlates with high churn. Fiber optic users also churn at a higher rate (41.89%) than DSL users (18.96%).

---

## 🤖 Model Comparison & Results

We trained and evaluated **Logistic Regression** and a **Random Forest Classifier** using 5-fold stratified cross-validation. 

### Performance Summary on Test Set

| Metric | Logistic Regression | Random Forest |
| :--- | :---: | :---: |
| **Accuracy** | **80.62%** | 80.41% |
| **Precision** | 65.93% | **67.63%** |
| **Recall** | **55.88%** | 50.27% |
| **F1-Score** | **60.49%** | 57.67% |
| **ROC-AUC** | 84.22% | **84.33%** |

### Top Predictors of Churn (Random Forest Importance)
1. **Tenure** (19.5%) - Loyalty/duration of subscription.
2. **TotalCharges** (14.4%) - Financial commitment.
3. **Fiber Optic Service** (9.6%) - Internet configuration.
4. **MonthlyCharges** (8.8%) - Monthly pricing pressure.
5. **Two-Year Contract** (8.0%) - Contractual lock-in.

---

## 💡 Recommended Customer Retention Strategies

Based on the model outcomes and EDA, we recommend the following four retention pillars:

1. **Incentivize Contract Migrations**: Implement targeted promotional offers (e.g., a 10% discount for 6 months or free premium channel upgrades) to migrate high-risk Month-to-Month customers to One-year contract terms. This directly addresses the 6.3x risk multiplier.
2. **Optimize Early Customer Onboarding**: Since churn is concentrated within the first year, establish an automated customer success campaign. Conduct automated check-ins and satisfaction surveys at the 30, 60, and 90-day marks.
3. **Promote Auto-Pay Subscriptions**: Incentivize Electronic Check users to switch to automatic payments (Credit Card or Direct Debit) by offering a one-time $5 billing credit. This increases payment reliability and decreases transaction friction.
4. **Bundle Value-Added Services**: Customers subscribing to `TechSupport` and `OnlineSecurity` display significantly higher retention. Bundle these services into higher-tier plans or offer them as complimentary services to high-charge fiber-optic customers showing warning signs of churn.

---

## 🚀 How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Rohansinha2003/Customer-Churn-Prediction-Model-.git
   cd Customer-Churn-Prediction-Model-
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the End-to-End Pipeline**:
   ```bash
   python main.py
   ```
   This script will download the dataset, perform EDA, save visualization plots to `plots/`, train the models, output validation reports, and display business insights.

4. **Run Jupyter Notebook**:
   ```bash
   jupyter notebook churn_analysis_notebook.ipynb
   ```
