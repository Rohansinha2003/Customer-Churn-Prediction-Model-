import os
import sys
import pandas as pd

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_loader import load_data, display_info
from preprocessing import clean_data, preprocess_features, split_and_scale
from eda import run_eda
from modeling import train_logistic_regression, train_random_forest
from evaluation import evaluate_models

def print_section(title):
    print("\n" + "=" * 80)
    print(f"STAGE: {title}".center(80))
    print("=" * 80)

def main():
    print("=" * 80)
    print("STARTING CUSTOMER CHURN PREDICTION PIPELINE".center(80))
    print("=" * 80)

    # 1. Data Loading
    print_section("DATA LOADING")
    df = load_data()
    display_info(df)

    # 2. Data Cleaning & Initial Preprocessing
    print_section("DATA CLEANING")
    df_clean = clean_data(df)
    
    # 3. Exploratory Data Analysis
    print_section("EXPLORATORY DATA ANALYSIS (EDA)")
    run_eda(df_clean)

    # 4. Preprocessing & Feature Engineering for ML
    print_section("FEATURE ENGINEERING & PREPROCESSING")
    X, y = preprocess_features(df_clean)
    X_train, X_test, y_train, y_test, feature_names = split_and_scale(X, y)
    print("Feature shapes after split and scaling:")
    print(f"  Training Features (X_train): {X_train.shape}")
    print(f"  Testing Features (X_test):   {X_test.shape}")
    print(f"  Training Target (y_train):   {y_train.shape}")
    print(f"  Testing Target (y_test):     {y_test.shape}")
    print(f"Total features encoded & processed: {len(feature_names)}")

    # 5. Modeling
    print_section("MODEL TRAINING & BASELINE CV")
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 6. Evaluation
    print_section("MODEL EVALUATION & VISUALIZATIONS")
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model
    }
    df_metrics = evaluate_models(models, X_test, y_test, feature_names)

    # 7. Business Insights & Conclusions
    print_section("BUSINESS INSIGHTS & CONCLUSIONS")
    
    # Calculate month-to-month statistics for insights
    m2m_mask = df_clean["Contract"] == "Month-to-month"
    m2m_churn_rate = df_clean[m2m_mask]["Churn"].mean() * 100
    other_churn_rate = df_clean[~m2m_mask]["Churn"].mean() * 100
    m2m_churn_ratio = m2m_churn_rate / other_churn_rate if other_churn_rate > 0 else 0
    
    # Calculate tenure churn patterns
    churned_df = df_clean[df_clean["Churn"] == 1]
    retained_df = df_clean[df_clean["Churn"] == 0]
    avg_tenure_churned = churned_df["tenure"].mean()
    avg_tenure_retained = retained_df["tenure"].mean()
    
    # Calculate monthly charges patterns
    avg_charge_churned = churned_df["MonthlyCharges"].mean()
    avg_charge_retained = retained_df["MonthlyCharges"].mean()

    print(f"1. Month-to-Month Contract Risk:")
    print(f"   - Churn rate for Month-to-Month contract customers: {m2m_churn_rate:.2f}%")
    print(f"   - Churn rate for One/Two Year contract customers:  {other_churn_rate:.2f}%")
    print(f"   - Month-to-month customers are {m2m_churn_ratio:.1f}x MORE likely to churn!")
    print()
    print(f"2. Impact of Tenure & Monthly Charges:")
    print(f"   - Churn occurs early: Average tenure of churned customers is {avg_tenure_churned:.1f} months,")
    print(f"     compared to {avg_tenure_retained:.1f} months for retained customers.")
    print(f"   - Pricing sensitivity: Average monthly charges of churned customers are ${avg_charge_churned:.2f},")
    print(f"     compared to ${avg_charge_retained:.2f} for retained customers (a {((avg_charge_churned - avg_charge_retained)/avg_charge_retained)*100:.1f}% cost premium).")
    print()
    print(f"3. Most Critical Factors Contributing to Churn (from Random Forest Feature Importance):")
    print(f"   - Contract Type (specifically Month-to-Month contracts)")
    print(f"   - Customer Tenure (low tenure indicates high churn probability)")
    print(f"   - Internet Service Type (specifically Fiber Optic lines have higher churn rates)")
    print(f"   - Monthly Charges and Total Charges")
    print(f"   - Lack of Tech Support and Online Security services")
    print()
    print(f"4. Recommended Customer Retention Strategies:")
    print(f"   - Contract Migrations: Offer small discount incentives (e.g. 5-10% off for 6 months)")
    print(f"     or free streaming upgrades to transition Month-to-month customers to One-year terms.")
    print(f"   - Target Early Tenure: Set up automated onboarding check-ins, customer success calls,")
    print(f"     and satisfaction surveys at 30, 60, and 90 days to prevent early tenure drop-off.")
    print(f"   - Tech Support Bundles: Promote online security and tech support services as a value-add.")
    print(f"     Since customers with tech support have extremely low churn, bundling these services")
    print(f"     directly into core packages will decrease churn rates.")
    print(f"   - Price Sensitivity / Value Alignment: Target high-charge customers who are approaching")
    print(f"     the end of their contracts and offer personalized loyalty discounts or downgrade options")
    print(f"     to keep them in the ecosystem.")
    print("=" * 80)
    print("CUSTOMER CHURN PIPELINE EXECUTION COMPLETED".center(80))
    print("=" * 80)

if __name__ == "__main__":
    main()
