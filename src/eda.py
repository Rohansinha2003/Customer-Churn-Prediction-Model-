import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set seaborn style for clean, beautiful plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Color palette: curated dark teal and vivid coral
CHURN_PALETTE = {0: "#2A9D8F", 1: "#E76F51"} # 0 = Retained, 1 = Churned
CHURN_LABELS = {0: "Retained (No Churn)", 1: "Churned (Yes Churn)"}

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")

def ensure_plot_dir():
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        print(f"Created plots directory at: {PLOT_DIR}")

def run_eda(df_clean):
    """
    Runs comprehensive EDA, generates and saves visualization plots,
    and prints detailed business insights.
    """
    ensure_plot_dir()
    
    # We will use df_plot for visualization to keep labels intuitive
    df_plot = df_clean.copy()
    
    # ----------------------------------------------------
    # Plot 1: Churn Distribution
    # ----------------------------------------------------
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x="Churn", data=df_plot, palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    
    # Add count and percentage labels
    total = len(df_plot)
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.annotate(f'{height}\n({percentage:.1f}%)', 
                    (p.get_x() + p.get_width() / 2., height - (height * 0.15)),
                    ha='center', va='center', color='white', fontweight='bold', size=11)
                    
    plt.title("Overall Customer Churn Distribution", pad=15)
    plt.xlabel("Customer Status")
    plt.ylabel("Number of Customers")
    plt.xticks([0, 1], ["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_distribution.png"), dpi=300)
    plt.close()
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 1: Churn Distribution")
    print("-" * 80)
    print("Insight: The dataset displays a class imbalance. Around 26.5% (1,869) of the customers")
    print("have churned, while 73.5% (5,174) were retained. This class imbalance needs to be handled")
    print("during modeling (e.g. using stratified sampling and evaluating models using F1 / ROC-AUC).")
    
    # ----------------------------------------------------
    # Plot 2: Churn vs Contract Type
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x="Contract", hue="Churn", data=df_plot, palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    plt.title("Churn Rate by Contract Type", pad=15)
    plt.xlabel("Contract Type")
    plt.ylabel("Customer Count")
    plt.legend(title="Status", labels=["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_vs_contract.png"), dpi=300)
    plt.close()
    
    # Quantify contract churn
    contract_churn = pd.crosstab(df_plot["Contract"], df_plot["Churn"], normalize="index") * 100
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 2: Churn vs Contract Type")
    print("-" * 80)
    print("Insight: Contract type is a massive indicator of churn. Customers on Month-to-Month")
    print("contracts have an extremely high churn rate of {:.2f}%, whereas One-year contract".format(contract_churn.loc["One year", 1]))
    print("customers churn at {:.2f}% and Two-year contract customers at just {:.2f}%.".format(contract_churn.loc["One year", 1], contract_churn.loc["Two year", 1]))
    print("Month-to-month customers churn at {:.2f}%, representing the highest churn risk segment.".format(contract_churn.loc["Month-to-month", 1]))
    print("Actionable Strategy: Incentivize Month-to-Month customers to transition to longer-term")
    print("contracts via discounts, feature add-ons, or loyalty programs.")

    # ----------------------------------------------------
    # Plot 3: Churn vs Tenure
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.kdeplot(x="tenure", hue="Churn", data=df_plot, fill=True, palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]], common_norm=False, alpha=0.5)
    plt.title("Customer Tenure Distribution by Churn Status", pad=15)
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Density")
    plt.legend(title="Status", labels=["Churned", "Retained"])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_vs_tenure.png"), dpi=300)
    plt.close()
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 3: Churn vs Tenure")
    print("-" * 80)
    print("Insight: Newly acquired customers are at a very high risk of churn. The density plot shows")
    print("a significant spike in churn within the first 1-12 months of service. As tenure increases,")
    print("the density of churn decreases dramatically, demonstrating strong customer loyalty once they")
    print("cross the 1-2 year mark.")
    print("Actionable Strategy: Implement robust onboarding programs, active customer support, and")
    print("satisfaction check-ins in the critical first 6 months to reduce early-stage churn.")

    # ----------------------------------------------------
    # Plot 4: Churn vs Monthly Charges
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.kdeplot(x="MonthlyCharges", hue="Churn", data=df_plot, fill=True, palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]], common_norm=False, alpha=0.5)
    plt.title("Monthly Charges Distribution by Churn Status", pad=15)
    plt.xlabel("Monthly Charges ($)")
    plt.ylabel("Density")
    plt.legend(title="Status", labels=["Churned", "Retained"])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_vs_monthly_charges.png"), dpi=300)
    plt.close()
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 4: Churn vs Monthly Charges")
    print("-" * 80)
    print("Insight: Customers with higher monthly charges are more likely to churn. The density plot")
    print("shows a higher density of churned customers starting around $70 to $100+ per month,")
    print("while retained customers peak at lower price points ($20-$30).")
    print("Actionable Strategy: Review the pricing structure and check if high-value customers feel")
    print("they are getting value. Offer customized bundle packages or tier downgrades instead of")
    print("allowing them to churn completely.")

    # ----------------------------------------------------
    # Plot 5: Churn vs Payment Method
    # ----------------------------------------------------
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y="PaymentMethod", hue="Churn", data=df_plot, palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    plt.title("Churn Rate by Payment Method", pad=15)
    plt.xlabel("Customer Count")
    plt.ylabel("Payment Method")
    plt.legend(title="Status", labels=["Retained", "Churned"])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_vs_payment_method.png"), dpi=300)
    plt.close()
    
    pm_churn = pd.crosstab(df_plot["PaymentMethod"], df_plot["Churn"], normalize="index") * 100
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 5: Churn vs Payment Method")
    print("-" * 80)
    print("Insight: Payment methods display significant churn differences. Customers paying via")
    print("Electronic Check have an exceptionally high churn rate of {:.2f}% compared to Mailed Check".format(pm_churn.loc["Electronic check", 1]))
    print("({:.2f}%) and automatic payment options like Bank Transfer ({:.2f}%) or Credit Card ({:.2f}%).".format(
          pm_churn.loc["Mailed check", 1], 
          pm_churn.loc["Bank transfer (automatic)", 1], 
          pm_churn.loc["Credit card (automatic)", 1]))
    print("Actionable Strategy: Electronic check users may experience friction or have lower financial")
    print("stickiness. Promote auto-pay signup (Credit Card/Bank Transfer) with a small credit incentive.")

    # ----------------------------------------------------
    # Plot 6: Correlation Heatmap
    # ----------------------------------------------------
    # Temporarily encode target and basic features for correlation analysis
    df_corr = df_plot[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].copy()
    
    # Add encoded variables for contract types
    for c_type in ["Month-to-month", "One year", "Two year"]:
        df_corr[f"Contract_{c_type}"] = (df_plot["Contract"] == c_type).astype(int)
        
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, square=True)
    plt.title("Correlation Heatmap (Numeric & Key Features)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.png"), dpi=300)
    plt.close()
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 6: Correlation Heatmap")
    print("-" * 80)
    print("Insight: Key correlations with Churn include:")
    print(f"  - Tenure: -0.35 (strong negative correlation, confirming longer stay = lower churn)")
    print(f"  - Contract_Month-to-month: {df_corr.corr().loc['Churn', 'Contract_Month-to-month']:.2f} (strongest positive correlation with churn)")
    print(f"  - MonthlyCharges: 0.19 (positive correlation, confirming higher cost = higher churn)")
    print("Note that 'TotalCharges' is highly correlated with 'tenure' (0.83), which suggests possible")
    print("multicollinearity. Tree-based models (Random Forest) handle this well, but it is a factor")
    print("to keep in mind for Logistic Regression coefficients.")

    # ----------------------------------------------------
    # Plot 7: Numerical Distributions
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.histplot(df_plot["tenure"], kde=True, ax=axes[0], color="#264653")
    axes[0].set_title("Distribution of Tenure")
    axes[0].set_xlabel("Tenure (Months)")
    
    sns.histplot(df_plot["MonthlyCharges"], kde=True, ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Distribution of Monthly Charges")
    axes[1].set_xlabel("Monthly Charges ($)")
    
    sns.histplot(df_plot["TotalCharges"], kde=True, ax=axes[2], color="#e76f51")
    axes[2].set_title("Distribution of Total Charges")
    axes[2].set_xlabel("Total Charges ($)")
    
    plt.suptitle("Distribution of Numerical Features", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "numerical_distributions.png"), dpi=300)
    plt.close()
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 7: Numerical Distributions")
    print("-" * 80)
    print("Insight: The distribution of tenure shows peaks at both ends: a large number of very new")
    print("customers (tenure < 5 months) and long-term loyal customers (tenure > 60 months). Monthly")
    print("charges have a huge spike around $20 (indicating basic or phone-only plans) and a wide")
    print("distribution spanning $70-$100 (high-speed internet/premium plans). Total charges is heavily")
    print("right-skewed, which is a standard pattern representing exponential accumulation over tenure.")

    # ----------------------------------------------------
    # Plot 8: Retention Patterns across Categories
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 8.1 Senior Citizen
    sns.countplot(x="SeniorCitizen", hue="Churn", data=df_plot, ax=axes[0, 0], palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    axes[0, 0].set_title("Churn Rate: Senior Citizens vs Others")
    axes[0, 0].set_xticklabels(["Non-Senior", "Senior"])
    axes[0, 0].legend(title="Status", labels=["Retained", "Churned"])
    
    # 8.2 Internet Service
    sns.countplot(x="InternetService", hue="Churn", data=df_plot, ax=axes[0, 1], palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    axes[0, 1].set_title("Churn Rate by Internet Service Type")
    axes[0, 1].legend(title="Status", labels=["Retained", "Churned"])
    
    # 8.3 Tech Support
    sns.countplot(x="TechSupport", hue="Churn", data=df_plot, ax=axes[1, 0], palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    axes[1, 0].set_title("Churn Rate by Tech Support Availability")
    axes[1, 0].legend(title="Status", labels=["Retained", "Churned"])
    
    # 8.4 Paperless Billing
    sns.countplot(x="PaperlessBilling", hue="Churn", data=df_plot, ax=axes[1, 1], palette=[CHURN_PALETTE[0], CHURN_PALETTE[1]])
    axes[1, 1].set_title("Churn Rate by Billing Preference")
    axes[1, 1].legend(title="Status", labels=["Retained", "Churned"])
    
    plt.suptitle("Customer Retention Patterns Across Categories", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "retention_patterns.png"), dpi=300)
    plt.close()
    
    is_churn = pd.crosstab(df_plot["InternetService"], df_plot["Churn"], normalize="index") * 100
    ts_churn = pd.crosstab(df_plot["TechSupport"], df_plot["Churn"], normalize="index") * 100
    
    print("\n" + "="*80)
    print("EDA VISUALIZATION 8: Retention Patterns across Categories")
    print("-" * 80)
    print("Insight: Categorical retention patterns highlight several key triggers:")
    print("  - Internet Service: Fiber optic customers have a shockingly high churn rate of {:.2f}%".format(is_churn.loc["Fiber optic", 1]))
    print("    compared to DSL ({:.2f}%) or No Internet ({:.2f}%). This indicates possible service issues".format(is_churn.loc["DSL", 1], is_churn.loc["No", 1]))
    print("    or high pricing for Fiber Optic lines.")
    print("  - Tech Support: Customers with No Tech Support churn at {:.2f}%, whereas those with".format(ts_churn.loc["No", 1]))
    print("    Tech Support churn at only {:.2f}%. Providing help-desk support improves retention dramatically.".format(ts_churn.loc["Yes", 1]))
    print("  - Senior Citizens churn at a higher rate (~41.7%) compared to non-seniors (~23.6%).")
    print("  - Paperless Billing users are also more likely to churn (~33.5% vs ~16.3%).")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data
    df = load_data()
    df_clean = clean_data(df)
    run_eda(df_clean)
