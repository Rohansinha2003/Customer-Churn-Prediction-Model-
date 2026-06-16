import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")

def evaluate_models(models, X_test, y_test, feature_names):
    """
    Evaluates classification models, prints classification metrics, and generates evaluation plots.
    
    models: dict of {model_name: model_object}
    X_test: scaled test features
    y_test: target test labels
    feature_names: list of feature column names
    """
    metrics_summary = []
    
    # ----------------------------------------------------
    # Calculate and Print Metrics
    # ----------------------------------------------------
    print("\n" + "="*80)
    print("MODEL EVALUATION METRICS SUMMARY")
    print("=" * 80)
    
    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate scores
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        
        metrics_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        
        print(f"\nModel: {name}")
        print("-" * 40)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
    df_metrics = pd.DataFrame(metrics_summary)
    print("\nSummary Comparison Table:")
    print("-" * 60)
    print(df_metrics.to_string(index=False))
    print("=" * 80 + "\n")
    
    # ----------------------------------------------------
    # Plot 9: Confusion Matrices (Side-by-side)
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, len(models), figsize=(12, 5))
    if len(models) == 1:
        axes = [axes]
        
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Display confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False,
                    xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
        axes[i].set_title(f"{name} Confusion Matrix")
        axes[i].set_xlabel("Predicted Label")
        axes[i].set_ylabel("True Label")
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrices.png"), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Plot 10: ROC Curves
    # ----------------------------------------------------
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
            
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.500)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "roc_curves.png"), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Plot 11: Feature Importance (Random Forest only)
    # ----------------------------------------------------
    rf_model = models.get("Random Forest")
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 15 features for clarity if there are many
        top_n = min(15, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=importances[top_indices], 
            y=np.array(feature_names)[top_indices], 
            hue=np.array(feature_names)[top_indices],
            palette="viridis",
            legend=False
        )
        plt.title(f"Top {top_n} Feature Importances (Random Forest Model)")
        plt.xlabel("Relative Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "feature_importances.png"), dpi=300)
        plt.close()
        
        # Print top feature importances text-wise
        print("TOP 10 MOST IMPORTANT FEATURES (Random Forest):")
        print("-" * 50)
        for rank, idx in enumerate(top_indices[:10], 1):
            print(f"{rank}. {feature_names[idx]}: {importances[idx]:.4f}")
        print("=" * 80 + "\n")
        
    return df_metrics

if __name__ == "__main__":
    # Test script execution
    from data_loader import load_data
    from preprocessing import clean_data, preprocess_features, split_and_scale
    from modeling import train_logistic_regression, train_random_forest
    
    df = load_data()
    df_clean = clean_data(df)
    X, y = preprocess_features(df_clean)
    X_train, X_test, y_train, y_test, feature_names = split_and_scale(X, y)
    
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model
    }
    
    evaluate_models(models, X_test, y_test, feature_names)
    print("Evaluation scripts executed and plots generated successfully!")
