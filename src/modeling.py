import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Trains a Logistic Regression model with cross-validation.
    """
    print("Training Logistic Regression model...")
    # Using L2 regularization, increased max_iter to ensure convergence
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    
    print(f"Logistic Regression 5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Fit the final model
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, random_state=42):
    """
    Trains a Random Forest classifier with cross-validation.
    """
    print("Training Random Forest Classifier model...")
    # Setting max_depth and min_samples_leaf to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1
    )
    
    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    
    print(f"Random Forest 5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Fit the final model
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data, preprocess_features, split_and_scale
    
    df = load_data()
    df_clean = clean_data(df)
    X, y = preprocess_features(df_clean)
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    print("Model training execution successful!")
