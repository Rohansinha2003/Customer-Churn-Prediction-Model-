import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    """
    Cleans raw customer data:
    - Convers TotalCharges to numeric, handles spaces as NaN and fills with 0.0.
    - Encodes Churn (target variable) to 0/1.
    - Drops customerID.
    """
    df_clean = df.copy()
    
    # Drop customerID if exists
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop("customerID", axis=1)
        
    # Handle TotalCharges missing/blank values
    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    
    # Check for missing values in TotalCharges
    num_missing = df_clean["TotalCharges"].isnull().sum()
    if num_missing > 0:
        print(f"Handling {num_missing} missing/blank values in TotalCharges by setting them to 0.0 (since tenure is 0)")
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)
        
    # Encode target variable
    if "Churn" in df_clean.columns:
        df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})
        
    return df_clean

def preprocess_features(df_clean):
    """
    Encodes categorical features and returns features (X) and target (y).
    """
    df_encoded = df_clean.copy()
    
    # Target variable
    y = df_encoded["Churn"]
    X = df_encoded.drop("Churn", axis=1)
    
    # Identify binary and multi-class categorical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    # Use one-hot encoding for all categorical variables
    # We set drop_first=True to prevent multicollinearity (especially important for Logistic Regression)
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X_encoded, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into train and test sets using stratified sampling, 
    and scales the numerical features using StandardScaler (fit on train, transform on train & test).
    """
    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Identify numerical columns to scale
    # tenure, MonthlyCharges, TotalCharges
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    # Scale numerical columns
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    # Test script execution
    from data_loader import load_data
    df = load_data()
    df_clean = clean_data(df)
    X_enc, y = preprocess_features(df_clean)
    X_train, X_test, y_train, y_test, cols = split_and_scale(X_enc, y)
    print("Preprocessing successful!")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"Features: {len(cols)}")
