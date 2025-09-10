import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_and_clean_data

def train_and_save_model(X: pd.DataFrame, y: pd.Series):
    """
    Splits data, scales features, trains a RandomForestClassifier,
    and saves the trained model and scaler to disk.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target data.
    """
    if X is None or y is None:
        print("Feature or target data is missing. Halting training.")
        return

   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")
    
  
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    
    joblib.dump(model, 'random_forest_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler have been saved successfully as 'random_forest_model.joblib' and 'scaler.joblib'.")


if __name__ == '__main__':
    
    _, X, y = load_and_clean_data('Medicaldataset.csv')
    
    train_and_save_model(X, y)
