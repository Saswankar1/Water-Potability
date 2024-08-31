import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_data(filepath: str) -> pd.DataFrame:
# Loads the test data from a CSV file."""
    return pd.read_csv(filepath)

def separate_features_and_target(data: pd.DataFrame, target_column: str) -> tuple:
# Separates the features and the target variable from the DataFrame."""
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]
    return X, y

def load_model(filepath: str):
# Loads the trained model from a file using pickle."""
    with open(filepath, "rb") as model_file:
        model = pickle.load(model_file)
    return model

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
# Makes predictions and calculates evaluation metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1score': f1_score(y_test, y_pred, average='binary')
    }
    return metrics

def save_metrics(metrics: dict, filepath: str) -> None:
# Saves the evaluation metrics to a JSON file."""
    with open(filepath, 'w') as file:
        json.dump(metrics, file, indent=4)

def main():
    # File paths
    test_data_filepath = "./data/processed/test_processed.csv"
    model_filepath = "model.pkl"
    metrics_filepath = "metrics.json"
    
    # Load and prepare data
    test_data = load_data(test_data_filepath)
    X_test, y_test = separate_features_and_target(test_data, 'Potability')
    
    # Load model
    model = load_model(model_filepath)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics
    save_metrics(metrics, metrics_filepath)
    
    print(f"Evaluation metrics have been saved to {metrics_filepath}")

if __name__ == "__main__":
    main()
