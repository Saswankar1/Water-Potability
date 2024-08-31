import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

def load_params(filepath: str) -> int:
# Loads the n_estimators parameter from a YAML file."""
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params["model_building"]["n_estimators"]

def load_data(filepath: str) -> pd.DataFrame:
# Loads the training data from a CSV file."""
    return pd.read_csv(filepath)

def separate_features_and_target(data: pd.DataFrame, target_column: str) -> tuple:
# Separates the features and the target variable from the DataFrame."""
    X = data.drop(columns=[target_column], axis=1)
    y = data[target_column]
    return X, y

def train_model(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int) -> RandomForestClassifier:
# Initializes and trains the Random Forest classifier."""
    clf = RandomForestClassifier(n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    return clf

def save_model(model: RandomForestClassifier, filepath: str) -> None:
# Saves the trained model to a file using pickle."""
    with open(filepath, "wb") as model_file:
        pickle.dump(model, model_file)

def main():
    # Load parameters
    params_filepath = "params.yaml"
    n_estimators = load_params(params_filepath)
    
    # Load data
    train_data_filepath = "./data/processed/train_processed.csv"
    train_data = load_data(train_data_filepath)
    
    # Separate features and target
    X_train, y_train = separate_features_and_target(train_data, 'Potability')
    
    # Train model
    clf = train_model(X_train, y_train, n_estimators)
    
    # Save model
    model_filepath = "model.pkl"
    save_model(clf, model_filepath)
    
    print(f"Model has been trained and saved to {model_filepath}")

if __name__ == "__main__":
    main()
"""
dvc stage add -n model_building \                  
              -d src/model_building.py \ 
              -d data/processed \
              -o model.pkl \                  
              python src/model_building.py
"""