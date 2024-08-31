import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

def construct_file_path(base_dir: str, relative_path: str) -> str:
# Constructs the full file path using the base directory and relative path.
    return os.path.join(os.path.dirname(base_dir), relative_path)

def load_data(csv_file_path: str) -> pd.DataFrame:
# Loads data from a CSV file.
    return pd.read_csv(csv_file_path)

def load_params(filepath: str) -> float:
# Loads the test size parameter from a YAML file.
    with open(filepath, 'r') as file:
        params = yaml.safe_load(file)
    return params["data_collection"]["test_size"]

def split_data(data: pd.DataFrame, test_size: float) -> tuple:
# Splits the data into training and test sets.
    return train_test_split(data, test_size=test_size, random_state=42)

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str) -> None:
# Saves the training and test data to CSV files in the specified directory.
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

def main():
    # Paths
    base_dir = __file__
    csv_file_path = construct_file_path(base_dir, '../csv/water_potability.csv')
    params_file_path = construct_file_path(base_dir, '../params.yaml')
    output_dir = construct_file_path(base_dir, '../data/raw')

    # Load data and parameters
    data = load_data(csv_file_path)
    test_size = load_params(params_file_path)

    # Split data
    train_data, test_data = split_data(data, test_size)

    # Save data
    save_data(train_data, test_data, output_dir)

    print(f"Training and test data have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    main()

"""
For adding this stage to a DVC pipeline, use the following command:

    dvc stage add -n data_collection \
                  -d src/data_collection.py \
                  -d csv/water_potability.csv \
                  -o data/raw \
                  python src/data_collection.py

Explanation:
- -n data_collection: Names the stage 'data_collection'.
- -d src/data_collection.py: Specifies the script as a dependency.
- -d csv/water_potability.csv: Adds the CSV file as a dependency since the data source is important.
- -o src/data/raw: Specifies the output directory where the processed data will be stored.
"""
