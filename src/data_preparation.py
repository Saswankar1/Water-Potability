import pandas as pd
import os
from sklearn.model_selection import train_test_split

def define_paths() -> tuple:
# Defines and returns the input and output paths.
    input_train_path = "./data/raw/train.csv"
    input_test_path = "./data/raw/test.csv"
    output_data_path = os.path.join("data", "processed")
    return input_train_path, input_test_path, output_data_path

def read_data(input_train_path: str, input_test_path: str) -> tuple:
# Reads the train and test data from CSV files.
    try:
        train_data = pd.read_csv(input_train_path)
        test_data = pd.read_csv(input_test_path)
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the input files exist.")
        return None, None

def fill_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
# Fills missing values with the median of each column in the DataFrame.
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
    return df

def save_processed_data(train_processed: pd.DataFrame, test_processed: pd.DataFrame, output_data_path: str) -> None:
# Saves the processed train and test data to CSV files.
    os.makedirs(output_data_path, exist_ok=True)
    train_processed.to_csv(os.path.join(output_data_path, "train_processed.csv"), index=False)
    test_processed.to_csv(os.path.join(output_data_path, "test_processed.csv"), index=False)

def main():
    # Define paths
    input_train_path, input_test_path, output_data_path = define_paths()

    # Read data
    train_data, test_data = read_data(input_train_path, input_test_path)
    if train_data is None or test_data is None:
        return

    # Process data
    train_processed = fill_missing_with_median(train_data)
    test_processed = fill_missing_with_median(test_data)

    # Save processed data
    save_processed_data(train_processed, test_processed, output_data_path)

    print(f"Processed data has been saved in the directory: {output_data_path}")

if __name__ == "__main__":
    main()

"""
dvc stage add -n data_preparation \
                  -d src/data_preparation.py \
                  -d data/raw \
                  -o data/processed \
                  python src/data_preparation.py

"""