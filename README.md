# Water Potability Prediction using DVC

This project aims to predict the potability of water using a Machine learn model integrated with DVC (Data Version Control) for managing datasets, experiments, and pipelines.

## Project Structure

- **data/**: Contains raw and processed datasets.
- **src/**: Holds the scripts for data processing, model training, and evaluation.
- **dvc.yaml**: Defines the DVC pipeline stages.
- **params.yaml**: Configuration parameters for the project.
- **dvc.lock**: Locks the stages with specific versions of data and code.

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/Saswankar1/Water-potability-DVC.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Initialize the DVC pipeline:
   ```bash
   dvc init
   ```
2. Run the pipeline:
   ```bash
   dvc repro
   ```

## Contributing

Feel free to fork this repository and submit pull requests.
