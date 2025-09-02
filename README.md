# Machine Learning Operations

[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/billyz313/Machine-Learning-Operations/blob/main/LICENSE)
[![IMapApps: Development](https://img.shields.io/badge/IMapApps-Development-green)](https://imapapps.com)

## 🎯 Purpose

This project (`ml_ops_project`) is a standalone collection of Python scripts designed to handle the **Machine Learning Operations (MLOps)** lifecycle for the `nesis_project` web application. It focuses on tasks like **data generation, preprocessing, and model training** for a Natural Language Understanding (NLU) model, intended to power a search or query understanding feature within `nesis_project`.

### Key Benefits
* **Separation of Concerns**: Cleaner distinction between web application logic and ML pipeline
* **Independent Dependencies**: Allows specific ML libraries and their versions without affecting the web app
* **Flexible Execution**: Enables running resource-intensive ML tasks on dedicated environments


## 📋 Prerequisites

Before setting up this project, ensure you have:

- Python 3.12 or higher installed
- Conda package manager
- Access to the `nesis_project` codebase
- Sufficient disk space (at least 2GB recommended)
- Git installed (for version control)

## 🛠️ Installation
 
### 1. Clone the Repository

```bash
git clone git@github.com:NASA-Earth-Rise/Machine-Learning-Operations.git ml_ops_project

cd ml_ops_project

```

### 2. Conda Environment Setup

1. **Create the Environment**:

```bash
conda env create -f environment.yml
```

2.  **Activate the Environment**:

    ```bash
    conda activate ml_ops_project
    ```
    You will need to activate this environment every time you work on the `ml_ops_project`.

### 3. Configuration

This project relies on a `data.json` file to configure paths to your `nesis_project` database and other critical directories.

1. **Create `data.json`**:
Create a file named `data.json` in the project root with the following structure:

    ```json
    {
      "nesis_project_full_path": "{full_path_to_local_nesis_project}",
      "csv_directory": "{full_path_to_directory}",
      "csv_file_name": "{name_for_training_csv}.csv",
      "training_data_directory": "{full_path_to_training_data_directory}",
      "temp_model_save_path": "{full_path_to_temp_model_save_directory}",
      "model_save_path": "{full_path_plus_name_you_want_for_model}",
      "model_results_directory": "{full_path_for_results}",
      "model_log_directory": "{full_path_for_logs}"
    }
    ```
    * `nesis_project_full_path`: **Absolute path** to the root directory of your `nesis_project` (the one containing `manage.py`).
    * `csv_directory`: Where generated raw CSV data will be stored.
    * `csv_file_name`: The name of the CSV file that `generate_ml_data.py` will output.
    * `training_data_directory`: Where preprocessed (tokenized, encoded) training data will be saved.
    * `temp_model_save_path`: Where temporary model checkpoints will be saved during training.
    * `model_save_path`: The full path and desired name for saving the trained NLU model.
    * `model_results_directory`: Where model evaluation results will be stored.
    * `model_log_directory`: Where training logs (e.g., TensorBoard logs) will be stored.

    **Important**: Use **forward slashes (`/`)** or **double backslashes (`\\\\`)** for paths, especially on Windows, to avoid escape sequence issues.

---

## 🚀 Usage

Once setup is complete, you can run the different stages of the ML pipeline. Ensure your Conda environment `ml_ops_env` is always activated before running any scripts.

### 1.Data Generation

This script connects to your `nesis_project`'s database and generates a CSV file containing training data for your NLU model. It combines database-derived examples with synthetically generated theme-focused queries.

To run:

```bash
python generate_ml_data.py
```
- **Output**: A CSV file (named what you set `csv_file_name` to in data.json) will be created in the directory specified by `csv_directory`.

### 2. Data Preprocessing 

This script reads the generated CSV data, preprocesses it (e.g., tokenizes text using a Hugging Face tokenizer), and prepares it for model training

To run:

```bash
python data_preprocessing.py
```

- **Input**: Reads the CSV file from `csv_directory/csv_file_name`
- **Output** Saves processed data (e.g., tokenized IDs, attention masks) in a format suitable for PyTorch (e.g., as .pt or .pkl files) to the `training_data_directory`.
  - `train_data_1.pt`
  - `val_data_1.pt`
  - `test_data_1.pt`


### 3. Model Training


```bash
python train_nlu_model.py
```
Trains the NLU model on the preprocessed data.

**Output**: Trained model and evaluation metrics

## ⏰ Automated Training (Production Setup)
For production environments, you can set up automated daily model training using the included shell script and cron job.

### 1. Shell Script Setup
The project includes run_model_training.sh which safely executes the complete pipeline:

    1. Data Generation → Data Preprocessing → Model Training
    2. Sequential Execution: Each step waits for the previous one to complete
    3. Error Handling: Pipeline stops if any step fails
    4. Lock File Protection: Prevents overlapping runs
    5. Detailed Logging: Tracks progress and errors
    6. Safe Deployment: Uses temporary directories and atomic model updates
### 2. Setting Up the Cron Job
To run the training pipeline automatically every day at 2:00 AM:

```bash

# Make the script executable
chmod +x /path/to/ml_ops_project/run_model_training.sh

# Test the script manually first
/path/to/ml_ops_project/run_model_training.sh

# Edit your crontab
crontab -e

# Add this line to run daily at 2:00 AM
0 2 * * * /path/to/ml_ops_project/run_model_training.sh
```

### 3. Monitoring Automated Runs
Logs: Check /path/to/ml_ops_project/logs/model_training_YYYY-MM-DD_HH-MM-SS.log
Lock File: /path/to/ml_ops_project/.model_training.lock (should not exist when idle)
Status Check: ps aux | grep run_model_training.sh
### 4. Cron Job Examples
```bash
# Daily at 2:00 AM (recommended for low system usage)
0 2 * * * /path/to/ml_ops_project/run_model_training.sh

# Every 12 hours (2:00 AM and 2:00 PM)
0 2,14 * * * /path/to/ml_ops_project/run_model_training.sh

# Weekly on Sundays at 3:00 AM
0 3 * * 0 /path/to/ml_ops_project/run_model_training.sh

# Monthly on the 1st at 1:00 AM
0 1 1 * * /path/to/ml_ops_project/run_model_training.sh
```
### 5. Production Safety Features
The automated pipeline includes:

Overlap Prevention: Won't start if previous run is still active
Timeout Protection: Automatically terminates after 2 hours
Automatic Cleanup: Rotates logs and removes temporary files
Error Recovery: Detailed error reporting and graceful failure handling
Lock File Management: Prevents resource conflicts


## 📊 Model Performance

The NLU model typically achieves:
- Token Classification Accuracy: ~90%
- Entity Recognition F1-Score: ~85%
- Query Intent Accuracy: ~92%

## 🔍 Troubleshooting

Common issues and their solutions:

1. **CUDA Out of Memory**
   - Reduce batch size in `train_nlu_model.py`
   - Use CPU-only training for smaller datasets

2. **Missing Dependencies**
   if any find and install

3. **Data.json Configuration**
   - Ensure all paths use forward slashes (/) or double backslashes (\\\\)
   - Verify all directories exist and are accessible


## ✉️ Contact

- **Maintainer**: Billy Ashmall
- **Email**: [billy.ashmall@nasa.gov](mailto:billy.ashmall@nasa.gov)
- **GitHub**: [@billyz313](https://github.com/billyz313)

## 🙏 Acknowledgments

- NASA for supporting this project
- The Hugging Face team for their transformers library
- Contributors and maintainers of the nesis_project

