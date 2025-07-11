# `Machine Learning Operations`

[![Python: 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/billyz313/Machine-Learning-Operations/blob/main/LICENSE)
[![IMapApps: Development](https://img.shields.io/badge/IMapApps-Development-green)](https://imapapps.com)

## 🚀 Purpose

This project (`ml_ops_project`) is a standalone collection of Python scripts designed to handle the **Machine Learning Operations (MLOps)** lifecycle for the `nesis_project` (EAX) web application. It focuses on tasks like **data generation, preprocessing, and model training** for a Natural Language Understanding (NLU) model, intended to power a search or query understanding feature within `nesis_project`.

By keeping this separate from the main `nesis_project`, we achieve:
* **Separation of Concerns**: Cleaner distinction between web application logic and ML pipeline.
* **Independent Dependencies**: Allows specific ML libraries and their versions without affecting the web app.
* **Flexible Execution**: Enables running resource-intensive ML tasks (like model training) on dedicated environments.

---

## 🛠️ Setup

Follow these steps to set up your development environment for the `ml_ops_project`.

### 1. Conda Environment Setup

It's crucial to set up a dedicated Conda environment to manage project dependencies.

1.  **Create the Conda Environment**:
    Open your terminal or Anaconda Prompt, navigate to the root directory of your `ml_ops_project`, and run:

    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate the Environment**:
    Once created, activate your new environment:

    ```bash
    conda activate ml_ops_env
    ```
    You will need to activate this environment every time you work on the `ml_ops_project`.

### 2. `data.json` Configuration

This project relies on a `data.json` file to configure paths to your `nesis_project` database and other critical directories.

1.  **Create `data.json`**:
    In the root directory of your `ml_ops_project`, create a file named `data.json`.

2.  **Populate `data.json`**:
    Copy the following template into your `data.json` file and **update all placeholder values** (`{...}`) with the actual absolute paths for your setup.

    ```json
    {
      "nesis_project_full_path": "{full_path_to_local_nesis_project}",
      "csv_directory": "{full_path_to_directory}",
      "csv_file_name": "{name_for_training_csv}.csv",
      "training_data_directory": "{full_path_to_training_data_directory}",
      "model_save_path": "{full_path_plus_name_you_want_for_model}",
      "model_results_directory": "{full_path_for_results}",
      "model_log_directory": "{full_path_for_logs}"
    }
    ```
    * `nesis_project_full_path`: **Absolute path** to the root directory of your `nesis_project` (the one containing `manage.py`).
    * `csv_directory`: Where generated raw CSV data will be stored.
    * `csv_file_name`: The name of the CSV file that `generate_ml_data.py` will output.
    * `training_data_directory`: Where preprocessed (tokenized, encoded) training data will be saved.
    * `model_save_path`: The full path and desired name for saving the trained NLU model.
    * `model_results_directory`: Where model evaluation results will be stored.
    * `model_log_directory`: Where training logs (e.g., TensorBoard logs) will be stored.

    **Important**: Use **forward slashes (`/`)** or **double backslashes (`\\\\`)** for paths, especially on Windows, to avoid escape sequence issues.

---

## 🚀 Usage

Once setup is complete, you can run the different stages of the ML pipeline. Ensure your Conda environment `ml_ops_env` is always activated before running any scripts.

### 1. Generate Training Data `generate_ml_data.py`

This script connects to your `nesis_project`'s database and generates a CSV file containing training data for your NLU model. It combines database-derived examples with synthetically generated theme-focused queries.

To run:

```bash
python generate_ml_data.py
```
- **Output**: A CSV file (named what you set `csv_file_name` to in data.json) will be created in the directory specified by `csv_directory`.

### 2. Preprocess Data `data_preprocessing.py`
This script reads the generated CSV data, preprocesses it (e.g., tokenizes text using a Hugging Face tokenizer), and prepares it for model training

To run:

```bash
python data_preprocessing.py
```

- **Input**: Reads the CSV file from `csv_directory/csv_file_name`
- **Output** Saves processed data (e.g., tokenized IDs, attention masks) in a format suitable for PyTorch (e.g., as .pt or .pkl files) to the `training_data_directory`.

### 3. Train the Model `train_nlu_model.py`

This script will take the preprocessed data and use it to train your Natural Language Understanding (NLU) model (e.g., a Hugging Face Transformer model for multi-label classification).

To run:

```bash
python train_nlu_model.py
```

- **Input**: Reads preprocessed data from `training_data_directory`.
- **Output** Saves the trained model to `model_save_path`, training logs to `model_log_directory`, and potentially evaluation results to `model_results_directory`.

When your model is trained, it is ready for testing and then deployment within the nesis_project (EAX)

### 📞 Contact

Please feel free to contact me if you have any questions.

### ✍️ Authors

- [Billy Ashmall (NASA/USRA)](https://github.com/billyz313)
