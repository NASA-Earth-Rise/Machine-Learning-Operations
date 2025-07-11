import json
import sys

import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast, TrainingArguments, Trainer
from torch.utils.data import Dataset
import os

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(script_dir, 'data.json')

model_save_path = None
training_data_directory = None
model_results_directory = None
model_log_directory = None
try:
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)
        training_data_directory = data.get('training_data_directory')
        model_results_directory = data.get('model_results_directory')
        model_log_directory = data.get('model_log_directory')
        model_save_path = data.get('model_save_path')
    if not training_data_directory:
        raise ValueError("training_data_directory not found in data.json")
    print(f"Loaded training_data_directory: {training_data_directory}")
except FileNotFoundError:
    print(f"Error: data.json not found at {DATA_JSON_PATH}. Please create it with 'training_data_directory'.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: data.json at {DATA_JSON_PATH} is not a valid JSON file.")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

TOKENIZER_NAME = 'distilbert-base-uncased'
MODEL_SAVE_PATH = model_save_path # Directory where the trained model will be saved
NUM_LABELS = 13 # Total number of unique labels (including 'O', B-, I- labels)
MAX_LEN = 128 # Must match MAX_LEN used in data_preprocessing.py

# --- Dataset Class ---
class NLUDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val for key, val in self.encodings[idx].items()}

    def __len__(self):
        return len(self.encodings)

# --- Load Processed Data ---
try:
    train_data = torch.load(os.path.join(training_data_directory,'train_data_1.pt'), weights_only=True)
    val_data = torch.load(os.path.join(training_data_directory,'val_data_1.pt'), weights_only=True)
    test_data = torch.load(os.path.join(training_data_directory,'test_data_1.pt'), weights_only=True)
    print("Processed data loaded successfully.")
except FileNotFoundError:
    print("Error: Processed data files (train_data.pt, val_data.pt, test_data.pt) not found.")
    print("Please run data_preprocessing.py first.")
    exit()

train_dataset = NLUDataset(train_data)
val_dataset = NLUDataset(val_data)
test_dataset = NLUDataset(test_data) # Test dataset is typically only for final evaluation

# --- Initialize Model and Tokenizer ---
print("Initializing DistilBERT model and tokenizer...")
# Load the tokenizer again, ensuring it's the 'fast' version
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_NAME)
# Load the pre-trained model for token classification
model = DistilBertForTokenClassification.from_pretrained(TOKENIZER_NAME, num_labels=NUM_LABELS)

# --- Training Arguments ---
# Adjust these parameters based on your dataset size and computational resources
training_args = TrainingArguments(
    output_dir=model_results_directory,                     # Output directory for checkpoints and logs
    num_train_epochs=3,                         # Number of training epochs
    per_device_train_batch_size=16,             # Batch size per device during training
    per_device_eval_batch_size=16,              # Batch size per device during evaluation
    warmup_steps=500,                           # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,                          # Strength of weight decay
    logging_dir=model_log_directory,                       # Directory for storing logs
    logging_steps=100,                          # Log every n steps
    eval_strategy="epoch",                # Evaluate every epoch
    save_strategy="epoch",                      # Save model checkpoint every epoch
    load_best_model_at_end=True,                # Load the best model at the end of training
    metric_for_best_model="eval_loss",          # Metric to use to compare models
    report_to="none"                            # Disable integrations like W&B if not needed
)

# --- Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, # Use validation set for evaluation during training
    tokenizer=tokenizer,
)

# --- Train the Model ---
print("Starting model training...")
trainer.train()
print("Training complete.")

# --- Save the fine-tuned model and tokenizer ---
# Ensure the directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Fine-tuned model and tokenizer saved to {MODEL_SAVE_PATH}")

# --- Optional: Run final evaluation on the test set ---
print("\nRunning final evaluation on the test set...")
metrics = trainer.evaluate(test_dataset)
print(f"Test Set Metrics: {metrics}")