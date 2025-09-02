import json
import sys
import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast, TrainingArguments, Trainer
from torch.utils.data import Dataset
import os
import shutil
from datetime import datetime
from pathlib import Path
import time

# --- Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(script_dir, 'data.json')

model_save_path = None
temp_model_save_path = None
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
        temp_model_save_path = data.get('temp_model_save_path')

    if not training_data_directory:
        raise ValueError("training_data_directory not found in data.json")
    if not model_save_path:
        raise ValueError("model_save_path not found in data.json")

    # If temp_model_save_path not specified, create one in same directory
    if not temp_model_save_path:
        model_save_dir = Path(model_save_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_model_save_path = str(model_save_dir / f"temp_model_{timestamp}")

    print(f"Loaded training_data_directory: {training_data_directory}")
    print(f"Production model path: {model_save_path}")
    print(f"Temporary model path: {temp_model_save_path}")

except FileNotFoundError:
    print(f"Error: data.json not found at {DATA_JSON_PATH}. Please create it with required paths.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: data.json at {DATA_JSON_PATH} is not a valid JSON file.")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)

TOKENIZER_NAME = 'distilbert-base-uncased'
NUM_LABELS = 13  # Total number of unique labels (including 'O', B-, I- labels)
MAX_LEN = 128  # Must match MAX_LEN used in data_preprocessing.py


# --- Dataset Class ---
class NLUDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val for key, val in self.encodings[idx].items()}

    def __len__(self):
        return len(self.encodings)


# --- Model Validation Function ---
def validate_model(model_path):
    """Validate that the saved model can be loaded and works correctly"""
    try:
        print(f"Validating model at {model_path}...")

        # Try to load the model and tokenizer
        model = DistilBertForTokenClassification.from_pretrained(model_path)
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

        # Simple test inference
        test_text = "Find water projects in California"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        print("✅ Model validation passed")
        return True

    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False


# --- Safe Model Deployment Function ---
def deploy_model_safely():
    """Deploy the trained model with backup and rollback capabilities"""
    try:
        print("\n" + "=" * 50)
        print("DEPLOYING MODEL SAFELY")
        print("=" * 50)

        prod_path = Path(model_save_path)
        temp_path = Path(temp_model_save_path)
        backup_path = prod_path.parent / f"{prod_path.name}_backup"

        # 1. Validate the temporary model
        if not validate_model(str(temp_path)):
            raise Exception("New model failed validation")

        # 2. Create backup of existing model (if it exists)
        if prod_path.exists():
            print(f"📦 Creating backup at {backup_path}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(prod_path, backup_path)
            print("✅ Backup created successfully")

        # 3. Remove old production model
        if prod_path.exists():
            print("🗑️ Removing old production model")
            shutil.rmtree(prod_path)

        # 4. Move new model to production (atomic operation)
        print(f"📦 Moving new model to production: {prod_path}")
        shutil.move(str(temp_path), str(prod_path))

        # 5. Final validation of production model
        if not validate_model(str(prod_path)):
            raise Exception("Production model failed validation after deployment")

        print("🎉 Model deployed successfully!")
        print(f"Production model location: {prod_path}")

        # 6. Clean up old backup (keep only most recent)
        if backup_path.exists():
            print("🧹 Cleaning up old backup")
            shutil.rmtree(backup_path)

        return True

    except Exception as e:
        print(f"❌ Model deployment failed: {e}")

        # Rollback procedure
        try:
            print("🔄 Attempting rollback...")

            # Remove failed production model if it exists
            if prod_path.exists():
                shutil.rmtree(prod_path)

            # Restore from backup if available
            if backup_path.exists():
                shutil.move(str(backup_path), str(prod_path))
                print("✅ Rollback successful - restored previous model")
            else:
                print("⚠️ No backup available for rollback")

            # Clean up temp model
            if temp_path.exists():
                shutil.rmtree(temp_path)

        except Exception as rollback_error:
            print(f"❌ Rollback failed: {rollback_error}")
            print("⚠️ MANUAL INTERVENTION REQUIRED")

        return False


# --- Load Processed Data ---
try:
    train_data = torch.load(os.path.join(training_data_directory, 'train_data_1.pt'), weights_only=True)
    val_data = torch.load(os.path.join(training_data_directory, 'val_data_1.pt'), weights_only=True)
    test_data = torch.load(os.path.join(training_data_directory, 'test_data_1.pt'), weights_only=True)
    print("Processed data loaded successfully.")
except FileNotFoundError:
    print("Error: Processed data files (train_data.pt, val_data.pt, test_data.pt) not found.")
    print("Please run data_preprocessing.py first.")
    exit()

train_dataset = NLUDataset(train_data)
val_dataset = NLUDataset(val_data)
test_dataset = NLUDataset(test_data)

# --- Initialize Model and Tokenizer ---
print("Initializing DistilBERT model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_NAME)
model = DistilBertForTokenClassification.from_pretrained(TOKENIZER_NAME, num_labels=NUM_LABELS)

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=model_results_directory,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=model_log_directory,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none"
)

# --- Initialize Trainer ---
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# --- Train the Model ---
print("Starting model training...")
trainer.train()
print("Training complete.")

# --- Save to Temporary Location First ---
print(f"\nSaving model to temporary location: {temp_model_save_path}")
os.makedirs(temp_model_save_path, exist_ok=True)
model.save_pretrained(temp_model_save_path)
tokenizer.save_pretrained(temp_model_save_path)
print(f"Model saved to temporary location: {temp_model_save_path}")

# --- Optional: Run final evaluation on the test set ---
print("\nRunning final evaluation on the test set...")
metrics = trainer.evaluate(test_dataset)
print(f"Test Set Metrics: {metrics}")

# --- Deploy Model Safely ---
if deploy_model_safely():
    print("\n🎉 TRAINING AND DEPLOYMENT COMPLETED SUCCESSFULLY!")

    # Clean up temporary model directory
    temp_path = Path(temp_model_save_path)
    if temp_path.exists():
        shutil.rmtree(temp_path)
        print("🧹 Cleaned up temporary model files")
else:
    print("\n❌ DEPLOYMENT FAILED - Check logs above for details")
    print(f"⚠️ Temporary model remains at: {temp_model_save_path}")
    sys.exit(1)

print(f"\nFinal model location: {model_save_path}")
print("=" * 50)