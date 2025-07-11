import json
import sys

import pandas as pd
from transformers import DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
import torch
import os

# --- Configuration ---
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_JSON_PATH = os.path.join(script_dir, 'data.json')

csv_directory = None
csv_file_name = None
training_data_directory = None
try:
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)
        nesis_project_full_path = data.get('nesis_project_full_path')
        csv_directory = data.get('csv_directory')
        csv_file_name = data.get('csv_file_name')
        training_data_directory = data.get('training_data_directory')
    if not nesis_project_full_path:
        raise ValueError("'nesis_project_full_path' not found in data.json")
    print(f"Loaded nesis_project_full_path: {nesis_project_full_path}")
except FileNotFoundError:
    print(f"Error: data.json not found at {DATA_JSON_PATH}. Please create it with 'nesis_project_full_path'.")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: data.json at {DATA_JSON_PATH} is not a valid JSON file.")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)


ANNOTATED_DATA_CSV = os.path.join(csv_directory, csv_file_name)
# Define our configuration constants
TOKENIZER_NAME = 'distilbert-base-uncased'
MAX_LEN = 128  # Maximum sequence length for DistilBERT
TEST_SIZE = 0.1  # 10% for test set
VAL_SIZE = 0.1  # 10% for validation set (from remaining data after test split)
RANDOM_SEED = 42  # For reproducibility

# --- Initialize Tokenizer ---
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_NAME)

# --- Define your entity labels ---
# These should match the columns in your CSV and the entity types your model will extract.
# 'O' is for tokens outside any entity.
# B- for Beginning of an entity, I- for Inside an entity.
ENTITY_LABELS = [
    'O',
    'B-RESULT_TYPE', 'I-RESULT_TYPE',
    'B-THEME', 'I-THEME',  # Corrected to B-THEME / I-THEME as per your project
    'B-COUNTRY_NAME', 'I-COUNTRY_NAME',
    'B-STATE_NAME', 'I-STATE_NAME',
    'B-SEARCH_TERM', 'I-SEARCH_TERM'
]
# Create a mapping from label string to ID and vice versa
label_to_id = {label: i for i, label in enumerate(ENTITY_LABELS)}
id_to_label = {i: label for i, label in enumerate(ENTITY_LABELS)}


def tokenize_and_align_labels(sentence, annotations, tokenizer, max_len):
    """
    Converts a sentence and its annotations into model-ready format.

    Args:
        sentence (str): The input query text
        annotations (dict): Dictionary containing entity annotations
        tokenizer: The DistilBERT tokenizer
        max_len (int): Maximum sequence length

    Returns:
        dict: Contains:
            - input_ids: Numeric tokens representing the text
            - attention_mask: Marks which tokens are real vs padding
            - labels: Numeric labels for each token
    """

    # Tokenize the sentence and get character-level mappings
    tokenized_inputs = tokenizer(
        sentence,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_offsets_mapping=True,  # Essential for character-level alignment
        return_tensors="pt"
    )

    # Initialize labels with -100 for all tokens (special tokens will remain -100,
    # non-entity tokens will be set to 'O' later)
    labels = [-100] * len(tokenized_inputs['input_ids'][0])

    # Get offset mappings: list of (start_char, end_char) for each token
    offsets = tokenized_inputs.offset_mapping[0].tolist()

    # Iterate over each entity type (e.g., 'THEME', 'RESULT_TYPE')
    for entity_col, entity_value_str in annotations.items():
        if not entity_value_str:
            continue

        # Handle multiple values if applicable (e.g., for RESULT_TYPE)
        # Using a list to ensure consistent iteration even for single values
        entity_values_to_process = entity_value_str.split(', ') if entity_col == 'RESULT_TYPE' else [entity_value_str]

        for entity_value in entity_values_to_process:
            # Find all occurrences of the entity value (case-insensitive) in the sentence
            # This handles cases where the same entity appears multiple times in a query.
            current_sentence_lower = sentence.lower()
            current_entity_lower = entity_value.lower()

            # Collect all start and end character positions for this entity instance
            entity_char_spans = []
            start_idx = 0
            while True:
                found_idx = current_sentence_lower.find(current_entity_lower, start_idx)
                if found_idx == -1:
                    break
                entity_char_spans.append((found_idx, found_idx + len(current_entity_lower)))
                start_idx = found_idx + len(current_entity_lower)  # Continue search after this occurrence

            if not entity_char_spans:
                # Debugging: Uncomment if you want to see which entities are not found
                # print(f"Warning: Entity '{entity_value}' not found as substring in sentence '{sentence}'")
                continue

            # Now, for each found span of the entity, assign B- and I- tags to tokens
            for entity_start_char, entity_end_char in entity_char_spans:

                is_first_token_of_this_entity_instance = True

                for token_idx, (token_start_char, token_end_char) in enumerate(offsets):
                    # Skip special tokens (usually token_start_char == token_end_char == 0 for [CLS], [SEP], [PAD])
                    if token_start_char == 0 and token_end_char == 0:
                        continue  # Keep -100 label for these

                    # Check if the token's character span overlaps with the *current* entity instance's span
                    # A token is considered part of the entity if any part of it is within the entity's span
                    # or if the entity's span is fully within the token's span (less common for short entities)

                    # Overlap condition: (token_start < entity_end) AND (entity_start < token_end)
                    if max(token_start_char, entity_start_char) < min(token_end_char, entity_end_char):
                        # This token overlaps with the entity span
                        if is_first_token_of_this_entity_instance:
                            labels[token_idx] = label_to_id[f'B-{entity_col}']
                            is_first_token_of_this_entity_instance = False
                        else:
                            labels[token_idx] = label_to_id[f'I-{entity_col}']
                    # If we've passed the current entity's span and were previously within it,
                    # reset the flag for the *next* potential entity instance in the sentence.
                    # This is important if an entity occurs multiple times.
                    elif token_start_char >= entity_end_char and not is_first_token_of_this_entity_instance:
                        # We've exited the current entity span, so reset for the next potential entity.
                        # Do not set to 'O' here explicitly, let the final 'O' pass handle it.
                        is_first_token_of_this_entity_instance = True  # Reset for the next time we find B-

    # Finally, set 'O' for any tokens that haven't been labeled as B- or I-
    # and are not special tokens (-100)
    for i, label_id in enumerate(labels):
        # Check if it's a non-special token (offset is not (0,0)) and still has the default -100 label
        if label_id == -100 and not (offsets[i][0] == 0 and offsets[i][1] == 0):
            labels[i] = label_to_id['O']
        # Special tokens should remain -100
        elif (offsets[i][0] == 0 and offsets[i][1] == 0) and labels[i] == -100:
            pass  # Keep -100 for special tokens

    # --- Debugging prints for preprocessing phase ---
    print(f"\nProcessing Query: '{sentence}'")
    # Convert token IDs back to tokens for readability
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][0].tolist())
    print(f"Tokens: {tokens}")
    print(f"Offset Mappings: {offsets}")

    # Convert label IDs back to names for human readability
    debug_labels = [id_to_label[label_id] if label_id != -100 else '-100' for label_id in labels]
    print(f"Assigned Labels: {debug_labels}")

    tokenized_inputs['labels'] = torch.tensor(labels, dtype=torch.long)

    # Remove offset_mapping before returning as it's not needed by the model directly
    del tokenized_inputs['offset_mapping']

    return tokenized_inputs


def preprocess_data(df):
    """
    Converts the entire dataset into model-ready format.

    Args:
        df (pandas.DataFrame): DataFrame containing queries and their annotations

    Returns:
        list: List of dictionaries containing processed data:
            - input_ids: Token IDs for the text
            - attention_mask: Mask for real vs padding tokens
            - labels: Numeric labels for each token
    """

    processed_data = []

    # Column names for entities (excluding 'Query' and 'Intent')
    entity_cols = [col for col in df.columns if col not in ['Query', 'Intent']]

    for index, row in df.iterrows():
        sentence = row['Query']

        # Collect annotations for this sentence
        annotations = {}
        for col in entity_cols:
            if pd.notna(row[col]) and row[col] != '':  # Check for non-empty annotations
                annotations[col] = str(row[col]).strip()  # Ensure string and remove whitespace

        try:
            encoded_inputs = tokenize_and_align_labels(sentence, annotations, tokenizer, MAX_LEN)
            processed_data.append({
                'input_ids': encoded_inputs['input_ids'].flatten(),
                'attention_mask': encoded_inputs['attention_mask'].flatten(),
                'labels': encoded_inputs['labels'].flatten()
            })
        except Exception as e:
            print(f"Error processing row {index} ('{sentence}'): {e}")
            continue

    return processed_data


if __name__ == "__main__":
    print(f"Loading data from {ANNOTATED_DATA_CSV}...")
    try:
        df = pd.read_csv(ANNOTATED_DATA_CSV)
        # Ensure that multiple RESULT_TYPE values are treated as a single string for parsing.
        # This assumes comma-separated values in the CSV for RESULT_TYPE.
        df['RESULT_TYPE'] = df['RESULT_TYPE'].fillna('').astype(str)
        # Fill NaN values in other entity columns with empty strings
        for col in ENTITY_LABELS[1:]:  # Skip 'O' label
            # Use the original column names from the CSV (e.g., 'THEME', not 'B-THEME')
            # Check if the entity type (e.g., 'THEME') exists as a column in the DataFrame
            # before trying to fill it.
            clean_col_name = col.replace('B-', '').replace('I-', '')
            if clean_col_name in df.columns:
                df[clean_col_name] = df[clean_col_name].fillna('').astype(str)

        print(f"Loaded {len(df)} entries.")
    except FileNotFoundError:
        print(f"Error: {ANNOTATED_DATA_CSV} not found. Please ensure the CSV file is in the correct directory.")
        exit()

    print("Preprocessing data...")
    # It's important to keep original text for debugging/review
    original_queries = df['Query'].tolist()
    processed_inputs = preprocess_data(df)

    if not processed_inputs:
        print("No data was successfully processed. Exiting.")
        exit()

    # Split into training, validation, and test sets
    train_val_inputs, test_inputs = train_test_split(
        processed_inputs, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_inputs, val_inputs = train_test_split(
        train_val_inputs, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_SEED
    )

    print(f"Total processed samples: {len(processed_inputs)}")
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    print(f"Test samples: {len(test_inputs)}")

    # You can now save these processed datasets or pass them to the training script.
    # For example, saving them as torch tensors or pickled files.
    # Example (saving to a file for later use in training script):
    torch.save(train_inputs, os.path.join(training_data_directory, 'train_data_1.pt'))
    torch.save(val_inputs, os.path.join(training_data_directory,'val_data_1.pt'))
    torch.save(test_inputs, os.path.join(training_data_directory,'test_data_1.pt'))
    print("Processed data saved to train_data_1.pt, val_data_1.pt, test_data_1.pt in: " + training_data_directory)
    print(
        "Note: You can also use the 'load_data()' function in 'train.py' to load these processed datasets."
    )

    # Optional: Print a sample to inspect
    print("\n--- Sample Processed Data ---")
    if processed_inputs:
        # Find a sample that contains 'Wildland Fires' or similar multi-word theme for verification
        sample_index = -1
        for i, query_text in enumerate(original_queries):
            if 'wildland fires' in query_text.lower():
                sample_index = i
                break

        if sample_index != -1:
            sample = processed_inputs[sample_index]
            print(f"Original Query (sample {sample_index}): {original_queries[sample_index]}")
            print(f"Input IDs: {sample['input_ids'][:15]}...")
            print(f"Attention Mask: {sample['attention_mask'][:15]}...")
            print(
                f"Labels: {[id_to_label[label.item()] if label.item() != -100 else '-100' for label in sample['labels']][:15]}...")
            print(f"Full Labels Length: {len(sample['labels'])}")
        else:
            print("No query with 'Wildland Fires' found in the sample for detailed print.")
            sample = processed_inputs[0]  # Fallback to first sample
            print(f"Original Query (first): {original_queries[0]}")
            print(f"Input IDs: {sample['input_ids'][:15]}...")
            print(f"Attention Mask: {sample['attention_mask'][:15]}...")
            print(
                f"Labels: {[id_to_label[label.item()] if label.item() != -100 else '-100' for label in sample['labels']][:15]}...")
            print(f"Full Labels Length: {len(sample['labels'])}")