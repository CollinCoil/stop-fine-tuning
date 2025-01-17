import json
from sklearn.model_selection import train_test_split
import pandas as pd

def stratified_split(json_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits a JSON dataset into training, validation, and test sets with stratification by label.

    :param json_file: Path to the input JSON file.
    :param train_file: Path to save the training dataset.
    :param val_file: Path to save the validation dataset.
    :param test_file: Path to save the test dataset.
    :param train_ratio: Proportion of the data to include in the training set.
    :param val_ratio: Proportion of the data to include in the validation set.
    :param test_ratio: Proportion of the data to include in the test set.
    """
    # Load the JSON file
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Organize data by label
    labels = [doc.get("document_label", "Unlabeled") for doc in data]

    # Split the data into training and temp (validation + test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels, test_size=(val_ratio + test_ratio), stratify=labels, random_state=42
    )

    # Calculate validation and test split ratio within temp
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)

    # Split temp data into validation and test sets
    val_data, test_data = train_test_split(
        temp_data, test_size=(1 - val_ratio_adjusted), stratify=temp_labels, random_state=42
    )

    # Save the splits into separate JSON files
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=4)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    # Count document chunks by label in each split
    def count_labels(data):
        label_counts = {}
        for doc in data:
            label = doc.get("document_label", "Unlabeled")
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts

    train_label_counts = count_labels(train_data)
    val_label_counts = count_labels(val_data)
    test_label_counts = count_labels(test_data)

    # Create a DataFrame to track label percentages
    total_counts = {label: train_label_counts.get(label, 0) + val_label_counts.get(label, 0) + test_label_counts.get(label, 0) for label in set(train_label_counts) | set(val_label_counts) | set(test_label_counts)}
    data_frame = pd.DataFrame({
        "Label": list(total_counts.keys()),
        "Total": list(total_counts.values()),
        "Train %": [train_label_counts.get(label, 0) / total_counts[label] * 100 for label in total_counts],
        "Validation %": [val_label_counts.get(label, 0) / total_counts[label] * 100 for label in total_counts],
        "Test %": [test_label_counts.get(label, 0) / total_counts[label] * 100 for label in total_counts]
    })

    # Print data split counts and label percentages
    print("Data Split Counts")
    print("=================")
    print(f"Training set: {len(train_data)} documents")
    print(f"Validation set: {len(val_data)} documents")
    print(f"Test set: {len(test_data)} documents")
    print("\nLabel Percentages by Split")
    print(data_frame.to_string(index=False))


# File paths
input_json = r"Data\labeled_document_chunks.json"
train_output = r"Data\Finetuning\train_data.json"
val_output = r"Data\Validation\val_data.json"
test_output = r"Data\Evaluation\test_data.json"

# Perform the split
stratified_split(input_json, train_output, val_output, test_output)
