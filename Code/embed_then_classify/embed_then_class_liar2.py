import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
liar_df_train = pd.read_csv("hf://datasets/chengxuphd/liar2/" + splits["train"])
liar_df_test = pd.read_csv("hf://datasets/chengxuphd/liar2/" + splits["test"])
liar_df_valid = pd.read_csv("hf://datasets/chengxuphd/liar2/" + splits["validation"])

MODELS = [
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "answerdotai/ModernBERT-base",
    "answerdotai/ModernBERT-large",
    "ibm-granite/granite-embedding-30m-english",
    "ibm-granite/granite-embedding-125m-english",
    "microsoft/deberta-v3-xsmall",
    "microsoft/deberta-v3-small",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "Alibaba-NLP/gte-base-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "xlnet/xlnet-base-cased",
    "xlnet/xlnet-large-cased",
    "intfloat/e5-small-v2",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2"
]

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "MLP": MLPClassifier,
    "XGBoost": XGBClassifier,
}

PARAMETERS = {
    "LogisticRegression": {"C": [0.1, 1, 10], "max_iter": [400], "solver": ["saga"], "n_jobs": [-1]},
    "KNN": {"n_neighbors": [5, 7, 9], "p": [1, 2], "weights": ["uniform", "distance"], "n_jobs": [-1]},
    "SVM": {"C": [1, 10, 100], "kernel": ["rbf"], "cache_size": [1800]},
    "MLP": {"hidden_layer_sizes": [(400, 200, 100, 50), (200, 150, 100, 50)], "alpha": [0.0001, 0.001]},
    "XGBoost": {
        "n_estimators": [100, 150],
        "learning_rate": [0.25, 0.5],
        "max_depth": [5, 10],
        "sampling_method": ["gradient_based"],
        "subsample": [0.33],
        "device": ["cuda"],
        "eval_metric": ["logloss"]   # Default metric for classification
    },
}

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

# Mapping from integer labels to textual labels
label_mapping = {
    0: "Pants on fire",
    1: "False",
    2: "Barely true",
    3: "Half true",
    4: "Mostly true",
    5: "True"
}

def load_data(data):
    texts = data["statement"]
    labels = data["label"].map(label_mapping)  # Map integers to strings
    return texts, labels

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Mask padding tokens in the last hidden states
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    # Compute the average pooling
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

def generate_embeddings(model_name, texts, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating embeddings for {model_name}"):
            # Move input tensors to device
            inputs = {k: v.squeeze(1).to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            # Apply average pooling
            batch_embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy()
            embeddings.append(batch_embeddings)

            # Clear GPU memory
            torch.cuda.empty_cache()

    return np.vstack(embeddings)

def train_and_evaluate(embeddings_train, labels_train, embeddings_val, labels_val, embeddings_test, labels_test):
    # Encode string labels as integers
    label_encoder = LabelEncoder()
    labels_train = label_encoder.fit_transform(labels_train)
    labels_val = label_encoder.transform(labels_val)
    labels_test = label_encoder.transform(labels_test)

    results = []

    for clf_name, clf_class in CLASSIFIERS.items():
        print(f"\nTraining {clf_name}")
        param_grid = PARAMETERS[clf_name]
        best_model = None
        best_score = -1
        start_fine_tune = time.time()  # Start time for this classifier's training

        # Grid search on validation set
        for params in tqdm(ParameterGrid(param_grid), desc="Grid search"):
            clf = clf_class(**params)
            clf.fit(embeddings_train, labels_train)
            val_preds = clf.predict(embeddings_val)
            score = accuracy_score(labels_val, val_preds)

            if score > best_score:
                best_score = score
                best_model = clf

        # Test the best model
        test_preds = best_model.predict(embeddings_test)
        test_probs = best_model.predict_proba(embeddings_test) if hasattr(best_model, "predict_proba") else None

        # Decode predictions back to original string labels
        decoded_preds = label_encoder.inverse_transform(test_preds)
        decoded_labels_test = label_encoder.inverse_transform(labels_test)

        metrics = {
            "Classifier": clf_name,
            "BestParams": best_model.get_params(),
            "Accuracy": accuracy_score(decoded_labels_test, decoded_preds),
            "F1_Micro": f1_score(decoded_labels_test, decoded_preds, average="micro"),
            "F1_Macro": f1_score(decoded_labels_test, decoded_preds, average="macro"),
            "F1_Weighted": f1_score(decoded_labels_test, decoded_preds, average="weighted"),
            "FineTuningTime": time.time() - start_fine_tune  # Fine-tuning time for this classifier
        }
        results.append(metrics)

    return results

def run_experiments(models=MODELS):
    os.makedirs("Results", exist_ok=True)
    final_results = []

    # Load all datasets
    print("Loading datasets...")
    train_texts, train_labels = load_data(liar_df_train)
    val_texts, val_labels = load_data(liar_df_valid)
    test_texts, test_labels = load_data(liar_df_test)

    for model_name in tqdm(models, desc="Processing models"):
        start_time = time.time()
        print(f"\nProcessing model: {model_name}")

        # Generate embeddings
        train_embeddings = generate_embeddings(model_name, train_texts)
        val_embeddings = generate_embeddings(model_name, val_texts)
        test_embeddings = generate_embeddings(model_name, test_texts)

        embedding_time = time.time() - start_time

        # Train and evaluate classifiers
        model_results = train_and_evaluate(
            train_embeddings, train_labels,
            val_embeddings, val_labels,
            test_embeddings, test_labels
        )

        # Add results and save intermediate results
        for result in model_results:
            result.update({
                "EmbeddingModel": model_name,
                "EmbeddingTime": embedding_time,
                "TotalTime": embedding_time + result["FineTuningTime"]  # Total time for this classifier
            })
            final_results.append(result)

        # Save intermediate results after each model
        pd.DataFrame(final_results).to_csv(
            "Results/embedding_classification_results_intermediate_liar2.csv",
            index=False
        )

    return final_results

if __name__ == "__main__":
    # Run experiments
    final_results = run_experiments()

    # Save final results
    results_df = pd.DataFrame(final_results)
    results_df.to_csv("Results/embedding_classification_results_liar2.csv", index=False)
