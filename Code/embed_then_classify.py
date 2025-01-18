import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# Define models and hyperparameters (keeping original configurations)
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
    "xlnet/xlnet-large-cased"
]

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression,
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "MLP": MLPClassifier,
    "AdaBoost": AdaBoostClassifier
}

PARAMETERS = {
    "LogisticRegression": {"C": [0.1, 1, 10], "max_iter": [250], "solver": ["saga"]},
    "KNN": {"n_neighbors": [3, 5, 7], "p": [1, 2]},
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "sigmoid"]},
    "MLP": {"hidden_layer_sizes": [(100, 100), (100, 50), (100, 50, 25)], "alpha": [0.0001, 0.001]},
    "AdaBoost": {"estimator": [SVC(probability=True)], "n_estimators": [50, 100, 200], "learning_rate": [0.1, 1]}
}

# Custom dataset for batch processing
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

# Load datasets
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["document_label"] for item in data]
    return texts, labels

# Optimized embedding generation with GPU support and batching
def generate_embeddings(model_name, texts, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating embeddings for {model_name}"):
            # Move input tensors to device
            inputs = {k: v.squeeze(1).to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
    
    return np.vstack(embeddings)

# Train and evaluate classifiers
def train_and_evaluate(embeddings_train, labels_train, embeddings_val, labels_val, embeddings_test, labels_test):
    results = []
    
    for clf_name, clf_class in CLASSIFIERS.items():
        print(f"\nTraining {clf_name}")
        param_grid = PARAMETERS[clf_name]
        best_model = None
        best_score = -1
        
        # Grid search on validation set
        for params in tqdm(ParameterGrid(param_grid), desc="Grid search"):
            clf = clf_class(**params)
            clf.fit(embeddings_train, labels_train)
            val_preds = clf.predict(embeddings_val)
            val_probs = clf.predict_proba(embeddings_val) if hasattr(clf, "predict_proba") else None
            score = roc_auc_score(labels_val, val_probs, multi_class="ovr", average="weighted") if val_probs is not None else accuracy_score(labels_val, val_preds)
            
            if score > best_score:
                best_score = score
                best_model = clf
        
        # Test the best model
        test_preds = best_model.predict(embeddings_test)
        test_probs = best_model.predict_proba(embeddings_test) if hasattr(best_model, "predict_proba") else None
        
        metrics = {
            "Classifier": clf_name,
            "BestParams": best_model.get_params(),
            "Accuracy": accuracy_score(labels_test, test_preds),
            "F1_Micro": f1_score(labels_test, test_preds, average="micro"),
            "F1_Macro": f1_score(labels_test, test_preds, average="macro"),
            "F1_Weighted": f1_score(labels_test, test_preds, average="weighted"),
            "ROC_AUC": roc_auc_score(labels_test, test_probs, multi_class="ovr") if test_probs is not None else None
        }
        results.append(metrics)
    
    return results

# Main experiment loop
def run_experiments(models=MODELS):
    os.makedirs("Results", exist_ok=True)
    final_results = []
    
    # Load all datasets
    print("Loading datasets...")
    train_texts, train_labels = load_data(r"Data\Finetuning\train_data.json")
    val_texts, val_labels = load_data(r"Data\Validation\val_data.json")
    test_texts, test_labels = load_data(r"Data\Evaluation\test_data.json")
    
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
        
        total_time = time.time() - start_time
        
        # Add results and save intermediate results
        for result in model_results:
            result.update({
                "EmbeddingModel": model_name,
                "EmbeddingTime": embedding_time,
                "TotalTime": total_time
            })
            final_results.append(result)
            
        # Save intermediate results after each model
        pd.DataFrame(final_results).to_csv(
            "Results/embedding_classification_results_intermediate.csv",
            index=False
        )
    
    return final_results

if __name__ == "__main__":
    # Run experiments
    final_results = run_experiments()
    
    # Save final results
    results_df = pd.DataFrame(final_results)
    results_df.to_csv("Results/embedding_classification_results.csv", index=False)
    print("\nExperiment complete. Results saved to 'Results/embedding_classification_results.csv'")