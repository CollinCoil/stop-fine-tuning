import time
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Define models and fine-tuning strategies
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

FINE_TUNING_STRATEGIES = ["full_fine_tuning", "LoRA", "head_only"]

# Load datasets
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["document_label"] for item in data]
    return texts, labels

train_texts, train_labels = load_data(r"Data\Finetuning\train_data.json")
val_texts, val_labels = load_data(r"Data\Validation\val_data.json")
test_texts, test_labels = load_data(r"Data\Evaluation\test_data.json")

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Dataset preprocessing
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def preprocess_data(tokenizer, texts, labels, max_length=256):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return TextDataset(encodings, labels)

# Main fine-tuning and evaluation function
def fine_tune_and_evaluate(strategy, model_name, train_dataset, val_dataset, test_dataset):
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(set(train_labels))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Apply fine-tuning strategy
    if strategy == "LoRA":
        lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none", target_modules=["query", "value"])
        model = get_peft_model(model, lora_config)
    elif strategy == "head_only":
        # Freeze all model parameters except the classification head
        for param in model.base_model.parameters():
            param.requires_grad = False

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"outputs/{strategy}_{model_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=128,  
        per_device_eval_batch_size=128,  
        num_train_epochs=5,  
        learning_rate=2e-5,  # Set learning rate
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=f"logs/{strategy}_{model_name}",
        logging_steps=10,
        seed=2024
    )

    # Trainer setup
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            "Accuracy": accuracy_score(p.label_ids, preds),
            "F1_Micro": f1_score(p.label_ids, preds, average="micro"),
            "F1_Macro": f1_score(p.label_ids, preds, average="macro"),
            "F1_Weighted": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="weighted"),
            "ROC_AUC": roc_auc_score(p.label_ids, p.predictions, multi_class="ovr"),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    training_time = time.time() - start_time
    start_eval_time = time.time()
    metrics = trainer.evaluate(test_dataset)
    evaluation_time = time.time() - start_eval_time

    # Add timing to metrics
    metrics.update({"TrainingTime": training_time, "EvaluationTime": evaluation_time})
    return metrics

# Main experiment loop
final_results = []

for strategy in FINE_TUNING_STRATEGIES:
    for model_name in MODELS:
        print(f"Fine-tuning {model_name} with {strategy}...")
        # Preprocess datasets
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = preprocess_data(tokenizer, train_texts, train_labels)
        val_dataset = preprocess_data(tokenizer, val_texts, val_labels)
        test_dataset = preprocess_data(tokenizer, test_texts, test_labels)

        # Fine-tune and evaluate
        metrics = fine_tune_and_evaluate(strategy, model_name, train_dataset, val_dataset, test_dataset)
        metrics.update({"Model": model_name, "Strategy": strategy})
        final_results.append(metrics)

# Save results to CSV
results_df = pd.DataFrame(final_results)
results_df.to_csv(r"Results\fine_tuning_classification_results.csv", index=False)
print("Experiment complete. Results saved to 'fine_tuning_classification_results.csv'.")
