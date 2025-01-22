import time
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import gc

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

FINE_TUNING_STRATEGIES = ["full_fine_tuning", "LoRA", "head_only"]

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], device=self.device) 
                for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], device=self.device)
        return item

def load_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["document_label"] for item in data]
    return texts, labels

def preprocess_data(tokenizer, texts, labels, max_length=256, batch_size=1000):
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    all_encodings = {}
    
    for i in tqdm(range(total_batches), desc="Preprocessing data"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        batch_encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        if not all_encodings:
            all_encodings = {k: v.cpu().numpy() for k, v in batch_encodings.items()}
        else:
            for k, v in batch_encodings.items():
                all_encodings[k] = np.vstack([all_encodings[k], v.cpu().numpy()])
        
        torch.cuda.empty_cache()
    
    return TextDataset(all_encodings, labels)

def fine_tune_and_evaluate(strategy, model_name, train_dataset, val_dataset, test_dataset, train_labels):
    start_time = time.time()
    print(f"\nFine-tuning {model_name} with {strategy}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(set(train_labels))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        trust_remote_code=True
    ).to(device)

    if strategy == "LoRA":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, lora_config)
    elif strategy == "head_only":
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=f"outputs/{strategy}_{model_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=256,
        num_train_epochs=10,
        learning_rate=2e-5,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=f"logs/{strategy}_{model_name}",
        logging_steps=10,
        seed=2024,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
    )

    def compute_metrics(p):
        with torch.no_grad():
            preds = np.argmax(p.predictions, axis=1)
            return {
                "Accuracy": accuracy_score(p.label_ids, preds),
                "F1_Micro": f1_score(p.label_ids, preds, average="micro"),
                "F1_Macro": f1_score(p.label_ids, preds, average="macro"),
                "F1_Weighted": f1_score(p.label_ids, preds, average="weighted"),
            }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()
    training_time = time.time() - start_time
    
    print("Evaluating model...")
    start_eval_time = time.time()
    metrics = trainer.evaluate(test_dataset)
    evaluation_time = time.time() - start_eval_time

    del model
    torch.cuda.empty_cache()
    gc.collect()

    metrics.update({
        "TrainingTime": training_time,
        "EvaluationTime": evaluation_time
    })
    return metrics

def run_experiments():
    os.makedirs("Results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    final_results = []
    total_experiments = len(FINE_TUNING_STRATEGIES) * len(MODELS) * 19  # 19 percentages (5% to 95%)
    
    print("Loading and preprocessing datasets...")
    train_texts, train_labels = load_data(r"Data\Finetuning\train_data.json")
    val_texts, val_labels = load_data(r"Data\Validation\val_data.json")
    test_texts, test_labels = load_data(r"Data\Evaluation\test_data.json")
    
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)
    test_labels = label_encoder.transform(test_labels)
    
    experiment_count = 0
    for strategy in FINE_TUNING_STRATEGIES:
        for model_name in MODELS:
            for percent in range(5, 100, 5):  # Loop over 5%-95%
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                print(f"Processing {model_name} with {strategy} using {percent}% of training data")
                
                # Sample the percentage of training data
                sample_size = int(len(train_texts) * (percent / 100))
                sampled_indices = np.random.choice(len(train_texts), sample_size, replace=False)
                sampled_train_texts = [train_texts[i] for i in sampled_indices]
                sampled_train_labels = [train_labels[i] for i in sampled_indices]
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                train_dataset = preprocess_data(tokenizer, sampled_train_texts, sampled_train_labels)
                val_dataset = preprocess_data(tokenizer, val_texts, val_labels)
                test_dataset = preprocess_data(tokenizer, test_texts, test_labels)
                
                try:
                    metrics = fine_tune_and_evaluate(
                        strategy, model_name, train_dataset, val_dataset, test_dataset, sampled_train_labels
                    )
                    metrics.update({"Model": model_name, "Strategy": strategy, "PercentTrainingData": percent})
                    final_results.append(metrics)
                    
                    pd.DataFrame(final_results).to_csv(
                        "Results/fine_tuning_classification_results_ablation_intermediate.csv",
                        index=False
                    )
                except Exception as e:
                    print(f"Error processing {model_name} with {strategy}: {str(e)}")
                    continue
                
                torch.cuda.empty_cache()
                gc.collect()
    
    return final_results

if __name__ == "__main__":
    final_results = run_experiments()
    results_df = pd.DataFrame(final_results)
    results_df.to_csv("Results/fine_tuning_classification_results.csv", index=False)
    print("\nExperiment complete. Results saved to 'Results/fine_tuning_classification_ablation_results.csv'")