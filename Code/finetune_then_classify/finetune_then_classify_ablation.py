import time
import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import gc
from peft import get_peft_model, LoraConfig, TaskType, IA3Config

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

FINE_TUNING_STRATEGIES = ["lora", "ia3", "full_fine_tuning", "head_only"]  

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Remove the batch dimension added by tokenizer
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        
        return item

def load_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item["text"] for item in data]
    labels = [item["document_label"] for item in data]
    return texts, labels

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "Accuracy": accuracy_score(labels, preds),
        "F1_Micro": f1_score(labels, preds, average="micro"),
        "F1_Macro": f1_score(labels, preds, average="macro"),
        "F1_Weighted": f1_score(labels, preds, average="weighted"),
    }

def fine_tune_and_evaluate(strategy, model_name, train_texts, train_labels, val_texts, val_labels, 
                          test_texts, test_labels):
    start_time = time.time()
    print(f"\nFine-tuning {model_name} with {strategy}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_labels = len(set(train_labels))
    
    # Create datasets with on-the-fly tokenization
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        trust_remote_code=True
    ).to(device)

    if strategy == "head_only":
        for param in model.base_model.parameters():
            param.requires_grad = False
    elif strategy == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif strategy == "ia3":
        peft_config = IA3Config(
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

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
        metric_for_best_model="Accuracy",
        logging_dir=f"logs/{strategy}_{model_name}",
        logging_steps=10,
        seed=2024,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        weight_decay=0.01,
    )

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
    
    print("Loading datasets...")
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
            for percent in range(5, 100, 5):
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                print(f"Processing {model_name} with {strategy} using {percent}% of training data")
                
                # Sample the percentage of training data
                sampled_train_texts, _, sampled_train_labels, _ = train_test_split(
                    train_texts, train_labels, train_size=percent / 100.0, 
                    stratify=train_labels, random_state=2025
                )
                
                try:
                    metrics = fine_tune_and_evaluate(
                        strategy, model_name, 
                        sampled_train_texts, sampled_train_labels,
                        val_texts, val_labels,
                        test_texts, test_labels
                    )
                    metrics.update({
                        "Model": model_name, 
                        "Strategy": strategy, 
                        "PercentTrainingData": percent
                    })
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