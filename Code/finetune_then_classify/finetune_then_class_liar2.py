import time
import json
import torch
import pandas as pd
import numpy as np
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

splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
liar_df_train = pd.read_csv("hf://datasets/chengxuphd/liar2/" + splits["train"])
liar_df_test = pd.read_csv("hf://datasets/chengxuphd/liar2/" + splits["test"])
liar_df_valid = pd.read_csv("hf://datasets/chengxuphd/liar2/" + splits["validation"])

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
    texts = data["statement"].astype(str).tolist()  # Ensure all values are strings and convert to list
    labels = data["label"].map(label_mapping).tolist()  # Convert labels to list
    return texts, labels

def preprocess_data(tokenizer, texts, labels, max_length=128, batch_size=1000):
    total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
    all_encodings = {}

    for i in tqdm(range(total_batches), desc="Preprocessing data"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        batch_encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "Accuracy": accuracy_score(labels, preds),
        "F1_Micro": f1_score(labels, preds, average="micro"),
        "F1_Macro": f1_score(labels, preds, average="macro"),
        "F1_Weighted": f1_score(labels, preds, average="weighted"),
    }

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
    total_experiments = len(FINE_TUNING_STRATEGIES) * len(MODELS)

    print("Loading and preprocessing datasets...")
    train_texts, train_labels = load_data(liar_df_train)
    val_texts, val_labels = load_data(liar_df_valid)
    test_texts, test_labels = load_data(liar_df_test)

    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    val_labels = label_encoder.transform(val_labels)
    test_labels = label_encoder.transform(test_labels)

    experiment_count = 0
    for strategy in FINE_TUNING_STRATEGIES:
        for model_name in MODELS:
            experiment_count += 1
            print(f"\nExperiment {experiment_count}/{total_experiments}")
            print(f"Processing {model_name} with {strategy}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            train_dataset = preprocess_data(tokenizer, train_texts, train_labels)
            val_dataset = preprocess_data(tokenizer, val_texts, val_labels)
            test_dataset = preprocess_data(tokenizer, test_texts, test_labels)

            try:
                metrics = fine_tune_and_evaluate(
                    strategy, model_name, train_dataset, val_dataset, test_dataset, train_labels
                )
                metrics.update({"Model": model_name, "Strategy": strategy})
                final_results.append(metrics)

                pd.DataFrame(final_results).to_csv(
                    "Results/fine_tuning_classification_results_intermediate_liar2.csv",
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
    results_df.to_csv("Results/fine_tuning_classification_results_liar2.csv", index=False)
