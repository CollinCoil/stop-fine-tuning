import torch
import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import LogitsProcessor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gc
import os
from huggingface_hub import login

# Login using your Hugging Face token
login("YOUR_TOKEN")

# Define models to test
DECODER_MODELS = [
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.3",
    "google/gemma-7b"
]

# Mapping for labels
LABEL_MAPPING = {0: "Hate speech", 1: "Offensive language", 2: "Neither"}

def load_data(file_url):
    print(f"Loading data from {file_url}")
    data = pd.read_csv(file_url)
    texts = data["tweet"].tolist()
    labels = data["class"].map(LABEL_MAPPING).tolist()
    return texts, labels

def create_prompt(text, label_options):
    return f"""
    You are a social scientist trained to classify tweets into categories based on their content.
    Below is a tweet. Your task is to classify it into one of the following categories: {', '.join(label_options)}.

    Tweet:
    "{text}"
    Output only the category name. The category is:"""

class LabelTokensLogitsProcessor(LogitsProcessor):
    """Restricts generation to only use tokens that appear in the valid labels"""
    def __init__(self, tokenizer, label_options):
        # Tokenize all possible labels
        self.allowed_tokens = set()
        for label in label_options:
            # Add space before tokenization to handle word boundaries
            tokens = tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
            self.allowed_tokens.update(tokens)

        # Add special tokens that might be needed
        self.allowed_tokens.add(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            self.allowed_tokens.add(tokenizer.pad_token_id)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Set log probabilities of all non-allowed tokens to -inf
        scores[:, [i for i in range(scores.shape[1]) if i not in self.allowed_tokens]] = float("-inf")
        return scores

def generate_response(model, tokenizer, prompt, label_options):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    # Create logits processor for label tokens
    logits_processor = LabelTokensLogitsProcessor(tokenizer, label_options)

    generation_config = GenerationConfig(
        max_new_tokens=10,
        min_new_tokens=1,
        do_sample=False,
        num_beams=1,  # Use greedy decoding since we're constraining tokens
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
            logits_processor=[logits_processor]
        )

    new_tokens = output_ids[:, inputs["input_ids"].shape[1]:]  # Ignore input prompt tokens
    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    return response

def evaluate_model(model_name, test_texts, test_labels, label_options):
    print(f"\nEvaluating {model_name} for zero-shot classification...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    predictions = []
    raw_outputs = []
    invalid_count = 0

    # Track start time
    start_time = time.time()

    for text in tqdm(test_texts, desc=f"Processing {model_name}"):
        prompt = create_prompt(text, label_options)
        pred = generate_response(model, tokenizer, prompt, label_options)

        raw_outputs.append(pred)  # Save raw output

        # Validate prediction
        if pred in label_options:
            predictions.append(pred)
        else:
            predictions.append("INVALID")  # Mark invalid predictions
            invalid_count += 1

    # Track end time
    end_time = time.time()
    runtime_seconds = end_time - start_time

    # Compute performance metrics
    predicted_labels = [p if p in label_options else "INVALID" for p in predictions]
    accuracy = accuracy_score(test_labels, predicted_labels)
    f1_macro = f1_score(test_labels, predicted_labels, average="macro")
    f1_micro = f1_score(test_labels, predicted_labels, average="micro")
    f1_weighted = f1_score(test_labels, predicted_labels, average="weighted")
    invalid_rate = invalid_count / len(test_labels)

    # Save raw outputs to CSV
    raw_df = pd.DataFrame({
        "Text": test_texts,
        "True_Label": test_labels,
        "Generated_Output": raw_outputs
    })
    raw_df.to_csv(f"Results/{model_name.replace('/', '_')}_raw_outputs.csv", index=False)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1_Macro": f1_macro,
        "F1_Micro": f1_micro,
        "f1_Weighted": f1_weighted,
        "Invalid_Output_Rate": invalid_rate,
        "Runtime (seconds)": runtime_seconds
    }

def run_experiments():
    texts, labels = load_data("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/refs/heads/master/data/labeled_data.csv")

    # Split the data into training, validation, and testing sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.2, random_state=2025)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=2025)

    label_options = list(LABEL_MAPPING.values())

    results = []
    for model_name in DECODER_MODELS:
        try:
            metrics = evaluate_model(model_name, test_texts, test_labels, label_options)
            results.append(metrics)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv("Results/zero_shot_classification_results_hate.csv", index=False)

if __name__ == "__main__":
    run_experiments()
