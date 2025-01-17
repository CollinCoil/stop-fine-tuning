import os
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

def split_into_chunks(text, min_tokens=120, max_tokens=240):
    """Splits text into chunks between min_tokens and max_tokens, ensuring splits occur at sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = word_tokenize(sentence)
        sentence_token_count = len(sentence_tokens)

        if current_tokens + sentence_token_count > max_tokens:
            if current_tokens >= min_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_token_count

    if min_tokens <= current_tokens <= max_tokens:
        chunks.append(" ".join(current_chunk))

    return chunks

def merge_text_files_to_json(input_directory, output_file):
    """Merges text files into a single JSON file with preprocessing."""
    documents = []
    total_chunks = 0

    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_directory, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            chunks = split_into_chunks(text)
            total_chunks += len(chunks)

            for chunk in chunks:
                documents.append({
                    "document_name": filename,
                    "text": chunk,
                    "document_label": None  # Placeholder for the document label, which will be done manually
                })

    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(documents, json_file, ensure_ascii=False, indent=4)

    print(f"Total text chunks saved: {total_chunks}")

# Specify the paths
input_directory = r"Data\Raw Text Files"
output_json_file = r"Data\document_chunks.json"

# Run the function
merge_text_files_to_json(input_directory, output_json_file)
