import json
import re

def auto_label_and_clean_documents(input_json, output_json):
    """Automatically labels documents and cleans text based on predefined rules."""

    # Replacement rules for UTF-8 characters and other text cleanups
    replacements = {
        'Mr.': 'Mr',
        'Mrs.': 'Mrs',
        'Ms.': 'Ms',
        'Dr.': 'Dr',
        '\u00a0': ' ',  # Non-breaking space
        '\u2019': "'",  # Right single quotation mark (apostrophe)
        '\u2018': "'",  # Left single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2010': '-',  # Hyphen
        '\u2022': '',   # Remove bullet point
        '\u00B6': '',   # Remove paragraph symbol
        '\u2761': '',   # Remove curved paragraph symbol
        '\u00A7': '',   # Remove section symbol 
        '\u2026': '',   # Remove ...
        '. . .': ' ',   # Remove ...
        '...': ' ',
        '[': '',
        ']': '',
        '*': ''
    }

    def clean_text(text):
        """Cleans the text by applying replacements and removing unwanted patterns."""
        text = text.replace('\t', ' ').replace('\n', ' ')
        text = re.sub(r'--+', '-', text)  # Replace multiple hyphens with one
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with one
        return text.strip()

    with open(input_json, 'r', encoding='utf-8') as file:
        documents = json.load(file)

    unlabeled_count = 0

    for doc in documents:
        # Clean the text
        doc['text'] = clean_text(doc.get('text', ''))

        # Apply labeling rules
        document_name = doc.get("document_name", "").lower()

        if document_name.startswith("speech_11"):
            doc["document_label"] = "floor speech"
        elif document_name.startswith("statement_for_record_"):
            doc["document_label"] = "statement for the record"
        elif document_name.startswith("witness_testimony_") or re.search(r"\d+-\d+-\d+.txt", document_name):
            doc["document_label"] = "witness testimony"
        elif "amicus" in document_name:
            doc["document_label"] = "amicus curiae"
        elif re.match(r"hr \d+ report \d+-\d+\.txt", document_name):
            doc["document_label"] = "committee report"
        else:
            doc["document_label"] = None
            unlabeled_count += 1

    with open(output_json, 'w', encoding='utf-8') as file:
        json.dump(documents, file, ensure_ascii=False, indent=4)

    print(f"Document labeling and cleaning complete. Results saved to {output_json}")
    print(f"Number of unlabeled document chunks: {unlabeled_count}")

# Specify the input and output file paths
input_json_file = r"Data\document_chunks.json"
output_json_file = r"Data\clean_document_chunks.json"

# Run the function
auto_label_and_clean_documents(input_json_file, output_json_file)
