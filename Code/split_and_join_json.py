import json
import os

def split_json_file(input_file, output_dir, num_parts=6):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the full JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if data is a list
    if not isinstance(data, list):
        raise ValueError("The JSON file must contain a list at the top level.")

    total_items = len(data)
    part_size = total_items // num_parts

    for i in range(num_parts):
        start = i * part_size
        end = None if i == num_parts - 1 else (i + 1) * part_size
        part_data = data[start:end]

        output_file = os.path.join(output_dir, f'part_{i + 1}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(part_data, f, ensure_ascii=False, indent=2)

    print(f"Split into {num_parts} parts successfully.")

def merge_json_files(input_dir, output_file):
    merged_data = []

    # Sort files to merge in order (optional but nice)
    files = sorted(f for f in os.listdir(input_dir) if f.endswith('.json'))

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Check if data is a list
            if not isinstance(data, list):
                raise ValueError(f"File {filename} does not contain a list at the top level.")

            merged_data.extend(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(files)} files into {output_file}.")


# Split the large JSON file
# split_json_file('Data/Finetuning/train_data.json', 'Data/Finetuning/Fragments')

# Merge back all JSON files in the directory
merge_json_files('Data/Finetuning/Fragments', 'Data/Finetuning/train_data.json')
