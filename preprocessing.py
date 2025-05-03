
# -----------------------------------------------------------------------------
# Run through the dataset using the following data processing functions
# -----------------------------------------------------------------------------

import json

def deduplicate_entities(entities):
    # Remove exact duplicate entity tuples and sort by start index.
    unique = list({(start, end, label) for start, end, label in entities})
    unique.sort(key=lambda x: x[0])
    return unique

def resolve_overlapping_entities(entities):
    # Sort entities by their start index
    entities = sorted(entities, key=lambda x: x[0])
    resolved = []
    for ent in entities:
        if not resolved:
            resolved.append(ent)
        else:
            last = resolved[-1]
            # Check if the new entity overlaps with the last one in the resolved list.
            if ent[0] < last[1]:
                # Overlap exists; choose the one with the longer span.
                if (ent[1] - ent[0]) > (last[1] - last[0]):
                    resolved[-1] = ent  # Replace the last entity with the longer one.
                # Otherwise, skip adding the new entity.
            else:
                resolved.append(ent)
    return resolved

def convert_labelstudio_to_spacy(filename):
    training_data = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # Extract the text from the "data" field (adjust key if needed)
        text = item.get("data", {}).get("text_data", "")
        if not text:
            continue

        entities = []
        # Loop through each annotation result.
        for annotation in item.get("annotations", []):
            for result in annotation.get("result", []):
                value = result.get("value", {})
                start = value.get("start")
                end = value.get("end")
                labels = value.get("labels", [])
                for label in labels:
                    if start is not None and end is not None:
                        entities.append((start, end, label))

        if text and entities:
            # Remove exact duplicates
            unique_entities = deduplicate_entities(entities)
            # Resolve overlapping spans by keeping the longer one.
            non_overlapping_entities = resolve_overlapping_entities(unique_entities)
            training_data.append((text, {"entities": non_overlapping_entities}))

    return training_data

def process_and_save_training_data(input_file, output_file):
        """
        Converts Label Studio data to spaCy training format and saves it to a file.

        Args:
            input_file (str): Path to the input JSON file containing Label Studio data.
            output_file (str): Path to the output file where converted data will be saved.
        """
        spacy_training_data = convert_labelstudio_to_spacy(input_file)

        # Save the converted training data in JSON format.
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(spacy_training_data, f_out, ensure_ascii=False, indent=2)

        print(f"Converted training data has been saved to {output_file}")


def load_and_split_dataset(data_file, output_folder, split_ratio=0.8):
    """
    Load dataset from a file, shuffle it, split into training and testing sets, and save them to a specified folder.

    Args:
        data_file (str): Path to the dataset file.
        output_folder (str): Path to the folder where the split files will be saved.
        split_ratio (float): Ratio of training data to the total dataset.

    Returns:
        tuple: Number of total examples, training examples, and testing examples.
    """
    import os
    import random

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)                  # Shuffle data
    split_index = int(len(data) * split_ratio)  # Split data based on ratio

    train_data = data[:split_index]       # Training Data
    test_data = data[split_index:]        # Testing Data

    # Define output file paths
    train_file = os.path.join(output_folder, "train_data.json")
    test_file = os.path.join(output_folder, "test_data.json")

    # Save training and testing data
    with open(train_file, "w", encoding="utf-8") as train_file_obj:
        json.dump(train_data, train_file_obj, ensure_ascii=False, indent=4)

    with open(test_file, "w", encoding="utf-8") as test_file_obj:
        json.dump(test_data, test_file_obj, ensure_ascii=False, indent=4)

    return len(data), len(train_data), len(test_data)
