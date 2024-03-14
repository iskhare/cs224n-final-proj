from datasets import Dataset, DatasetDict
import json

def load_dataset_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return Dataset.from_dict(data)

# Paths to your JSON files
train_json_path = 'bigtrainENSW.json'
validation_json_path = 'bigdevENSW.json'
test_json_path = 'bigtestENSW.json'

# Load datasets
train_dataset = load_dataset_from_json(train_json_path)
validation_dataset = load_dataset_from_json(validation_json_path)
test_dataset = load_dataset_from_json(test_json_path)

# Create a DatasetDict
bigdatasetdict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

train_json_path = 'smalltrainENSW.json'
validation_json_path = 'smalldevENSW.json'
test_json_path = 'smalltestENSW.json'

# Load datasets
train_dataset = load_dataset_from_json(train_json_path)
validation_dataset = load_dataset_from_json(validation_json_path)
test_dataset = load_dataset_from_json(test_json_path)

# Create a DatasetDict
smalldatasetdict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})