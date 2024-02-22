import os
import json
import nltk
import nltk.translate.bleu_score as bleu
import random
from nltk.tokenize import word_tokenize

# Make sure you have the NLTK data
nltk.download('punkt')

# Function to load JSON data from a file
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Function to perform Random Words translation
def random_words_translation(data):
    english_vocabulary = set()
    for _, english_sentence in data['train']:
        english_vocabulary.update(word_tokenize(english_sentence.lower()))
        print

    random_translations = []
    for source_sentence, target_sentence, _ in data['test']:
        tokens = word_tokenize(source_sentence)
        random_translation = ' '.join(random.choice(list(english_vocabulary)) for _ in tokens)
        random_translations.append((random_translation, target_sentence))
    return random_translations

# Function to calculate BLEU scores for one file
def calculate_bleu_for_file(file_path):
    data = load_json(file_path)
    translations = random_words_translation(data)
    bleu_scores = []
    for model_output, reference in translations:
        hypothesis = word_tokenize(model_output)
        reference = [word_tokenize(reference)]
        score = bleu.sentence_bleu(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(score)
    return bleu_scores

# Replace 'your_folder_path' with the actual path to your folder containing JSON files
folder_path = 'trial_data_with_answers'

# List all JSON files in the folder
json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

# Calculate BLEU scores for all files and accumulate them
all_bleu_scores = []
for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    file_bleu_scores = calculate_bleu_for_file(file_path)
    all_bleu_scores.extend(file_bleu_scores)

# Calculate and print the average BLEU score over all files
if all_bleu_scores:
    average_bleu_score = sum(all_bleu_scores) / len(all_bleu_scores)
    print(f"Average BLEU score over all files: {average_bleu_score}")
else:
    print("No BLEU scores to average.")
