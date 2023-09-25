import os
import random
import string
from bert_score import score
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Initialize BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

# Function to calculate BERT score between two sentences
def calculate_bert_score(sent1, sent2):
    P, R, F1 = score([sent1], [sent2], lang="en", model_type="bert-base-uncased", rescale_with_baseline=True)
    return F1.item()

# Function to modify a sentence
def modify_sentence(sentence):
    words = sentence.split()
    choice = random.choice(["replace", "delete", "add"])
    if choice == "replace" and len(words) > 1:
        idx = random.randint(0, len(words) - 1)
        words[idx] = random.choice(string.ascii_lowercase)
    elif choice == "delete" and len(words) > 1:
        idx = random.randint(0, len(words) - 1)
        del words[idx]
    elif choice == "add":
        idx = random.randint(0, len(words))
        words.insert(idx, random.choice(string.ascii_lowercase))
    return " ".join(words)

# Create processed_data folder if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

# Read sentences from data folder
sentences = {}
for filename in os.listdir('data'):
    with open(f'data/{filename}', 'r') as f:
        sentences[filename] = f.read().strip()

# Initialize variables
best_sentences = sentences.copy()

# Modify sentences to minimize BERT score
for _ in range(1000):  # Number of iterations
    # Generate a new set of modified sentences
    new_sentences = {filename: modify_sentence(sent) for filename, sent in sentences.items()}
    
    # Calculate the total BERT score for the new set
    total_score = 0
    for filename1, sent1 in new_sentences.items():
        for filename2, sent2 in new_sentences.items():
            if filename1 != filename2:
                bert_score = calculate_bert_score(sent1, sent2)
                print(f"BERT Score between {filename1} and {filename2}: {bert_score}")
                total_score += bert_score
    
    # Update if the new total score is lower
    if total_score < sum(calculate_bert_score(best_sentences[filename1], best_sentences[filename2])
                         for filename1 in best_sentences for filename2 in best_sentences if filename1 != filename2):
        best_sentences = new_sentences.copy()

# Save the modified sentences to processed_data folder
for filename, sent in best_sentences.items():
    with open(f'processed_data/{filename}', 'w') as f:
        f.write(sent)
