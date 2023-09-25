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

# Loop through each file in the data folder
for filename in os.listdir('data'):
    with open(f'data/{filename}', 'r') as f:
        sent1 = f.read().strip()

    # Initialize variables
    max_score_diff = 0
    best_sent1 = sent1

    # Modify sentence to maximize BERT score difference
    for _ in range(1000):  # Number of iterations
        # Modify sentence
        new_sent1 = modify_sentence(sent1)
        
        # Calculate new BERT score
        new_score = calculate_bert_score(new_sent1, sent1)
        
        # Update if the new score is better
        if abs(new_score) > max_score_diff:
            max_score_diff = abs(new_score)
            best_sent1 = new_sent1

    # Save the modified sentence to processed_data folder
    with open(f'processed_data/{filename}', 'w') as f:
        f.write(best_sent1)
