import os
import random
import string
import logging
from bert_score import score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging
import torch
from transformers import RobertaForMaskedLM


# Suppress warnings from transformers library
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()

# Initialize Roberta model for Masked Language Modeling
mlm_model = RobertaForMaskedLM.from_pretrained("roberta-large")

# Initialize DeBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large")

# Function to calculate BERT score between two sentences
def calculate_bert_score(sent1, sent2):
    P, R, F1 = score([sent1], [sent2], lang="en", model_type="microsoft/deberta-xlarge-mnli", rescale_with_baseline=True)
    return F1.item()

# Function to modify a sentence using Masked Language Modeling
def modify_sentence(sentence):
    words = sentence.split()
    if len(words) <= 2:
        return sentence  # Don't modify if the sentence is too short
    
    # Randomly select a word to mask (avoid first and last word)
    mask_idx = random.randint(1, len(words) - 2)
    words[mask_idx] = tokenizer.mask_token
    
    # Prepare input for MLM
    input_text = " ".join(words)
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    
    # Get MLM prediction for the masked token
    with torch.no_grad():
        output = mlm_model(input_ids).logits
    mask_token_logits = output[0, mask_idx]
    mask_token_id = torch.argmax(mask_token_logits).item()
    
    # Replace the masked token with the predicted token
    predicted_token = tokenizer.convert_ids_to_tokens([mask_token_id])[0]
    words[mask_idx] = predicted_token
    
    return " ".join(words)

# Create processed_data folder if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

# Read sentences from data folder
sentences = {}
for filename in os.listdir('data'):
    with open(f'data/{filename}', 'r') as f:
        sentences[filename] = f.read().strip()

# Calculate and print initial BERT scores
print("Initial BERT Scores:")
for filename1, sent1 in sentences.items():
    for filename2, sent2 in sentences.items():
        if filename1 != filename2:
            bert_score = calculate_bert_score(sent1, sent2)
            print(f"BERT Score between {filename1} and {filename2}: {bert_score}")

# Initialize variables
best_sentences = sentences.copy()

print("Modified BERT Scores:")

# Modify sentences to minimize BERT score
for i in range(10):  # Number of iterations
    # Generate a new set of modified sentences
    new_sentences = {filename: modify_sentence(sent) for filename, sent in best_sentences.items()}
    
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

    # Save the modified sentences to processed_data folder after each iteration
    for filename, sent in best_sentences.items():
        with open(f'processed_data/{filename}', 'w', encoding='utf-8') as f:
            f.write(sent)
    
    print(f"Iteration {i+1} completed. Sentences saved.")

