import os
import logging
import random
from bert_score import score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging, RobertaForMaskedLM
import itertools
import numpy as np

# Suppress warnings from transformers library
logging.basicConfig(level=logging.ERROR)
hf_logging.set_verbosity_error()

# Initialize Roberta model for Masked Language Modeling
mlm_model = RobertaForMaskedLM.from_pretrained("roberta-large")

# Initialize DeBERTa model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForSequenceClassification.from_pretrained("roberta-large")

def calculate_bert_score(sent1, sent2):
    P, R, F1 = score([sent1], [sent2], lang="en", model_type="roberta-large", rescale_with_baseline=True)
    bert_score = F1.item()
    print(f"BERT score between '{sent1}' and '{sent2}': {bert_score}")
    return bert_score

def read_sentences(file_path):
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    return [sentence.strip() for sentence in sentences]

def calculate_total_bert_score(sentences):
    return np.mean([calculate_bert_score(sent1, sent2) for sent1, sent2 in itertools.combinations(sentences, 2)])

def select_sentences(input_sentences, output_file, total_sentences=50):
    selected_sentences = random.sample(input_sentences, total_sentences)
    while True:
        for i in range(total_sentences):
            current_score = calculate_total_bert_score(selected_sentences)
            new_sentence = random.choice(input_sentences)
            temp_sentences = selected_sentences.copy()
            temp_sentences[i] = new_sentence
            new_score = calculate_total_bert_score(temp_sentences)

            if new_score < current_score:
                selected_sentences[i] = new_sentence
                with open(output_file, 'w') as file:
                    for sentence in selected_sentences:
                        file.write(sentence + '\n')
                print(f"Replaced sentence at position {i} with '{new_sentence}'. New total BERT score: {new_score}")

def main():
    input_file = os.path.join('data', 'stimuli_wordcloud_sentences.txt')
    output_file = os.path.join('processed_data', 'output.txt')

    sentences = read_sentences(input_file)
    select_sentences(sentences, output_file)

if __name__ == "__main__":
    main()
