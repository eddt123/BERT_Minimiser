import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import pandas as pd

#use 15 topics to generate 4 sentences per topic

def generate_sentences(keyword, num_sentences=10):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")

    # Encode the keyword
    input_ids = tokenizer.encode(f"generate a sentence of 7-18 words for this word: {keyword}", return_tensors="pt")

    # Set pad_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # Generate text using the model
    output = model.generate(
        input_ids, 
        max_length=100, 
        num_return_sequences=1
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Split the text into sentences and return the first `num_sentences`
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences[:num_sentences]


input_file = os.path.join('wordcloud_sentences.xlsx')
output_file = os.path.join('processed_data', 'output.txt')
data = pd.read_excel(input_file)
keywords = data.iloc[:, 3].tolist()

for keyword in keywords:
    sentences = generate_sentences(keyword)
    for sentence in sentences:
        with open(output_file, 'a') as file:
            file.write(f"{keyword}\t{sentence}\n")
