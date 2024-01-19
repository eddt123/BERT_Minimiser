import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import pandas as pd

#use 15 topics to generate 4 sentences per topic

def generate_sentences(keyword, topic, num_sentences=1):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")

    # Encode the keyword
    input_ids = tokenizer.encode(f"generate a sentence of 7-18 words for this word: {topic} it must be related to this topic {keyword}.", return_tensors="pt")

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
    for sentence in sentences[:num_sentences]:
        print(sentence)
        # Add the sentence to the output file
        with open(output_file, 'a') as file:
            file.write(f"{keyword}\t{sentence}\n")
    return sentences[:num_sentences]


input_file = os.path.join('data', 'wordcloud_sentences.xlsx')
output_file = os.path.join('processed_data', 'output.txt')
data = pd.read_excel(input_file, usecols=[0, 3], nrows=60)
keywords = data.iloc[:, 0].tolist()

generated_sentences = set()  # To keep track of generated sentences

for keyword, topic in zip(keywords, data.iloc[:, 1]):
    sentences = generate_sentences(keyword, topic)
    for sentence in sentences:
        # Check if the sentence has already been generated for the keyword and topic
        if (keyword, sentence) not in generated_sentences:
            generated_sentences.add((keyword, sentence))
            # Open the file in append mode to append new content
            with open(output_file, 'a') as file:
                file.write(f"{keyword,topic}\t{sentence}\n")
