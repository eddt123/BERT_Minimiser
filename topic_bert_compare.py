import pandas as pd
from transformers import RobertaForMaskedLM, AutoTokenizer, AutoModelForSequenceClassification
from bert_score import score as bert_score  # Rename the imported function to avoid conflict
import numpy as np

# Function to calculate BERT score
def calculate_bert_score(sent1, sent2):
    print(f"Calculating BERT score between: \nSentence 1: {sent1} \nSentence 2: {sent2}")
    P, R, F1 = bert_score([sent1], [sent2], lang="en", model_type="roberta-large", rescale_with_baseline=True)
    bert_score_value = F1.item()
    print(f"BERT score: {bert_score_value}\n")
    return bert_score_value

# Load your data
file_path = 'test.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Group sentences by topic
grouped_data = data.groupby('Name')['Sent'].apply(list)

# Initialize variables for intra-topic and inter-topic score calculations
intra_topic_scores = {}
inter_topic_scores = []

# Intra-topic BERT score calculation
for topic, sentences in grouped_data.items():
    if len(sentences) > 1:
        scores = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                score = calculate_bert_score(sentences[i], sentences[j])
                scores.append(score)
        intra_topic_scores[topic] = np.mean(scores)

# Inter-topic BERT score calculation (comparing first sentence of each topic)
topics = list(grouped_data.keys())
for i in range(len(topics)):
    for j in range(i + 1, len(topics)):
        score = calculate_bert_score(grouped_data[topics[i]][0], grouped_data[topics[j]][0])
        inter_topic_scores.append({'topic1': topics[i], 'topic2': topics[j], 'score': score})

# Output
print("Intra-topic BERT Scores:")
for topic, score in intra_topic_scores.items():
    print(f"{topic}: {score}")

print("\nInter-topic BERT Scores:")
for score_info in inter_topic_scores:
    print(f"{score_info['topic1']} vs {score_info['topic2']}: {score_info['score']}")
