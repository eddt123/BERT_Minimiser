import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Make sure to download these once before running the script
nltk.download('punkt')
nltk.download('stopwords')

def extract_keyword(sentence, exclude_word):
    # Tokenize the sentence
    words = word_tokenize(sentence)

    # Lowercase for comparison and remove punctuation
    words = [word.lower() for word in words if word.isalpha()]

    # Remove stopwords and the exclude word
    filtered_words = [word for word in words if word not in stopwords.words('english') and word != exclude_word.lower()]

    # Return the most common word as keyword, if available
    if filtered_words:
        most_common_word = Counter(filtered_words).most_common(1)[0][0]
        return most_common_word
    else:
        return None

# Load the Excel file
df = pd.read_excel('passages_96concepts.xlsx')

# Apply the function to extract keywords
df['Keyword'] = df.apply(lambda row: extract_keyword(row['Sent'], row['Name']), axis=1)

# Save the updated dataframe to a new Excel file
df.to_excel('updated_excel_file.xlsx', index=False)
