import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import torch

# Download the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load the data
file_path = 'sentiment data.csv'
data = pd.read_csv(file_path)

# Inspect the data
print("Initial Data Info:")
print(data.info())
print("\nInitial Data Head:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Standardize formats
# Remove leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Ensure the 'Sentiment' column is in lowercase
data['Sentiment'] = data['Sentiment'].str.lower()

# Remove any non-alphanumeric characters, emojis, and sequences of dots from the 'Sentence' column
def clean_text(text):
    text = re.sub(r'\.{2,}', '', text)  # Remove sequences of dots
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove non-alphanumeric characters
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    return text

data['Sentence'] = data['Sentence'].apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

data['Sentence'] = data['Sentence'].apply(remove_stopwords)

# Tokenization: Break down sentences into individual words or tokens
data['Tokens'] = data['Sentence'].apply(word_tokenize)

# Text Embedding using BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

data['Embedding'] = data['Sentence'].apply(embed_text)

# Verify the changes
print("\nCleaned, Tokenized, and Embedded Data Info:")
print(data.info())
print("\nCleaned, Tokenized, and Embedded Data Head:")
print(data.head())

# Save the cleaned, tokenized, and embedded data to a new CSV file
cleaned_file_path = 'cleaned_tokenized_embedded_sentiment_data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned, tokenized, and embedded data saved to {cleaned_file_path}")