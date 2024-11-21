import pandas as pd
import numpy as np
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
import torch

# Download the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Verify the changes
print("\nCleaned Data Info:")
print(data.info())
print("\nCleaned Data Head:")
print(data.head())

# Save the cleaned data to a new CSV file
cleaned_file_path = 'cleaned_sentiment_data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned data saved to {cleaned_file_path}")

# Remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

data['Sentence'] = data['Sentence'].apply(remove_stopwords)
# Verify the changes
print("\nRemove stopwords Data Info:")
print(data.info())
print("\nRemove stopwords Data Head:")
print(data.head())

# Save the Remove stopwords data to a new CSV file
cleaned_file_path = 'remove_stopwords_sentiment_data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nRemove stopwords saved to {cleaned_file_path}")

# Lemmatization
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

data['Sentence'] = data['Sentence'].apply(lemmatize_text)

# Verify the changes
print("\nStemming/Lemmatization Data Info:")
print(data.info())
print("\nStemming/Lemmatization Data Head:")
print(data.head())

# Save the Stemming/Lemmatization data to a new CSV file
cleaned_file_path = 'stem_lemmatize_sentiment_data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nStemming/Lemmatization data saved to {cleaned_file_path}")

# Tokenization: Break down sentences into individual words or tokens
data['Tokens'] = data['Sentence'].apply(word_tokenize)

# Verify the changes
print("\nTokenized Data Info:")
print(data.info())
print("\nTokenized Data Head:")
print(data.head())

# Save the tokenized data to a new CSV file
cleaned_file_path = 'tokenized_sentiment_data.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nTokenized data saved to {cleaned_file_path}")
print("\nTokenized Data Head:")
print(data.head())

# Feature Extraction using FinBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = BertModel.from_pretrained('yiyanghkust/finbert-tone').to(device)

def embed_text_with_finbert(text):
    inputs = finbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

data['Embedding'] = data['Sentence'].apply(embed_text_with_finbert)

# Verify the changes
print("\nCleaned, Tokenized, and Embedded Data Info:")
print(data.info())
print("\nCleaned, Tokenized, and Embedded Data Head:")
print(data.head())

# Save the cleaned, tokenized, and embedded data to a new CSV file
cleaned_file_path = 'cleaned_tokenized_embedded_sentiment_data_with_finbert.csv'
data.to_csv(cleaned_file_path, index=False)
print(f"\nCleaned, tokenized, and embedded data saved to {cleaned_file_path}")