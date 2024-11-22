import pandas as pd
from transformers import BertTokenizer, BertModel
import torch

# Load the tokenized data
file_path = 'tokenized_sentiment_data.csv'
data = pd.read_csv(file_path)

# Inspect the data
print("Initial Data Info:")
print(data.info())
print("\nInitial Data Head:")
print(data.head())

# Feature Extraction using FinBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finbert_tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
finbert_model = BertModel.from_pretrained('yiyanghkust/finbert-tone').to(device)

def embed_text_with_finbert(tokens):
    text = ' '.join(tokens)  # Join tokens back into a single string
    inputs = finbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

data['Embedding'] = data['Tokens'].apply(eval).apply(embed_text_with_finbert)

# Verify the changes
print("\nTokenized and Embedded Data Info:")
print(data.info())
print("\nTokenized and Embedded Data Head:")
print(data.head())

# Save the tokenized and embedded data to a new CSV file
embedded_file_path = 'embedded_sentiment_data_with_finbert.csv'
data.to_csv(embedded_file_path, index=False)
print(f"\nTokenized and embedded data saved to {embedded_file_path}")