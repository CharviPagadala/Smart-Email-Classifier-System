import pandas as pd
import re
import string
import os
import nltk

from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load data
def preprocess_emails(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    if 'text' not in df.columns:
        print("❌ Column 'text' not found in input CSV.")
        return

    df['clean_text'] = df['text'].apply(clean_text)

    # Save preprocessed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed emails saved to: {output_path}")

# Run if executed directly
if __name__ == "__main__":
    input_csv = os.path.join("data", "emails.csv")
    output_csv = os.path.join("data", "emails_cleaned.csv")
    preprocess_emails(input_csv, output_csv)
