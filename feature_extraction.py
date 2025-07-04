import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed data
input_path = os.path.join("data", "emails_cleaned.csv")
output_features_path = os.path.join("data", "features.pkl")
output_vectorizer_path = os.path.join("data", "tfidf_vectorizer.pkl")

if not os.path.exists(input_path):
    print(f"❌ File not found: {input_path}")
    exit()

df = pd.read_csv(input_path)

if 'clean_text' not in df.columns:
    print("❌ 'clean_text' column missing in the dataset.")
    exit()

# Apply TF-IDF
vectorizer = TfidfVectorizer(max_features=500)  # limit to top 500 words
X = vectorizer.fit_transform(df['clean_text'])

# Save features and vectorizer
with open(output_features_path, 'wb') as f:
    pickle.dump(X, f)

with open(output_vectorizer_path, 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"✅ TF-IDF features saved to: {output_features_path}")
print(f"✅ TF-IDF vectorizer saved to: {output_vectorizer_path}")
