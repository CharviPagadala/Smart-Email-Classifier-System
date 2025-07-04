import os
import pickle
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function (same as earlier)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load model and vectorizer
model_path = os.path.join("models", "nb_model.pkl")
vectorizer_path = os.path.join("data", "tfidf_vectorizer.pkl")

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    print("❌ Model or vectorizer file not found.")
    exit()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# ---- MAIN LOGIC ----
def predict_email(email_text):
    cleaned = clean_text(email_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0].max()
    print(f"📧 Prediction: {prediction.upper()} ({proba * 100:.2f}% confidence)")

# ---- EXAMPLE USAGE ----
if __name__ == "__main__":
    # Replace this with any email content you want to test
    test_email = """
    Hello user! You've won a $1000 gift card. Click here to claim your prize now!
    """
    predict_email(test_email)
