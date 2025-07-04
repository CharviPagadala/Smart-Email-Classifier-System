import pandas as pd
import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load features and labels
features_path = os.path.join("data", "features.pkl")
vectorizer_path = os.path.join("data", "tfidf_vectorizer.pkl")
cleaned_csv_path = os.path.join("data", "emails_cleaned.csv")
model_path = os.path.join("models", "nb_model.pkl")

if not os.path.exists(features_path) or not os.path.exists(cleaned_csv_path):
    print("❌ Required files not found.")
    exit()

# Load feature matrix and original data
with open(features_path, 'rb') as f:
    X = pickle.load(f)

df = pd.read_csv(cleaned_csv_path)

# Check label column exists
if 'label' not in df.columns:
    print("❌ 'label' column missing. Cannot train model.")
    exit()

y = df['label']

# Split data for testing accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Model trained. Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Model saved to: {model_path}")
