import pandas as pd
from modules.preprocessing import clean_text
from modules.feature_engineering import extract_features
from modules.model import train_model
from modules.predictor import classify_email

# Step 1: Load and clean data
data = pd.read_csv('data/emails.csv')
data['cleaned'] = data['EmailText'].apply(clean_text)

# Step 2: Feature extraction
X = extract_features(data['cleaned'])
y = data['Label']

# Step 3: Train model
train_model(X, y)

# Step 4: Predict a new email
email = "Congratulations! You've won free coupons!"
prediction, confidence = classify_email(email)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
