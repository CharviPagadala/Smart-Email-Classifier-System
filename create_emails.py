import pandas as pd
import os

# Set your data folder path
data_path = r"C:\Users\Student\Desktop\Spam\data"
os.makedirs(data_path, exist_ok=True)

# Sample email data
data = {
    "text": [
        "Congratulations! You've won a $1000 gift card. Click here to claim.",
        "Hi John, just wanted to confirm our meeting for 3PM tomorrow.",
        "Earn money working from home, no experience needed!",
        "Don’t forget to submit the assignment by tonight.",
        "Limited offer: Buy now and get 50% off!",
        "Hey, are you joining the study session later today?",
        "You’ve been selected for a free vacation!",
        "Your account statement is available online.",
        "Act now! This deal won't last long!",
        "Team meeting postponed to 4PM. Check calendar invite."
    ],
    "label": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"]
}

# Save as emails.csv in the correct folder
df = pd.DataFrame(data)
df.to_csv(os.path.join(data_path, "emails.csv"), index=False)

print("✅ emails.csv created successfully at:", data_path)
