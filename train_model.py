"""
train_model.py
--------------
Trains a Naive Bayes spam classifier using scikit-learn
and saves the model to disk so the Flask server can load it.
"""

import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── 1. Training data ──────────────────────────────────────────────────────────
# A curated set of spam and ham (not-spam) messages.
# In a real project you would load a CSV dataset (e.g. the UCI SMS Spam dataset).

SPAM_MESSAGES = [
    "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim now!",
    "FREE entry in 2 a weekly competition to win FA Cup Final tkts. Text FA to 87121",
    "You have been selected as a lucky winner! Call now to claim your prize!",
    "URGENT: Your account has been compromised. Verify now at http://secure-login.xyz",
    "Make money fast working from home! No experience needed. Earn $500 daily!",
    "Win a brand new iPhone 15! Just complete this short survey: bit.ly/xxxxx",
    "You owe back taxes. Call the IRS immediately or face arrest: 800-xxx-xxxx",
    "Cheap Viagra and Cialis! No prescription needed. Discreet shipping worldwide.",
    "Investment opportunity! Double your money in 30 days guaranteed!",
    "Your PayPal account is suspended. Verify here: paypal-secure.phish.com",
    "WINNER!! As a valued network customer you have been selected to receive a prize.",
    "Get rich quick! Thousands of people already making $5000/week from home!",
    "Limited time offer: 90% discount on luxury watches. Order now!",
    "Claim your lottery prize of $750,000. Reply with your bank details.",
    "Hot singles in your area! Click here to meet them tonight!",
    "You have won a Nokia 6230! To claim call 09064012160. T&C apply.",
    "CASH PRIZE: You've been chosen for a $500 reward. Text PRIZE to 5555",
    "Urgent loan offer! Get $10,000 in 24 hours, no credit check required.",
    "Your subscription to Adult dating is active. Cancel at adultsite.com",
    "FREE RINGTONES! Reply RINGTONE to get 50 free ringtones for your phone!",
    "Congratulations you have won the email lottery prize of 1,000,000 USD",
    "Click here to unsubscribe or your account will be charged $49.99/month",
    "You are pre-approved for a credit card with a $5000 limit. Apply now!",
    "ALERT: Unusual activity detected. Login immediately to secure your account.",
    "Make $3000 per week stuffing envelopes at home. No experience needed!",
]

HAM_MESSAGES = [
    "Hey, are you coming to the meeting tomorrow at 10am?",
    "I'll pick up groceries on my way home. Need anything?",
    "Just finished the report. Can you review it when you get a chance?",
    "Happy birthday! Hope you have a wonderful day!",
    "The project deadline has been moved to next Friday.",
    "Can we reschedule our call to 3pm instead of 2pm?",
    "Thanks for your help with the presentation yesterday!",
    "The weather looks great for the weekend. Planning a hike?",
    "I sent you the document via email. Please check your inbox.",
    "Reminder: your dentist appointment is on Thursday at 2pm.",
    "Loved the book you recommended. Finished it last night!",
    "Running 10 minutes late, stuck in traffic. See you soon.",
    "The code review is done. I left a few comments on line 42.",
    "Let's catch up over coffee next week. Are you free Monday?",
    "Thanks for covering my shift last week. I owe you one!",
    "The package was delivered. Left it at the front door.",
    "Team lunch is on Friday at noon. We're going to that Italian place.",
    "I fixed the bug. The app is now working correctly.",
    "Mom called. She wants you to call her back when you get a chance.",
    "The quarterly report is ready. Sharing it with the team now.",
    "Your flight is confirmed for March 15. Check your email for details.",
    "I'll be at the gym until 7pm. Join me if you want!",
    "The new version of the software is out. I updated our systems.",
    "Great job on the presentation today! The client was very impressed.",
    "Can you send me the Wi-Fi password? I'm working from the office today.",
]

# ── 2. Build dataset ──────────────────────────────────────────────────────────
texts  = SPAM_MESSAGES + HAM_MESSAGES
labels = ["spam"] * len(SPAM_MESSAGES) + ["ham"] * len(HAM_MESSAGES)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ── 3. Build & train pipeline ─────────────────────────────────────────────────
# TfidfVectorizer converts raw text → numerical feature matrix.
# MultinomialNB is fast and works great for text classification.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",   # remove common words like "the", "is"
        ngram_range=(1, 2),     # use single words AND word pairs as features
        max_features=5000,      # keep only the top 5000 features
    )),
    ("clf", MultinomialNB(alpha=0.1)),  # alpha = Laplace smoothing
])

pipeline.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print()
print(classification_report(y_test, y_pred))

# ── 5. Save model ─────────────────────────────────────────────────────────────
model_path = os.path.join(os.path.dirname(__file__), "spam_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Model saved to: {model_path}")
