# 🛡️ SpamShield — AI Spam Detector
A complete beginner-friendly spam detector using **scikit-learn**, **Flask**, and **Vanilla JS**.

---

## 📁 Project Structure

```
spam-detector/
├── backend/
│   ├── train_model.py    ← trains & saves the ML model
│   ├── app.py            ← Flask REST API
│   ├── spam_model.pkl    ← saved model (auto-generated)
│   └── requirements.txt  ← Python dependencies
└── frontend/
    └── index.html        ← complete UI (no build step needed)
```

---

## 🚀 How to Run (Step-by-Step for Beginners)

### Step 1 — Install Python
Download Python 3.10+ from https://python.org and install it.
Make sure to check **"Add Python to PATH"** during installation!

Verify in terminal:
```bash
python --version
```

### Step 2 — Open a terminal in the project folder
```bash
cd path/to/spam-detector/backend
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
This installs: Flask, scikit-learn, flask-cors, numpy.

### Step 4 — Train the model
```bash
python train_model.py
```
You'll see accuracy results and a file `spam_model.pkl` is created.

### Step 5 — Start the Flask server
```bash
python app.py
```
You'll see: `🚀 Starting Spam Detector API on http://localhost:5000`

### Step 6 — Open the frontend
Simply open `frontend/index.html` in your browser (double-click it).

That's it! Paste any message and click **Analyze**.

---

## 🧠 How the ML Works

```
Raw Text  →  TF-IDF Vectorizer  →  MultinomialNB  →  spam / ham
```

1. **TF-IDF** (Term Frequency–Inverse Document Frequency)
   - Converts text into numbers
   - Words that appear often in spam but rarely elsewhere get high scores
   - E.g. "congratulations", "click here", "free" → high spam score

2. **Naive Bayes**
   - Learns the probability of each word given spam/ham class
   - At prediction time: P(spam | words) ∝ P(spam) × P(w1|spam) × P(w2|spam) × ...
   - "Naive" because it assumes words are independent (simplification that works!)

3. **Pipeline**
   - scikit-learn Pipeline chains steps together
   - `model.predict(["some text"])` runs both steps automatically

---

## 🔌 API Reference

**POST** `http://localhost:5000/predict`

Request body:
```json
{ "text": "Congratulations! You have won a prize!" }
```

Response:
```json
{
  "prediction": "spam",
  "is_spam": true,
  "confidence": 97.3,
  "spam_prob": 97.3,
  "ham_prob": 2.7,
  "signals": {
    "keywords_found": ["congratulations", "win", "prize"],
    "has_url": false,
    "has_phone": false,
    "all_caps_count": 0,
    "exclamation_count": 2,
    "dollar_signs": 0
  },
  "word_count": 7
}
```

---

## 📈 Improving the Model

1. **More data** — Use the [UCI SMS Spam dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection):
```python
import pandas as pd
df = pd.read_csv("spam.csv", encoding="latin-1")
texts  = df["v2"].tolist()
labels = df["v1"].tolist()  # "spam" or "ham"
```

2. **Try other classifiers** — Replace `MultinomialNB` with:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
```

3. **Cross-validation** — Get more reliable accuracy:
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, texts, labels, cv=5)
print(f"CV Accuracy: {scores.mean():.2%}")
```
