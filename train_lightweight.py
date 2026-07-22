"""
Lightweight trainer — replaces TensorFlow/Keras with scikit-learn.
Run this once locally to generate: chatbot_model.pkl
"""
import json
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).parent
INTENTS_PATH = BASE_DIR / "intents.json"

intents = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))

X, y = [], []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern.lower())
        y.append(intent["tag"])

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf",   LogisticRegression(max_iter=500, C=5.0)),
])
model.fit(X, y)

with open(BASE_DIR / "chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("[SUCCESS] Lightweight model saved to chatbot_model.pkl")
print(f"   Classes: {model.classes_.tolist()}")
