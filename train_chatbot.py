"""Train the Keras NLP intent-classification model for ResumeBot AI."""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).parent
DATA = json.loads((BASE_DIR / "intents.json").read_text(encoding="utf-8"))["intents"]

texts, labels = [], []
for intent in DATA:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

class_names = sorted(set(labels))
label_to_id = {label: index for index, label in enumerate(class_names)}
y = np.array([label_to_id[label] for label in labels])
text_array = tf.constant([[t] for t in texts], dtype=tf.string)

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=2000,
    output_mode="int",
    output_sequence_length=20,
    standardize="lower_and_strip_punctuation",
)
vectorize.adapt(text_array)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorize,
    tf.keras.layers.Embedding(input_dim=2000, output_dim=64),
    tf.keras.layers.Conv1D(128, 5, activation="relu"),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation="softmax"),
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(text_array, y, epochs=180, verbose=0)
model.save(BASE_DIR / "resume_chatbot.keras")
(BASE_DIR / "labels.json").write_text(json.dumps(class_names), encoding="utf-8")
print(f"Training complete. Saved model for {len(class_names)} intents.")
