import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Define paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "resume_chatbot.keras"
LABELS_PATH = BASE_DIR / "labels.json"
INTENTS_PATH = BASE_DIR / "intents.json"

def load_chatbot():
    """Load the Keras model and its intent labels after training."""
    if not (MODEL_PATH.exists() and LABELS_PATH.exists()):
        return None, []
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels

def load_intents():
    return json.loads(INTENTS_PATH.read_text(encoding="utf-8"))["intents"]

def response_for(tag: str, intents: list[dict]) -> str:
    for intent in intents:
        if intent["tag"] == tag:
            return intent["responses"][0]
    return "I am sorry, I could not understand that. Please ask another résumé-related question."

def predict(message: str, model, labels: list[str]) -> tuple[str, float]:
    probabilities = model.predict(tf.constant([[message]], dtype=tf.string), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return labels[index], float(probabilities[index])

def main():
    print("\n===================================================")
    print("      ResumeBot AI - Terminal Edition")
    print("===================================================\n")
    print("Loading AI Engine...\n")

    model, labels = load_chatbot()
    intents = load_intents()

    if model is None:
        print("[ERROR] AI model not found. Please run 'python train_chatbot.py' first.")
        sys.exit(1)

    print("Next Gen Resume Analyzer is ACTIVE. (Type 'quit' or 'exit' to stop)\n")
    print("Bot: Hello! I am ResumeBot AI. Ask me about résumé skills, interview preparation, experience, or improving a résumé.")

    while True:
        try:
            question = input("\nYou: ")
            if question.lower().strip() in ['quit', 'exit']:
                print("\nBot: Goodbye and good luck with your job search!\n")
                break
            
            if not question.strip():
                continue

            tag, confidence = predict(question, model, labels)
            
            if confidence < 0.45:
                answer = "I am not fully sure I understood. Try asking about skills, résumé improvements, experience, education, or interview preparation."
            else:
                answer = response_for(tag, intents)
                
            print(f"Bot: {answer}")
            
        except KeyboardInterrupt:
            print("\nBot: Goodbye!\n")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TensorFlow warnings
    main()
