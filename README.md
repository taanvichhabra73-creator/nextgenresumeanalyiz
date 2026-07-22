# ResumeBot AI — Keras Language Chatbot

A Streamlit chatbot trained with Python, TensorFlow, and Keras to understand common résumé-screening questions.

## Features

- Natural-language intent classification with a Keras neural network
- Training data for eight résumé and interview topics
- Chat interface built with Streamlit
- No API key required
- Fair-use guardrails: supports human review and never makes hiring decisions

## Run locally

```bash
git clone <your-repository-url>
cd resumebot-ai
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python train_chatbot.py
streamlit run app.py
```

Open the local URL shown by Streamlit (usually `http://localhost:8501`).

The training command creates `resume_chatbot.keras` and `labels.json`. These are generated files and do not need to be uploaded to GitHub; anyone can recreate them by running the training command.

## Suggested deployment

Deploy on Streamlit Community Cloud. Add the same `pip install -r requirements.txt` and `python train_chatbot.py` commands to the build process, or commit the generated model files after training.

## Responsible use

Only evaluate job-related qualifications. Do not use this tool to infer or evaluate protected characteristics, and keep a human reviewer responsible for all hiring decisions.
