# Fake News Detection System

## 🚀 Live Demo
Coming soon – Deployment on Streamlit Cloud.

## Key Features

- Dual input: Headline + Article content
- TF-IDF feature extraction (5000 features)
- Logistic Regression & Naive Bayes comparison
- 98.84% test accuracy
- Interactive Streamlit interface

## Overview
This project detects whether a news article is Fake or Real using Natural Language Processing (NLP) and Machine Learning techniques.
\## Overview

This project detects whether a news article is Fake or Real using Natural Language Processing (NLP) and Machine Learning techniques.



\## Tech Stack

\- Python

\- Scikit-learn

\- TF-IDF Vectorization

\- Logistic Regression

\- Naive Bayes

\- Streamlit



\## Dataset

Fake and Real News Dataset (Kaggle)



\## Model Performance



\### Logistic Regression

\- Accuracy: 98.84%

\- Precision: 0.99

\- Recall: 0.99

\- F1-score: 0.99



\### Naive Bayes

\- Accuracy: 93.80%

\- Precision: 0.94

\- Recall: 0.94

\- F1-score: 0.94



Logistic Regression outperformed Naive Bayes and was selected as the final deployed model.



\## How to Run



Install dependencies:

python -m pip install -r requirements.txt



Train model:

python train\_model.py



Run app:

python -m streamlit run app.py

