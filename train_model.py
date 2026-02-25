import pandas as pd
import numpy as np
import re
import pickle
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Combine title + text
data["content"] = data["title"] + " " + data["text"]

# Clean combined content
data["clean_text"] = data["content"].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, pred))
print("\nLogistic Regression Report:\n", classification_report(y_test, pred))
print("\nLogistic Regression Confusion Matrix:\n", confusion_matrix(y_test, pred))

# 🔥 ADD THIS PART BELOW

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("\nNaive Bayes Report:\n", classification_report(y_test, nb_pred))
print("\nNaive Bayes Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel saved successfully.")