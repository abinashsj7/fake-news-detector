import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("Fake News Detection System")

input_text = st.text_area("Enter News Article Text")

if st.button("Analyze"):
    if input_text.strip() != "":
        cleaned = clean_text(input_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)

        confidence = round(max(probability[0]) * 100, 2)

        if prediction[0] == 0:
            st.error(f"⚠️ This news is likely FAKE\nConfidence: {confidence}%")
        else:
            st.success(f"✅ This news is likely REAL\nConfidence: {confidence}%")
    else:
        st.warning("Please enter some text.")