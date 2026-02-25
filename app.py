import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(page_title="Fake News Detector", layout="centered")

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)        # remove extra spaces
    text = re.sub(r'[^a-zA-Z]', ' ', text) # remove special characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
# UI
st.title("📰 Fake News Detection System")
st.markdown("### NLP + TF-IDF + Logistic Regression")

st.markdown("---")

headline = st.text_input("Enter News Headline")
article = st.text_area("Enter News Article Content", height=200)

if st.button("Analyze Article"):

    if headline.strip() == "" and article.strip() == "":
        st.warning("Please enter headline and/or article content.")
    else:
        combined_input = headline + " " + article
        combined_input = combined_input[:10000]   # limit very long articles
        cleaned = clean_text(combined_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)[0]

        fake_prob = round(probability[0] * 100, 2)
        real_prob = round(probability[1] * 100, 2)

        st.markdown("---")

        if prediction[0] == 0:
            st.error("🚨 Prediction: FAKE NEWS")
            st.progress(int(fake_prob))
            st.write(f"Confidence: {fake_prob}%")
        else:
            st.success("✅ Prediction: REAL NEWS")
            st.progress(int(real_prob))
            st.write(f"Confidence: {real_prob}%")

        st.markdown("### 📊 Probability Breakdown")
        st.write(f"Fake Probability: {fake_prob}%")
        st.write(f"Real Probability: {real_prob}%")

        st.markdown("---")
        st.info("Model: Logistic Regression | Features: TF-IDF (5000)")