import gradio as gr
import joblib
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download("stopwords")

# Load stopwords
stop_words = set(stopwords.words("english"))

# Load Model and TF-IDF
model = joblib.load("best_linear_svm.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# ==========================
# Text Cleaning Function
# ==========================
def clean_text(text):

    text = text.lower()

    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# ==========================
# Prediction Function
# ==========================
def predict_sentiment(review):

    review = clean_text(review)

    vector = tfidf.transform([review])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "😊 Positive Review"

    return "😞 Negative Review"


# ==========================
# Gradio Interface
# ==========================

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Enter your movie review here..."
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="🎬 Movie Review Sentiment Analysis",
    description="Predict whether a movie review is Positive or Negative using a trained Linear SVM model."
)

demo.launch()