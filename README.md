# 🎬 Movie Review Sentiment Analysis using NLP and Linear SVM

## 📌 Project Overview

This project is a Natural Language Processing (NLP) based Sentiment Analysis system that predicts whether a movie review is **Positive** or **Negative**.

The model is trained on the IMDb Movie Reviews dataset using **TF-IDF Vectorization** and a **Linear Support Vector Machine (Linear SVM)** classifier. Hyperparameter tuning with **GridSearchCV** was used to improve model performance.

A Gradio web application is included for easy interaction and deployment on Hugging Face Spaces.

---

## 🚀 Features

* Movie Review Sentiment Analysis
* Text Preprocessing
* TF-IDF Feature Extraction
* Linear SVM Classification
* Hyperparameter Tuning using GridSearchCV
* Performance Evaluation
* Gradio Web Interface
* Hugging Face Deployment

---

## 📂 Dataset

Dataset: IMDb Movie Reviews Dataset

The dataset contains 50,000 labeled movie reviews.

* Positive Reviews
* Negative Reviews

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* BeautifulSoup
* Gradio
* Joblib

---

## 📊 Text Preprocessing

The following preprocessing techniques were applied:

* Convert text to lowercase
* Remove HTML tags
* Remove URLs
* Remove numbers
* Remove punctuation
* Remove extra spaces
* Remove English stopwords

---

## 🤖 Machine Learning Pipeline

Movie Review

↓

Text Cleaning

↓

TF-IDF Vectorization

↓

Linear SVM

↓

Sentiment Prediction

---

## ⚙ Hyperparameter Tuning

GridSearchCV was used to find the best value of the **C** parameter for the Linear SVM classifier.

Example:

```python
param_grid = {
    'C': [0.5, 1, 2, 5]
}
```

---

## 📈 Model Performance

| Metric    | Score                                           |
| --------- | ----------------------------------------------- |
| Accuracy  | 91% |
| Precision | 90-91%                                           |
| Recall    | 90-91%                                           |
| F1-Score  | 91%                                              |

---

## 📁 Project Structure

```
Movie-Sentiment-Analysis/

│── app.py
│── Movie_Sentiment_Analysis.ipynb
│── requirements.txt
│── README.md
│── best_linear_svm.pkl
│── tfidf_vectorizer.pkl
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/Movie-Sentiment-Analysis.git
```

Move into the project directory:

```bash
cd Movie-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## 🌐 Deployment

The application can be deployed using:

* Hugging Face Spaces
* Gradio

---

## 👨‍💻 Author

Abdul Ahad

Software Engineering Student

AI & Machine Learning Enthusiast
