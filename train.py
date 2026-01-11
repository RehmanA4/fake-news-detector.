import pandas as pd

# Load the datasets
true_news = pd.read_csv("data/True.csv")
fake_news = pd.read_csv("data/Fake.csv")

# Add labels
true_news["label"] = 1   # Real news
fake_news["label"] = 0   # Fake news

# Combine both datasets
data = pd.concat([fake_news, true_news])

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

data["content"] = data["title"] + " " + data["text"]


# Check the result
#print(data.head())
#print(data["label"].value_counts())


import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

sample = "BREAKING!!! Visit https://news.com 😱 India WON 2024!!!"
cleaned = clean_text(sample)

data["content"] = data["content"].apply(clean_text)

#print(data["content"].head())


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)


vectors = vectorizer.fit_transform(data["content"])

#print(vectorizer.get_feature_names_out())
#print(vectors.shape)

from sklearn.model_selection import train_test_split

X = vectors
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

#print("Accuracy:", accuracy)


def predict_news(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]
    
    return {
        "Fake Probability":  round(float(prob[0]), 2),
        "Real Probability": round(float(prob[1]), 2)
    }




import pickle
import os

os.makedirs("model", exist_ok=True)

with open("model/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

