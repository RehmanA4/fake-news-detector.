from flask import Flask, request, jsonify, render_template
import pickle
import re
import string

app = Flask(__name__)

# load saved model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    news = data["news"]

    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned])
    prob = model.predict_proba(vector)[0]

    return jsonify({
        "fake_probability": round(float(prob[0]), 2),
        "real_probability": round(float(prob[1]), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
