# 📰 Fake News Detector

An end-to-end **Machine Learning powered web application** that predicts whether a given news article is **Fake or Real**, using Natural Language Processing (NLP) techniques.

🌐 **Live Demo:**  
https://fake-news-detector-ds3z.onrender.com

---

## 🚀 Features

- 🔍 Detects fake vs real news articles
- 🧠 Uses **TF-IDF Vectorization** for text representation
- 📊 Probabilistic prediction (Fake vs Real)
- 🌐 Web interface built with **Flask**
- ☁️ Deployed on **Render (Free Tier)**

---

## 🛠️ Tech Stack

### Backend
- Python
- Flask
- Scikit-learn
- Pandas
- NumPy

### Machine Learning
- TF-IDF Vectorizer
- Logistic Regression
- Binary Text Classification

### Deployment
- Render
- Gunicorn

---

## 📂 Project Structure

fake-news-detector/
├── app.py
├── train.py
├── requirements.txt
├── model/
│ ├── fake_news_model.pkl
│ └── tfidf_vectorizer.pkl
├── templates/
│ └── index.html
└── README.md



---

## 🧠 How It Works

1. User enters a news sentence/article
2. Text is cleaned and preprocessed
3. TF-IDF converts text into numerical vectors
4. Trained ML model predicts:
   - Fake Probability
   - Real Probability
5. Result is displayed on the web interface

⚠️ **Note:**  
The model is probabilistic and predicts likelihood, not absolute truth.

---

## 🧪 Example Inputs

**Likely Real:**
Reuters reported that the government said inflation eased in December.


**Likely Fake:**
Scientists confirm humans can now live without oxygen for 30 minutes.



---

## 📦 Installation (Local Setup)

``bash
pip install -r requirements.txt
python app.py

Open browser:
http://127.0.0.1:5000

☁️ Deployment

#This project is deployed using Render (Free Tier).

#Service sleeps during inactivity

#Automatically wakes up on request

#First request may take ~30–60 seconds

📈 Future Improvements

#Improve dataset diversity

#Reduce fake-bias in predictions

#Add explanation for predictions

#Support multi-language news

#Add user authentication

👤 Author
MOHAMMAD ABDUL REHMAN
ATRIFICIAL INTELLIGENCE AND DATA SCIENCE
Aspiring Data Science & AI Engineer

GitHub: https://github.com/RehmanA4


SCREENSHOTS FROM MY PROJRCT:
IF THE NEWS IS REAL:
<img width="1911" height="962" alt="Image" src="https://github.com/user-attachments/assets/fe804bab-fb00-4db7-b3ea-6a4de8e34a92" />

IF THE NEWS IS FAKE:
<img width="1919" height="959" alt="Image" src="https://github.com/user-attachments/assets/1336422f-bb3f-4650-92f8-34c4fe93061a" />

⭐ Acknowledgements

#ISOT Fake News Dataset

#Scikit-learn Documentation

#Render Deployment Platform
