import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the vectorizer correctly
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl") 

# Load a trained model
model = joblib.load("../models/naive_bayes.pkl") 


# Sample test input
sample_news = ["Breaking news: Scientists discover a new planet with water!"]

# Transform input using TF-IDF
X_sample = vectorizer.transform(sample_news)

# Make prediction
prediction = model.predict(X_sample)

# Print result
print("Prediction:", "Fake" if prediction[0] == 1 else "Real")
