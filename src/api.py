from flask import Flask, request, jsonify
import joblib
import os

# Define model paths inside the container
MODEL_DIR = "/app/models"

vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
model_path = os.path.join(MODEL_DIR, "random_forest.pkl")

# Check if models exist
if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    raise FileNotFoundError("Model files not found. Ensure they are in the 'models' directory.")

# Load the model and vectorizer
vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint to predict if a news article is fake or real"""
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]

    # Transform text using TF-IDF vectorizer
    text_vectorized = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    return jsonify({"prediction": "Fake" if prediction == 1 else "Real"})

if __name__ == '__main__':
    # Only run in development mode; Gunicorn is used in production
    app.run(host='0.0.0.0', port=5001, debug=True)
