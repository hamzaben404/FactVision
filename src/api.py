import logging
import os
import joblib
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Log INFO level and above
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
    handlers=[
        logging.FileHandler("app.log"),  # Write logs to a file named app.log
        logging.StreamHandler()          # Also print logs to stdout (useful for Docker logs)
    ]
)

logging.info("Starting Flask API with logging enabled.")

# Define model paths inside the container
MODEL_DIR = "/app/models"
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
model_path = os.path.join(MODEL_DIR, "random_forest.pkl")

# Check if model files exist
if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    logging.error("Model files not found. Ensure they are in the 'models' directory.")
    raise FileNotFoundError("Model files not found. Ensure they are in the 'models' directory.")

# Load the model and vectorizer
try:
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.exception("Error loading model or vectorizer:")
    raise e

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint to predict if a news article is fake or real."""
    try:
        data = request.get_json()
        logging.info(f"Received request data: {data}")
        
        if not data or "text" not in data:
            logging.warning("No text provided in the request.")
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # Transform text using TF-IDF vectorizer
        text_vectorized = vectorizer.transform([text])

        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        result = "Fake" if prediction == 1 else "Real"
        
        logging.info(f"Prediction: {result} for input: {text}")

        return jsonify({"prediction": result})
    
    except Exception as ex:
        logging.exception("Exception during prediction:")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Only run in development mode; Gunicorn is used in production.
    app.run(host='0.0.0.0', port=5001, debug=True)
