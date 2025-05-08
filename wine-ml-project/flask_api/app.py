from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load the model using an absolute path
model_path = os.path.join(os.path.dirname(__file__), "../model_training/random_forest_wine_model.pkl")
model = joblib.load(model_path)


# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Wine Quality Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert data to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Return result as JSON
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
