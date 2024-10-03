from flask import Flask, request, jsonify
import pickle
import os
import numpy as np

app = Flask(__name__)

#function to read model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Set base directory relative to this file's location
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained model
model_path = os.path.join(base_dir, 'models', 'trigram_model.pkl')

# Load pre-trained model using the new model path
ngram_model = load_model(model_path)

@app.route('/')
def home():
    return 'Hello, World! This is the ML model API.'

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Ensure that we received the expected array of features
    try:
        features = data['features']
    except KeyError:
        return jsonify(error="The 'features' key is missing from the request payload."), 400
    
    # Convert features into the right format and make a prediction
    prediction = ngram_model.predict([features])
    
    # Return the prediction
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
