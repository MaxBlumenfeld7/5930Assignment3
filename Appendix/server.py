from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('5930Assignment3/Part 2/models/trigram_model.pkl', 'rb'))

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
    prediction = model.predict([features])
    
    # Return the prediction
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
