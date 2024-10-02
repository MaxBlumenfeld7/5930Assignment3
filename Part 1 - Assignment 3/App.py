from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Define a route for the root URL
@app.route('/')
def home():
    return 'Hello, World!'

# Load the pre-trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    print("Model loaded successfully")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = data['features']
    prediction = model.predict([features])
    return jsonify(prediction=int(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)
