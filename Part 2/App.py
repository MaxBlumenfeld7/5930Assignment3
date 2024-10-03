

from flask import Flask, request, jsonify, render_template
import pickle
import os
import re
from flask_cors import CORS
import logging




app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Set up logging to a file
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', 
                    force=False)  # Ensure this doesn't forcefully override other handlers




#function to read model
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Load pre-trained model
# ngram_model = load_model('5930Assignment3/Part 2/models/trigram_model.pkl')


# Set base directory relative to this file's location
base_dir = os.path.dirname(os.path.abspath(__file__))

# Use os.path.join to dynamically build the path to the model
model_path = os.path.join(base_dir, 'models', 'trigram_model.pkl')

# Load pre-trained model using the new model path
ngram_model = load_model(model_path)

def simple_tokenize(text):
    # This function splits on spaces and punctuation
    return re.findall(r'\w+|[^\w\s]', text.lower())


def predict_with_trigram(text):
    words = simple_tokenize(text)
    
    if len(words) < 2:
        return "Not enough context for prediction"
    
    context = tuple(words[-2:])
    
    cfd = ngram_model['cfd']
    
    print(f"Context: {context}")
    print(f"Number of contexts in model: {len(cfd)}")
    print(f"Sample contexts: {list(cfd.keys())[:5]}")  # Print first 5 contexts
    
    if context in cfd:
        predicted_word = cfd[context].max()
        return f"Predicted word: {predicted_word}"
    else:
        # Try backing off to just the last word
        last_word = words[-1]
        single_word_contexts = [key for key in cfd.keys() if key[1] == last_word]
        if single_word_contexts:
            best_context = max(single_word_contexts, key=lambda k: sum(cfd[k].values()))
            predicted_word = cfd[best_context].max()
            return f"Backed-off prediction: {predicted_word}"
        else:
            return "No prediction available for this context"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Log that a POST request was received
            logging.info("Received POST request for prediction")
            
            # Validate and parse input data
            data = request.get_json(force=True)
            if not data or 'text' not in data:
                logging.error("Invalid input: 'text' key is missing from request")
                return jsonify(error="Invalid input: Expected 'text' in the request."), 400
            
            text = data['text']

            # Validate that 'text' is a string
            if not isinstance(text, str):
                logging.error("Invalid input: 'text' must be a string")
                return jsonify(error="Invalid input: 'text' must be a string."), 400

            # Print the input text as before
            print('Input was:', text)

            # Print logging info for input
            logging.info(f"Input text: {text}")

            # Proceed with prediction as before
            print('Predicting n-gram')
            ngram_prediction = predict_with_trigram(text)

            # Print the n-gram prediction as before
            print(f"N-gram prediction: {ngram_prediction}")
            logging.info(f"N-gram prediction: {ngram_prediction}")

            return jsonify({
                'ngram': ngram_prediction,
            })

        except Exception as e:
            # Print the error as before and log it
            print(f"An error occurred: {str(e)}")
            logging.error(f"An error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 400
    else:  # GET request
        return render_template('predict.html')
    
    
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # print("Current working directory:", os.getcwd())
    # print("Files in the current directory:", os.listdir())
    app.run(debug=True)