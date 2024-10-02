# from flask import Flask, request, jsonify, render_template
# import pickle
# import os
# import transformers
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from flask_cors import CORS
# import re

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# def load_model(filename):
#     with open(filename, 'rb') as file:
#         model = pickle.load(file)
#     return model

# # Load pre-trained models
# ngram_model = load_model('5930Assignment3/Part 2/models/trigram_model.pkl')
# sentiment_model = load_model('5930Assignment3/Part 2/models/sentiment_model.pkl')

# def predict_sentiment(text):
#     result = sentiment_model(text)
#     prediction = result[0]
#     label = prediction['label']
#     return {
#         'sentiment': label,
#     }

# def simple_tokenize(text):
#     # This function splits on spaces and punctuation
#     return re.findall(r'\w+|[^\w\s]', text.lower())



# def predict_with_trigram(text):
#     words = simple_tokenize(text)
    
#     if len(words) < 2:
#         return "Not enough context for prediction"
    
#     context = tuple(words[-2:])
    
#     cfd = ngram_model['cfd']
    
#     print(f"Context: {context}")
#     print(f"Number of contexts in model: {len(cfd)}")
#     print(f"Sample contexts: {list(cfd.keys())[:5]}")  # Print first 5 contexts
    
#     if context in cfd:
#         predicted_word = cfd[context].max()
#         return f"Predicted word: {predicted_word}"
#     else:
#         # Try backing off to just the last word
#         last_word = words[-1]
#         single_word_contexts = [key for key in cfd.keys() if key[1] == last_word]
#         if single_word_contexts:
#             best_context = max(single_word_contexts, key=lambda k: sum(cfd[k].values()))
#             predicted_word = cfd[best_context].max()
#             return f"Backed-off prediction: {predicted_word}"
#         else:
#             return "No prediction available for this context"

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             data = request.get_json(force=True)
#             text = data['text']
#             print('Input was:', text)

#             print('Predicting n-gram')
#             ngram_prediction = predict_with_trigram(text)
#             print(f"N-gram prediction: {ngram_prediction}")

#             print('Predicting sentiment')
#             sentiment_result = predict_sentiment(text)

#             return jsonify({
#                 'ngram': ngram_prediction,
#                 'sentiment': sentiment_result['sentiment'],
#             })
#         except Exception as e:
#             print(f"An error occurred: {str(e)}")
#             return jsonify({'error': str(e)}), 400
#     else:  # GET request
#         return render_template('predict.html')

# @app.route('/')
# def home():
#     return render_template('index.html')

# if __name__ == '__main__':
#     print("Current working directory:", os.getcwd())
#     print("Files in the current directory:", os.listdir())
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import pickle
import os
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Load pre-trained model
ngram_model = load_model('5930Assignment3/Part 2/models/trigram_model.pkl')

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
            data = request.get_json(force=True)
            text = data['text']
            print('Input was:', text)

            print('Predicting n-gram')
            ngram_prediction = predict_with_trigram(text)
            print(f"N-gram prediction: {ngram_prediction}")

            return jsonify({
                'ngram': ngram_prediction,
            })
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 400
    else:  # GET request
        return render_template('predict.html')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    print("Files in the current directory:", os.listdir())
    app.run(debug=True)