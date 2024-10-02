from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Mock models (replace these with your actual models later)
class MockNGramModel:
    def predict(self, text):
        words = text.split()
        if len(words) >= 2:
            return f"Next word might be: '{words[-1]}'"
        return "Not enough context for prediction"

class MockSentimentModel:
    def predict(self, text):
        positive_words = ['good', 'great', 'excellent', 'amazing']
        negative_words = ['bad', 'poor', 'terrible', 'awful']
        
        words = text.lower().split()
        positive_count = sum(word in positive_words for word in words)
        negative_count = sum(word in negative_words for word in words)
        
        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else:
            return "Neutral"

# Initialize mock models
ngram_model = MockNGramModel()
sentiment_model = MockSentimentModel()

@app.route('/')
def home():
    return "Welcome to the Text Analysis App. Go to /predict to use the predictor."

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        text = data['text']
        
        ngram_prediction = ngram_model.predict(text)
        sentiment_prediction = sentiment_model.predict(text)
        
        return jsonify({
            'ngram': ngram_prediction,
            'sentiment': sentiment_prediction
        })
    else:  # GET request
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)