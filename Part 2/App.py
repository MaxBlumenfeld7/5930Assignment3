from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "Running in Flask!"

@app.route('/hello')
def hello():
    return "Hello, Flask!"

@app.route('/greet/<name>')
def greet(name):
    return f"Hello, {name}!"

@app.route('/page')
def page():
    return render_template('page.html', message="This is a simple Flask app")

if __name__ == '__main__':
    app.run(debug=True)