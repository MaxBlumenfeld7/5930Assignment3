from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pickle

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Increase max_iter to allow more iterations for convergence
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save the model to a file using pickle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

