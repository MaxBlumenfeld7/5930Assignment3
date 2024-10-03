

import pickle  # Import the pickle module for loading the model

try:
    # Attempt to open the serialized model file in binary read mode
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)  # Load the model from the file using pickle
        print("Model loaded successfully")  # Print confirmation message
except Exception as e:
    # If an error occurs during loading, print an error message with details
    print(f"Error loading model: {e}")
