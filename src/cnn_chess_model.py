import os
import threading
import time
import chess
import tensorflow as tf
from tensorflow import keras as tfk
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras import Model as KerasModel
import numpy as np



class CNNChessModel:
    def __init__(self):
        self.model = self.load_or_build_cnn()
        self.SAVE_PATH = "C:\\StrategosAI"  # Define the save path
        self.MODEL_NAME = 'saved_model'
       
        threading.Thread(target=self.CNN_autosave, daemon=True).start()

    def build_cnn(self):
        # Build and compile the CNN model using TensorFlow
        model = tfk.Sequential([
            tfk.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 1)),  # Modified input shape
            tfk.layers.Flatten(),
            tfk.layers.Dense(64, activation='relu'),
            tfk.layers.Dense(1, activation='linear')  # Single output unit for regression
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def load_or_build_cnn(self):
        # Check if the directory and model file exist
        self.SAVE_PATH = "C:\\StrategosAI"
        self.MODEL_NAME = 'saved_model'
        if os.path.exists(os.path.join(self.SAVE_PATH, self.MODEL_NAME)):
            # Load the model if the file exists
            try:
                loaded_model = load_model(os.path.join(self.SAVE_PATH, self.MODEL_NAME))
                print("Model loaded successfully.")
                return loaded_model
            except Exception as e:
                print("Error loading the model:", e)
                print("Building a new model...")
                return self.build_cnn()
        else:
            print(f"No model found in {self.SAVE_PATH}. Building a new model...")
            return self.build_cnn()
        
    def CNN_autosave(self):
        while True:
            # Save the model if it's available and is a Keras model
            if isinstance(self.model, KerasModel):
                try:
                    self.model.save(os.path.join(self.SAVE_PATH, self.MODEL_NAME))
                    print("Model saved successfully.")
                except Exception as e:
                    print("Error saving model:", e)
            else:
                print("No Keras model available to save.")
                
            time.sleep(15)  # Wait for 15 seconds



    
    def predict(self, processed_board):
        # Ensure processed_board is a numpy array
        processed_board = np.array(processed_board)
        
        # Make predictions using the model
        try:
            predictions = self.model.predict(processed_board)
            return predictions  # Return the predictions
        except Exception as e:
            print("Error during prediction:", e)
            return None  # Return None in case of an error