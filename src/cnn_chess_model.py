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
    """
    A class that encapsulates a Convolutional Neural Network (CNN) model for chess.
    This class provides functionalities to build, load, save, and make predictions with the CNN model.
    """

    def __init__(self):
        """
        Initialize the CNNChessModel instance.
        Sets up the model save path and either loads an existing model or builds a new one.
        A separate thread for auto-saving the model can be activated if needed.
        """
        self.SAVE_PATH = os.path.join(os.getcwd(), "StrategosCNNModel")
        self.MODEL_NAME = "saved_model"
        self.model = self.load_or_build_cnn()
        # Uncomment the next line to enable auto-saving the model periodically
        # threading.Thread(target=self.CNN_autosave, daemon=True).start()

    def build_cnn(self):
        """
        Build and compile the CNN model using TensorFlow and Keras.
        The model consists of convolutional and dense layers suitable for processing chess board states.

        Returns:
        The compiled CNN model.
        """
        model = tfk.Sequential(
            [
                tfk.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(8, 8, 1)),
                tfk.layers.Flatten(),
                tfk.layers.Dense(64, activation="relu"),
                tfk.layers.Dense(1, activation="linear"),
            ]
        )
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def load_or_build_cnn(self):
        """
        Load an existing CNN model from the save path, or build a new one if none exists.

        Returns:
        The loaded or newly built CNN model.
        """
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)

        model_path = os.path.join(self.SAVE_PATH, self.MODEL_NAME)

        if os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print("Error loading the model:", e)
                print("Building a new model...")
                self.model = self.build_cnn()
        else:
            print(f"No model found in {self.SAVE_PATH}. Building a new model...")
            self.model = self.build_cnn()

        return self.model

    def CNN_autosave(self):
        """
        Periodically saves the CNN model.
        This function is designed to run in a separate thread and saves the model every 15 seconds.
        """
        while True:
            if isinstance(self.model, KerasModel):
                try:
                    self.model.save(os.path.join(self.SAVE_PATH, self.MODEL_NAME))
                except Exception as e:
                    pass
            else:
                print("No Keras model available to save.")
            time.sleep(15)

    def predict(self, processed_board):
        """
        Make predictions using the CNN model on the processed chess board.

        Parameters:
        - processed_board: The preprocessed board data for making predictions.

        Returns:
        The prediction made by the model, or None in case of an error.
        """
        processed_board = np.array(processed_board)
        try:
            predictions = self.model.predict(processed_board, verbose=0)
            return predictions
        except Exception as e:
            print("Error during prediction:", e)
            return None

    def get_model(self):
        """
        Retrieve the current CNN model.

        Returns:
        The CNN model used by the instance.
        """
        return self.model

    def save_model(self, file_path):
        """
        Save the current CNN model to a specified file path.

        Parameters:
        - file_path: The file path where the model should be saved.
        """
        self.model.save(file_path)
