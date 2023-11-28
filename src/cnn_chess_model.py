import os
import threading
import time
import chess
import tensorflow as tf
from tensorflow import keras as tfk
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.layers import Input, Concatenate
from keras.models import Model
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
        # Convolutional pathway for board state
        board_input = Input(shape=(8, 8, 1), name='board_input')
        conv_layer = Conv2D(64, (3, 3), activation="relu")(board_input)
        flatten_layer = Flatten()(conv_layer)

        # Separate input for evaluation score
        eval_input = Input(shape=(1,), name='eval_input')

        # Separate input for game outcome
        outcome_input = Input(shape=(1,), name='outcome_input')

        # Combining the features
        combined = Concatenate()([flatten_layer, eval_input, outcome_input])

        # Dense layers after combining features
        dense_layer = Dense(64, activation="relu")(combined)
        output = Dense(1, activation="linear")(dense_layer)

        # Create the model
        model = Model(inputs=[board_input, eval_input, outcome_input], outputs=output)

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
        # Create dummy arrays for the additional inputs that the model expects.
        # The shapes should match the input shape that the model expects, except for the batch size dimension.
        # Here, we assume processed_board.shape[0] is the batch size.
        dummy_eval_score = np.zeros((processed_board.shape[0], 1))
        dummy_outcomes = np.zeros((processed_board.shape[0], 1))
        
        try:
            # Pass all three inputs to the model's predict method
            predictions = self.model.predict([processed_board, dummy_eval_score, dummy_outcomes], verbose=0)
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
