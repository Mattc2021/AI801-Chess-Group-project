import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from cnn_chess_model import CNNChessModel
from datetime import date

def load_data(file_names):
    all_states = []
    all_evaluations = []

    for file_name in file_names:
        try:
            data = torch.load(file_name)
            states, evaluations = preprocess_data(data)
            all_states.extend(states)
            all_evaluations.extend(evaluations)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    return np.array(all_states), np.array(all_evaluations)

def one_hot_encode_board(board_state):
    """One-hot encode the chess board state."""
    piece_to_index = {
        0: 0,   # Empty square
        1: 1,   # White Pawn
        -1: 2,  # Black Pawn
        2: 3,   # White Knight
        -2: 4,  # Black Knight
        3: 5,   # White Bishop
        -3: 6,  # Black Bishop
        4: 7,   # White Rook
        -4: 8,  # Black Rook
        5: 9,   # White Queen
        -5: 10, # Black Queen
        6: 11,  # White King
        -6: 12, # Black King
    }
    one_hot_board = np.zeros((8, 8, 13), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            piece = board_state[i][j]
            one_hot_board[i, j, piece_to_index[piece]] = 1
    return one_hot_board

def augment_data(board_states, evaluations):
    """Apply data augmentation like rotation or mirroring."""
    augmented_data = []
    augmented_evaluations = []
    for state, eval in zip(board_states, evaluations):
        # Add original state
        augmented_data.append(state)
        augmented_evaluations.append(eval)
        # Add rotated states
        for _ in range(3):
            state = np.rot90(state)
            augmented_data.append(state)
            augmented_evaluations.append(eval)  # Append the same evaluation
        # Add mirrored state
        mirrored_state = np.flip(state, axis=1)
        augmented_data.append(mirrored_state)
        augmented_evaluations.append(eval)  # Append the same evaluation
    return augmented_data, augmented_evaluations

def preprocess_data(data):
    """Preprocess the chess board states."""
    board_states = data['states']
    evaluations = data['evaluations']

    # Normalize evaluations
    evaluations = (np.array(evaluations) + 1000) / 2000

    # Preprocess each board state
    processed_states = [one_hot_encode_board(state.numpy()) for state in board_states]

    # Augment data and replicate evaluations accordingly
    processed_states, processed_evaluations = augment_data(processed_states, evaluations)

    return np.array(processed_states, dtype=np.float32), np.array(processed_evaluations, dtype=np.float32)

class ChessDataGenerator(Sequence):
    def __init__(self, X, y, batch_size):
        self.X, self.y = X, y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

def train_model(X_train, y_train, X_val, y_val):
    cnn_model = CNNChessModel().build_cnn()

    batch_size = 32
    epochs = 10

    training_generator = ChessDataGenerator(X_train, y_train, batch_size)
    validation_generator = ChessDataGenerator(X_val, y_val, batch_size)

    cnn_model.fit(training_generator, epochs=epochs, validation_data=validation_generator)

    return cnn_model

if __name__ == '__main__':
    tensor_files = [os.path.join('../assets/chess_position_tensors', f) for f in os.listdir('../assets/chess_position_tensors') if f.endswith('.pt')]
    X, y = load_data(tensor_files)
    # TODO: Fix Training Errors
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_model = train_model(X_train, y_train, X_val, y_val)
    
    model_save_path = f'../model_variations/AlphaPawn_{str(date.today())}'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    trained_model.save(model_save_path)
