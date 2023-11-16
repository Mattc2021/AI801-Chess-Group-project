from copy import copy, deepcopy
from imp import load_module
import os
import AlphaPawn
import chess.svg
import tkinter as tk
from tkinter import messagebox, ttk
import random
import numpy as np
import threading
import time
from keras import Model as KerasModel

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100
    
}

import tensorflow as tf

import os
import threading
import time
from tensorflow import keras as tfk

from keras.models import load_model
class CNNChessModel:
    def __init__(self):
        self.model = self.load_or_build_cnn()
        self.SAVE_PATH = "C:\\StrategosAI"  # Define the save path
        self.MODEL_NAME = 'saved_model'
       
        threading.Thread(target=self.CNN_autosave, daemon=True).start()

    def build_cnn(self):
        # Build and compile the CNN model using TensorFlow
        model = tfk.Sequential([
            tfk.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(8, 8, 12)),
            tfk.layers.Flatten(),
            tfk.layers.Dense(64, activation='relu'),
            tfk.layers.Dense(4096, activation='linear')  # Using linear activation for raw probabilities
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
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

# Define MCTS algorithm
class MCTS:
    def __init__(self, cnn_model):
        self.cnn_model = cnn_model
        # Other MCTS initialization code

    def select_move(self, board):
        # Use MCTS for move selection
        chosen_move_mcts = self.run_mcts(board)

        # Use Alpha-Beta Pruning with the MCTS-chosen move as the starting point
        chosen_move_ab = self.alpha_beta_pruning(board, chosen_move_mcts)

        return chosen_move_ab

    def run_mcts(self, board):
        # Implement MCTS algorithm here
        # This method should build a tree, perform simulations, and select a move
        # Placeholder implementation:
        best_move_mcts = None
        # ... MCTS logic ...
        return best_move_mcts

    def alpha_beta_pruning(self, board, chosen_move):
        # Implement Alpha-Beta Pruning for move selection, using the MCTS-chosen move
        best_move_ab = None
        alpha = -float('inf')
        beta = float('inf')
        max_val = -float('inf')
        
        legal_moves = [chosen_move] if chosen_move else list(board.legal_moves)  # Start with the MCTS-chosen move if available
        for move in legal_moves:
            board.push(move)
            val = self.min_value(board, alpha, beta, 0)
            board.pop()
            if val > max_val:
                max_val = val
                best_move_ab = move
            alpha = max(alpha, max_val)
        return best_move_ab

    def process_board(self, board):
        # Convert the board state to a format suitable for the CNN input
        board_representation = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_value = PIECE_VALUES[piece.piece_type] * (1 if piece.color == board.turn else -1)
                board_representation.append(piece_value)
            else:
                board_representation.append(0)

        # Reshape the representation to fit the CNN input shape
        processed_board = tf.convert_to_tensor([board_representation], dtype=tf.float32)
        processed_board = tf.reshape(processed_board, (1, 8, 8, 1))
        
        return processed_board

    def max_value(self, board, alpha, beta, depth):
        # Maximizer function for Alpha-Beta Pruning
        if depth == 0 or board.is_game_over():
            # Return evaluation function or utility value
            return self.evaluate(board)
        
        val = -float('inf')
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            board.push(move)
            val = max(val, self.min_value(board, alpha, beta, depth + 1))
            board.pop()
            if val >= beta:
                return val
            alpha = max(alpha, val)
        return val

    def min_value(self, board, alpha, beta, depth):
        # Minimizer function for Alpha-Beta Pruning
        if depth == 0 or board.is_game_over():
            # Return evaluation function or utility value
            return self.evaluate(board)
        
        val = float('inf')
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            board.push(move)
            val = min(val, self.max_value(board, alpha, beta, depth + 1))
            board.pop()
            if val <= alpha:
                return val
            beta = min(beta, val)
        return val
    
    def evaluate(self, board):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }

        # Evaluation weights for different factors
        material_weight = 1.0
        mobility_weight = 0.5
        pawn_structure_weight = 0.3
        king_safety_weight = 0.2
        center_control_weight = 0.4

        score = 0

        # Material advantage
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    score += piece_values[piece.piece_type]
                else:
                    score -= piece_values[piece.piece_type]

        # Mobility - Number of legal moves available
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK  # Switch turns to assess opponent's mobility
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE  # Switch back to original turn
        mobility = white_mobility - black_mobility
        score += mobility_weight * mobility

        # Pawn structure - Evaluation based on pawn structure can be complex
        # Here, a simple evaluation based on the number of pawns present is used
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        pawn_structure = white_pawns - black_pawns
        score += pawn_structure_weight * pawn_structure

        # King safety - Distance of kings from the center
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        white_king_safety = abs(chess.square_file(white_king_square) - 4) + abs(chess.square_rank(white_king_square) - 4)
        black_king_safety = abs(chess.square_file(black_king_square) - 4) + abs(chess.square_rank(black_king_square) - 4)
        king_safety = black_king_safety - white_king_safety
        score += king_safety_weight * king_safety

        # Control of the center - Presence of pieces in the center
        center_control = 0
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    center_control += 1
                else:
                    center_control -= 1
        score += center_control_weight * center_control

        return score



# AlphaPawn class using MCTS and CNN
class AlphaPawn:
    def __init__(self):
        self.cnn_model = CNNChessModel()
        self.mcts = MCTS(self.cnn_model)

    def choose_move(self, board):
        return self.mcts.select_move(board)

class PawnPromotionDialog:
    def __init__(self, parent):
        self.parent = parent
        self.piece_chosen = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Pawn Promotion")
        
        label = tk.Label(self.dialog, text="Choose a piece for promotion:")
        label.pack()

        self.promotion_choice = tk.StringVar()

        pieces = ["Queen", "Rook", "Bishop", "Knight"]
        for piece in pieces:
            rb = tk.Radiobutton(self.dialog, text=piece, variable=self.promotion_choice, value=piece)
            rb.pack(anchor='w')
        
        confirm_button = tk.Button(self.dialog, text="Confirm", command=self.confirm)
        confirm_button.pack()

    def confirm(self):
        self.piece_chosen = self.promotion_choice.get()
        self.dialog.destroy()

class ChessGUI:
    """
    Creates a Graphical User Interface for the chess game using tkinter.
    """
    def __init__(self, root: tk.Tk, piece_images: dict):
        """
        @brief Initialize the Chess GUI.

        @parameter: root (tk.Tk): The tkinter root widget.
        @parameter: piece_images (dict): Dictionary mapping piece symbols to images.
        """
        print("--ChessGUI __init__--")

        self.root = root
        self.board = chess.Board()
        self.ai = AlphaPawn()
        self.piece_images = piece_images
        print("starting thread...")
        self.choose_side()
   
        threading.Thread(target=self.print_turn_periodically, daemon=True).start()
        # Attributes for game state

        self.selected_piece_square = None

        # Canvas for the board
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self.on_square_clicked)

     

        # If the player chose Black, make the AI's first move immediately
        if self.ai_color == chess.WHITE:
            self.ai_move()
            self.board.turn == self.player_color
        else:
            self.draw_board()

        self.eval_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.eval_bar.pack(pady=20)

    def choose_side(self):
        """
        Prompt the user to select a side.
        """
        
        print("--choose_side--")
        response = messagebox.askquestion("Choose Side", "Do you want to play as White?")
        if response == 'yes':
            self.player_color = chess.WHITE
            self.ai_color = chess.BLACK
        else:
            self.player_color = chess.BLACK
            self.ai_color = chess.WHITE

    def get_evaluation(self):
        print("--get_evaluation--")
        # Number of pieces remaining for both players
        self.board2 = deepcopy(self.board)
        white_piece_count = sum(len(self.board2.pieces(piece_type, chess.WHITE)) for piece_type in PIECE_VALUES)
        black_piece_count = sum(len(self.board2.pieces(piece_type, chess.BLACK)) for piece_type in PIECE_VALUES)

        # Factor for mobility based on the number of legal moves available
        white_mobility = len(list(self.board2.legal_moves))
        self.board2.turn = chess.BLACK  # Switch turns to assess opponent's mobility
        black_mobility = len(list(self.board2.legal_moves))
        self.board2.turn = chess.WHITE  # Switch back to original turn

        # Factor for king safety (e.g., distance from the center)
        white_king_square = self.board2.king(chess.WHITE)
        black_king_square = self.board2.king(chess.BLACK)
        white_king_safety = abs(chess.square_file(white_king_square) - 4) + abs(chess.square_rank(white_king_square) - 4)
        black_king_safety = abs(chess.square_file(black_king_square) - 4) + abs(chess.square_rank(black_king_square) - 4)

        # Normalize the evaluations and return a combined score
        evaluation = (white_piece_count - black_piece_count) + 0.2 * (white_mobility - black_mobility) + 0.1 * (white_king_safety - black_king_safety)
        return evaluation


    def update_eval_bar(self):
        print("--update_eval_bar--")
        evaluation = self.get_evaluation()
        normalized_eval = (evaluation + 39) / 78  # Normalize assuming max material difference is 39
        self.eval_bar["value"] = normalized_eval * 100  # Convert to percentage for progress bar

    def print_turn_periodically(self):
        while True:
            print("---")
            print(f"Player's Turn: {self.board.turn==self.player_color}")  # Print the current player's turn
            if (self.player_color==False):
                print("Player's color: Black")
            else:
                print("Player's color: White")
            print("---")
            time.sleep(5)  # Wait for 5 seconds
            

            
    def start_game(self):
        """
        Start the game after the player chooses a side.
        """
        print("--start_game--")
        self.choose_side()
        self.board = chess.Board()
        self.ai = AlphaPawn()

        self.selected_piece_square = None

        # Canvas for the board
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self.on_square_clicked)

     

        # If the player chose Black, make the AI's first move immediately
        if self.ai_color == chess.WHITE:
            self.ai_move()
            self.board.turn == self.player_color
        else:
            self.draw_board()

        self.eval_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.eval_bar.pack(pady=20)


 

    def draw_board(self):
        """
        @brief: Draw the chessboard and place the pieces on the board
        """
        print("--draw_board--")
        
        # Clear the previous canvas content
        self.canvas.delete("all")

        # Draw the squares
        colors = ["#DDBB88", "#AA8844"]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(col*50, (7-row)*50, (col+1)*50, (7-row+1)*50, fill=color)

        # Highlight possible moves if a piece is selected
        if self.selected_piece_square:
            moves = [move.to_square for move in self.board.legal_moves if move.from_square == self.selected_piece_square]
            for move in moves:
                x, y = chess.square_file(move), chess.square_rank(move)
                self.canvas.create_rectangle(x*50, (7-y)*50, (x+1)*50, (7-y+1)*50, fill='green')

        # Place pieces on the board
        for square, piece in self.board.piece_map().items():
            x, y = chess.square_file(square), chess.square_rank(square)
            image = self.piece_images[str(piece)]
            self.canvas.create_image(x*50, (7-y)*50, anchor=tk.NW, image=image)


    def ai_move(self):
        """
        @brief: Execute an AI move on the board.
        """
        
        print("--ai_move--")
        
        move = self.ai.choose_move(self.board)
        self.board.push(move)
        self.draw_board()
        if self.board.is_game_over():
            self.game_over()

    def on_square_clicked(self, event: tk.Event):
        """
        @brief: Handle square click events to make player moves.
    
        @paramter: event (tk.Event): The click event.
        """
        print("--on_square_clicked--")
        
        col = event.x // 50
        row = 7 - event.y // 50
        square = chess.square(col, row)

        if self.selected_piece_square == square:  # Deselect the piece
            print("Deselecting the piece")
            self.selected_piece_square = None
            self.draw_board()  # Redraw the board to remove highlighted valid moves
            return
        if self.selected_piece_square is None:
            print("Selected piece square is None")
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                print("Selected piece belongs to the player's turn")
                self.selected_piece_square = square
                self.draw_board()  # Redraw the board to show highlighted valid moves
                # Highlight valid moves for the selected piece
                print("populating legal moves")
                for move in self.board.legal_moves:
                    
                    if move.from_square == square:
                        x, y = chess.square_file(move.to_square), chess.square_rank(move.to_square)
                        self.canvas.create_oval(x * 50 + 10, (7 - y) * 50 + 10, (x + 1) * 50 - 10, (7 - y + 1) * 50 - 10, fill="blue")
                print("Legal moves populated. Players turn?", self.player_color==self.board.turn)
        else:
            print("Selected piece square is not None")
            move = chess.Move(self.selected_piece_square, square)
            # Check for pawn promotion
            selected_piece = self.board.piece_at(self.selected_piece_square)
            if selected_piece and selected_piece.piece_type == chess.PAWN:
                # Check if the pawn reaches the end of the board for promotion
                if (self.board.turn == chess.WHITE and chess.square_rank(square) == 7) or \
                    (self.board.turn == chess.BLACK and chess.square_rank(square) == 0):
                    # Ask for pawn promotion choice
                    promotion_choice = self.promote_pawn(self.board.turn)
                    move = chess.Move(self.selected_piece_square, square, promotion=promotion_choice)

            # I  nailed it down to this being the explict cause of the system bugging out while playing as black.
            if move in self.board.legal_moves:
                print("Move is legal")
                self.board.push(move)
                self.draw_board()
                if self.board.is_game_over():
                    self.game_over()
                elif self.board.turn == self.ai_color:
                    print("AI's move")
                    self.ai_move()
            else:  # If the move is not legal, just deselect the piece
                print("Move is not legal. Deselecting the piece")
                self.selected_piece_square = None
                self.draw_board()
        # eval bar is causing turn tracking to be off. 
        # self.update_eval_bar()

    def game_over(self):
        """
        @brief: Handle game over scenarios and display the results
        """
        print("--game_over--")
        result = "Draw" if self.board.result() == "1/2-1/2" else "Win for " + ("White" if "1-0" == self.board.result() else "Black")
        messagebox.showinfo("Game Over", f"Game Over! Result: {result}")
        
        # Destroy the canvas and button
        self.canvas.destroy()
        self.eval_bar.destroy()
      
        
        # Restart the game by prompting the player to choose a side again
        self.start_game()
        
    def promote_pawn(self, color: chess.Color) -> chess.PieceType:
        """
        Open a custom dialog to let the user choose a piece for pawn promotion.
        """
        print("--promote_pawn--")
        
        dialog = PawnPromotionDialog(self.root)
        self.root.wait_window(dialog.dialog)
        
        pieces = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }

        chosen_piece = dialog.piece_chosen
        if chosen_piece in pieces:
            return pieces[chosen_piece]
        else:
            # Default to Queen if an invalid choice or the dialog is closed
            return chess.QUEEN




if __name__ == "__main__":
   
    root = tk.Tk()
    root.title("Strategos Chess")

    # Load images
    piece_names = ['wp', 'wn', 'wb', 'wr', 'wq', 'wk', 'bp', 'bn', 'bb', 'br', 'bq', 'bk']
    piece_images = {}

    for name in piece_names:
        image_path = f"./assets/{name}.png"
        original_image = tk.PhotoImage(file=image_path)
        
        # Resize the image to fit the square (assuming 50x50 squares for this example)
        resized_image = original_image.subsample(int(original_image.width() // 50), int(original_image.height() // 50))

        # Map the filename to its corresponding chess.Piece symbol
        symbol = name[1].upper() if name[0] == 'w' else name[1].lower()
        piece_images[symbol] = resized_image

    gui = ChessGUI(root, piece_images)
    root.mainloop()