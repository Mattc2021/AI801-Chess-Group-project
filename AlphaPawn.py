from copy import copy, deepcopy
import AlphaPawn
import chess.svg
import tkinter as tk
from tkinter import messagebox, ttk
import random
import numpy as np
import threading
import time

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100
    
}

import tensorflow.python.keras as tfk
import tensorflow as tf

class CNNChessModel:
    def __init__(self):
        self.model = self.build_cnn()

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


# Define MCTS algorithm
class MCTS:
    def __init__(self, cnn_model):
        self.cnn_model = cnn_model
        # Other MCTS initialization code

    def select_move(self, board):
        # Use the CNN to guide move selection in MCTS
        legal_moves = list(board.legal_moves)
        move_probabilities = {}
        
        for move in legal_moves:
            # Generate board after making a move
            board.push(move)
            
            processed_board = self.process_board(board)
            
            # Get move probabilities from the CNN
            # Assuming processed_board shape is (1, 8, 8, 1) and model expects the last axis to be 12
            processed_board = tf.concat([processed_board] * 12, axis=-1)
            model = self.cnn_model.model
            move_prob = model(processed_board)

            # Ensure probabilities are 1D and sum to 1
            move_prob = np.squeeze(move_prob)
            move_prob = move_prob / np.sum(move_prob)


            move_probabilities[move] = move_prob
            
            # Undo the move for the next iteration
            board.pop()
        
        # Choose the move based on probabilities, perhaps by using some MCTS policy
        chosen_move = self.choose_based_on_probabilities(move_probabilities)
        return chosen_move

    def process_board(self, board):
        # Convert the board state to a format suitable for the CNN input
        board_representation = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                #print(set(PIECE_VALUES.keys()))
                #print(piece.piece_type)
                piece_value = PIECE_VALUES[piece.piece_type] * (1 if piece.color == board.turn else -1)
                board_representation.append(piece_value)
            else:
                board_representation.append(0)

        # Reshape the representation to fit the CNN input shape
        processed_board = tf.convert_to_tensor([board_representation], dtype=tf.float32)
        processed_board = tf.reshape(processed_board, (1, 8, 8, 1))  # Reshape the tensor

        
        return processed_board


    def choose_based_on_probabilities(self, move_probabilities):
        legal_moves = list(move_probabilities.keys())

        if len(legal_moves) == 0:
            return self.choose_default_move()  # No legal moves; choose a default move

        adjusted_probabilities = {}
        for move, probs in move_probabilities.items():
            total_prob = sum(probs)
            normalized_probs = [p / total_prob for p in probs] if total_prob > 0 else [1 / len(probs)] * len(probs)
            adjusted_probabilities[move] = sum(normalized_probs)

        # Use NumPy to normalize the probabilities
        total_prob = sum(adjusted_probabilities.values())
        normalized_probabilities = {move: prob / total_prob for move, prob in adjusted_probabilities.items()}

        # Choose a move with the adjusted probabilities
        chosen_move = np.random.choice(list(normalized_probabilities.keys()), p=list(normalized_probabilities.values()))

        return chosen_move


    def choose_default_move(self):
        legal_moves = list(self.board.legal_moves)
        return np.random.choice(legal_moves)



# AlphaPawn class using MCTS and CNN
class AlphaPawn:
    def __init__(self):
        self.cnn_model = CNNChessModel()
        self.mcts = MCTS(self.cnn_model)

    def choose_move(self, board):
        return self.mcts.select_move(board)



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
            
    def start_game(self, player_color):

        """
        Start the game after the player chooses a side.
        """
        
        print("--start_game--")
        
        self.player_color = player_color
        self.ai_color = chess.WHITE if player_color == chess.BLACK else chess.BLACK
        self.side_window.destroy()


        # Rest of the initialization here...
        self.board = chess.Board()
        self.ai = AlphaPawn()

        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack(pady=20)
        self.draw_board()

        self.button = tk.Button(self.root, text="Make AI Move", command=self.ai_move)
        self.button.pack(pady=20)

        self.canvas.bind("<Button-1>", self.on_square_clicked)
        self.selected_piece_square = None
        

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
        self.board.reset()
        self.draw_board()

    def promote_pawn(self, color: chess.Color) -> chess.PieceType:
        """
        Open a dialog to let the user choose a piece for pawn promotion.
        """
        print("--promote_pawn--")
        
        pieces = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }

        choice = tk.simpledialog.askstring("Pawn Promotion", "Choose a piece (Queen, Rook, Bishop, Knight):", parent=self.root)
        if choice in pieces:
            return pieces[choice]
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