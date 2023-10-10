import AlphaPawn
import chess.svg
import tkinter as tk
from tkinter import messagebox
import random

# TODO: Actually create the AI and maybe have different ones for different algos that we can choose between
class AlphaPawn:
    """
    @brief Represents a basic AI for the chess game.
    @brief Currently, the AI chooses a random legal move.
    """
    def __init__(self):
        pass

    # Right now it chooses a random move of all legal move choies
    def choose_move(self, board: chess.board) -> chess.Move:
        """
        @brief: Choose a move for the current board state.

        @parameter: board (chess.board): The current game state.
        @returns: chess.Move: A random legal move.
        """
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)


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

        self.root = root
        self.board = chess.Board()
        self.ai = AlphaPawn()  
        self.ai_color = chess.WHITE

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack(pady=20)

        self.piece_images = piece_images 

        self.draw_board()

        self.button = tk.Button(root, text="Make AI Move", command=self.ai_move)
        self.button.pack(pady=20)

        self.canvas.bind("<Button-1>", self.on_square_clicked)
        self.selected_piece_square = None

    def draw_board(self):
        """
        @brief: Draw the chessboard and place the pieces on the board
        """
        # Draw the squares
        colors = ["#DDBB88", "#AA8844"]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(col*50, (7-row)*50, (col+1)*50, (7-row+1)*50, fill=color)

        # Place pieces on the board
        for square, piece in self.board.piece_map().items():
            x, y = chess.square_file(square), chess.square_rank(square)
            image = self.piece_images[str(piece)]
            self.canvas.create_image(x*50, (7-y)*50, anchor=tk.NW, image=image)

    def ai_move(self):
        """
        @brief: Execute an AI move on the board.
        """
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
        # Convert canvas coordinates to chess board squares
        col = event.x // 50
        row = 7 - event.y // 50
        square = chess.square(col, row)

        if self.selected_piece_square is None:
            if self.board.piece_at(square):
                self.selected_piece_square = square
        else:
            move = chess.Move(self.selected_piece_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.draw_board()
                if self.board.is_game_over():
                    self.game_over()
                elif self.board.turn == self.ai_color:
                    self.ai_move()
            self.selected_piece_square = None

    def game_over(self):
        """
        @brief: Handle game over scenarios and display the results
        """
        result = "Draw" if self.board.result() == "1/2-1/2" else "Win for " + ("White" if "1-0" == self.board.result() else "Black")
        messagebox.showinfo("Game Over", f"Game Over! Result: {result}")
        self.board.reset()
        self.draw_board()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Chess AI with AlphaPawn")

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