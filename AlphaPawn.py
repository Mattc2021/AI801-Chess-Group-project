import AlphaPawn
import chess.svg
import tkinter as tk
from tkinter import messagebox
import random
import tkinter.simpledialog as simpledialog

# TODO: Actually create the AI and maybe have different ones for different algos that we can choose between
class AlphaPawn:
    """
    @brief Represents a basic AI for the chess game.
    @brief Currently, the AI chooses a random legal move.
    """
    def __init__(self):
        pass

    # Right now it chooses a random move of all legal move choies
    def choose_move(self, board: chess.Board) -> chess.Move:
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
        self.piece_images = piece_images

        # Attributes for game state
        self.player_color = chess.WHITE
        self.ai_color = chess.BLACK
        self.selected_piece_square = None

        # Canvas for the board
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self.on_square_clicked)

        self.choose_side()

        # If the player chose Black, make the AI's first move immediately
        if self.ai_color == chess.WHITE:
            self.ai_move()
        else:
            self.draw_board()

    def choose_side(self):
        """
        Prompt the user to select a side.
        """
        response = messagebox.askquestion("Choose Side", "Do you want to play as White?")
        if response == 'yes':
            self.player_color = chess.WHITE
            self.ai_color = chess.BLACK
        else:
            self.player_color = chess.BLACK
            self.ai_color = chess.WHITE

    def start_game(self, player_color):
        """
        Start the game after the player chooses a side.
        """
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

        
        col = event.x // 50
        row = 7 - event.y // 50
        square = chess.square(col, row)

        if self.selected_piece_square == square:  # Deselect the piece
            self.selected_piece_square = None
            self.draw_board()  # Redraw the board to remove highlighted valid moves
            return

        if self.selected_piece_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_piece_square = square
                self.draw_board()  # Redraw the board to show highlighted valid moves
                # Highlight valid moves for the selected piece
                for move in self.board.legal_moves:
                    if move.from_square == square:
                        x, y = chess.square_file(move.to_square), chess.square_rank(move.to_square)
                        self.canvas.create_oval(x * 50 + 10, (7 - y) * 50 + 10, (x + 1) * 50 - 10, (7 - y + 1) * 50 - 10, fill="blue")
        else:
            move = chess.Move(self.selected_piece_square, square)
            # Check for pawn promotion
            selected_piece = self.board.piece_at(self.selected_piece_square)
            if selected_piece and selected_piece.piece_type == chess.PAWN:
                if (self.board.turn == chess.WHITE and chess.square_rank(square) == 7) or \
                   (self.board.turn == chess.BLACK and chess.square_rank(square) == 0):
                    move = chess.Move(self.selected_piece_square, square, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                self.board.push(move)
                self.draw_board()
                if self.board.is_game_over():
                    self.game_over()
                elif self.board.turn == self.ai_color:
                    self.ai_move()
            else:  # If the move is not legal, just deselect the piece
                self.selected_piece_square = None
                self.draw_board()

    def game_over(self):
        """
        @brief: Handle game over scenarios and display the results
        """
        result = "Draw" if self.board.result() == "1/2-1/2" else "Win for " + ("White" if "1-0" == self.board.result() else "Black")
        messagebox.showinfo("Game Over", f"Game Over! Result: {result}")
        self.board.reset()
        self.draw_board()

    def promote_pawn(self, color: chess.Color) -> chess.PieceType:
        """
        Open a dialog to let the user choose a piece for pawn promotion.
        """

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