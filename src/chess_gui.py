import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import tkinter 
import chess
import chess.svg
from alpha_pawn import AlphaPawn
import threading
import time
from copy import deepcopy
import traceback
class ChessGUI:
    """
    Creates a Graphical User Interface for the chess game using tkinter.
    This GUI supports playing chess against an AI, displaying the board,
    handling user inputs, and showing the game's progress.
    """
    def __init__(self, root: tk.Tk, piece_images: dict):
        """
        Initialize the Chess GUI.

        Parameters:
        - root (tk.Tk): The tkinter root widget.
        - piece_images (dict): Dictionary mapping piece symbols to images.
        """
        print("--ChessGUI __init__--")
        
        self.eval_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.root = root
        self.board = chess.Board()
        self.ai = AlphaPawn()
        self.piece_images = piece_images
        self.PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
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
            

        self.eval_bar.pack(pady=20)
        threading.Thread(target=self.update_eval_bar_periodically, daemon=True).start()

    def choose_side(self):
        """
        Prompt the user to select a side at the beginning of the game.
        Allows the user to play as either White or Black.
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
        """
        Calculate a simple evaluation score for the current chess board state.
        Considers the number of pieces, mobility, and king safety.

        Returns:
        - evaluation (float): The calculated evaluation score for the current board.
        """
        print("--get_evaluation--")
        
        # Number of pieces remaining for both players
        self.board2 = deepcopy(self.board)
        white_piece_count = sum(len(self.board2.pieces(piece_type, chess.WHITE)) for piece_type in self.PIECE_VALUES)
        black_piece_count = sum(len(self.board2.pieces(piece_type, chess.BLACK)) for piece_type in self.PIECE_VALUES)

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
        """
        Update the evaluation progress bar based on the current board state.
        Normalizes the evaluation score to fit the progress bar's scale.
        """
        print("--update_eval_bar--")
        try:
            evaluation = self.get_evaluation()
            normalized_eval = (evaluation + 39) / 78  # Normalize assuming max material difference is 39
            self.eval_bar["value"] = normalized_eval * 100  # Convert to percentage for progress bar
        except Exception as e:
            print("!!! Progress Bar Update Failed !!!")
            traceback.print_exc()  # Print the exception traceback
            
    def update_eval_bar_periodically(self):
        """
        Periodically update the evaluation progress bar.
        This runs in a separate thread to continuously update the UI.
        """
        while True:
            self.update_eval_bar()
            time.sleep(5)  # Update the progress bar every 5 seconds
        
    def print_turn_periodically(self):
        """
        Print information about the current turn periodically.
        This includes the player's turn and color. Runs in a separate thread.
        """
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
        Start a new game of chess.
        Resets the board, selects sides, and prepares the UI for a new game.
        """
        print("--start_game--")
        self.choose_side()
        self.board = chess.Board()
        self.ai = AlphaPawn()

        self.selected_piece_square = None

        # Canvas for the board
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack(pady=20)
        self.canvas.bind("<Button-1>", self.on_square_clicked)

        # Create or reset the progress bar
        self.create_or_reset_progress_bar()

        # If the player chose Black, make the AI's first move immediately
        if self.ai_color == chess.WHITE:
            self.ai_move()
            self.board.turn == self.player_color
        else:
            self.draw_board()

    def create_or_reset_progress_bar(self):
        """
        Create or reset the evaluation progress bar.
        If the progress bar already exists, it is reset to 0%. If not, a new progress bar is created.
        This is used to reflect the current state of the game in terms of piece advantage.
        """
        # Check if the progress bar already exists
        if hasattr(self, 'eval_bar') and self.eval_bar:
            # If it exists, reset it to 0%
            self.eval_bar.destroy()
            self.eval_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
            self.eval_bar.pack(pady=20)
        else:
            # Otherwise, create a new progress bar
            self.eval_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
            self.eval_bar.pack(pady=20)


    def draw_board(self):
        """
        Draw the chessboard and place the pieces on the board.
        The board is drawn as an 8x8 grid with alternating colored squares.
        Pieces are placed on their respective squares according to the current board state.
        If a piece is selected, possible moves for that piece are highlighted.
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
        Execute the AI's move in the game.
        The AI selects a move based on the current board state and applies it to the board.
        After the move, the board is redrawn, and the game state is checked for a game over condition.
        The evaluation bar is also updated to reflect the new state.
        """
        print("--ai_move--")
        move = self.ai.choose_move(self.board)
        self.board.push(move)
        self.draw_board()
        if self.board.is_game_over():
            self.game_over()
        self.update_eval_bar() 
        
    def on_square_clicked(self, event: tk.Event):
        """
        Handle square click events to make player moves.

        Parameters:
        - event (tk.Event): The click event, containing the coordinates of the mouse click.
        This method determines which square was clicked and either selects a piece, makes a move,
        or deselects a piece based on the game's current state and the rules of chess.
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
        Handle the end of the game, displaying the result and resetting the game state.
        The result of the game (win, loss, or draw) is displayed to the user.
        The board and evaluation bar are reset, and the game is restarted.
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
        Open a dialog to let the user choose a piece for pawn promotion.

        Parameters:
        - color (chess.Color): The color of the pawn being promoted.

        Returns:
        - chess.PieceType: The type of chess piece chosen for promotion.
        This dialog allows the user to choose between a Queen, Rook, Bishop, or Knight.
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
        """
        Confirm the selected piece for promotion.
        Closes the dialog and records the selected piece type for pawn promotion.
        """
        self.piece_chosen = self.promotion_choice.get()
        self.dialog.destroy()