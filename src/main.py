import tkinter as tk
from chess_gui import ChessGUI
import chess.svg
import tkinter as tk
from tkinter import messagebox, ttk
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