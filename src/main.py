import os
import tkinter as tk
from chess_gui import ChessGUI 

# Main script to initialize and run the chess game GUI using Tkinter
if __name__ == "__main__":
    # Initialize the main window for the chess application
    root = tk.Tk()
    root.title("Strategos Chess")  # Set the title of the main window

    # Define the filenames for the chess pieces
    piece_names = [
        "wp", "wn", "wb", "wr", "wq", "wk",
        "bp", "bn", "bb", "br", "bq", "bk",
    ]
    piece_images = {}  # Dictionary to hold the images for each chess piece

    # Load images for each chess piece
    for name in piece_names:
        # Check for image existence in a predefined path
        first_path = f"./assets/{name}.png"
        if os.path.exists(first_path):
            image_path = first_path
        else:
            # If not found, use an alternative path
            second_path = f"../assets/{name}.png"
            image_path = second_path

        # Load the image using Tkinter's PhotoImage
        original_image = tk.PhotoImage(file=image_path)

        # Resize the image to fit the square size (assuming 50x50 squares)
        resized_image = original_image.subsample(
            int(original_image.width() // 50), int(original_image.height() // 50)
        )

        # Map each filename to its corresponding chess piece symbol
        symbol = name[1].upper() if name[0] == "w" else name[1].lower()
        piece_images[symbol] = resized_image

    # Initialize and display the chess game GUI with the loaded images
    gui = ChessGUI(root, piece_images)
    root.mainloop()  # Start the Tkinter event loop to run the application
