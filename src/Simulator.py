from alpha_pawn import AlphaPawn
import chess
import numpy as np
import torch
import chess.engine
import csv
from tqdm import tqdm
from datetime import datetime
import random
from utils import PIECE_VALUES
import graph_utils
import pandas as pd
from cnn_chess_model import CNNChessModel
import torch.nn as nn

class ChessSimulation:
    """
    A class for simulating chess games, analyzing game results, and generating training data for a neural network model.
    """
    run_counter = 0 # Static variable to keep track of simulation runs
    def __init__(self, num_games, player1, player2, stockfish_path, openings_file_path):
        """
        Initialize the ChessSimulation instance.

        Parameters:
        - num_games: Number of games to simulate.
        - player1, player2: The two players (can be AI or Stockfish).
        - stockfish_path: Path to the Stockfish engine executable.
        - openings_file_path: Path to a file containing chess openings in FEN format.
        """
        self.num_games = num_games
        self.player1 = player1
        self.player2 = player2
        self.stockfish_path = stockfish_path
        self.results = []
        self.openings = self.load_openings(openings_file_path)
        ChessSimulation.run_counter += 1
        self.current_run_number = ChessSimulation.run_counter
        self.cnn_model = CNNChessModel()

    def run_simulation(self):
        """
        Run the chess simulation, playing the specified number of games and generating training data.

        Returns:
        - all_training_data: A collection of training data generated from the simulated games.
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        date_str = datetime.now().strftime("%Y%m%d")

        # Initialize an empty list to store the number of moves for each game
        move_counts = []
        all_training_data = []

        for game_num in tqdm(range(self.num_games), desc="Simulating Games", unit="game"):
            white, black = (self.player1, self.player2) if game_num % 2 == 0 else (self.player2, self.player1)
            game_result, game_training_data = self.play_game(white, black, game_num)
            self.results.append(game_result)
            all_training_data.extend(game_training_data)

            # Append the number of moves to the move_counts list
            move_counts.append(game_result['move_count'])

        self.engine.quit()
        self.write_to_csv("../assets/chess_simulation_results.csv")

        # Convert the material advantages and evaluations to DataFrames
        results_df = pd.DataFrame(self.results)
        # Convert the material advantages and evaluations to DataFrames
        material_advantages_df = pd.DataFrame({'material_advantage': [result['material_advantage'] for result in self.results]})
        evaluations_df = pd.DataFrame({'evaluations': [result['evaluations'] for result in self.results]})

        # Ensure material advantages are not stored as strings
        material_advantages_df['material_advantage'] = material_advantages_df['material_advantage'].apply(lambda x: list(map(int, x.split(', '))) if isinstance(x, str) else x)

        # Call the plotting functions with DataFrames
        graph_utils.plot_material_advantage_over_time(material_advantages_df, self.current_run_number, date_str)
        graph_utils.plot_position_evaluation_over_time(evaluations_df, self.current_run_number, date_str)

        graph_utils.plot_win_loss_distribution(results_df, self.current_run_number, date_str)

        # After all games have been simulated, extract the move counts
        move_counts = [result['move_count'] for result in self.results]
        print(move_counts)

        # Call the plotting function from graph_utils with the move counts
        graph_utils.plot_game_lengths(move_counts, self.current_run_number, date_str)

        return all_training_data
    
    def play_game(self, white, black, game_num):
        """
        Play a single game of chess between two players.

        Parameters:
        - white, black: The two players, with white playing first.
        - game_num: The game number in the series of simulations.

        Returns:
        - game_result: A dictionary containing details about the game outcome and statistics.
        - training_data: Training data generated from the game for neural network training.
        """
        board = chess.Board()
        opening = random.choice(self.openings)
        board.set_fen(opening)
        tensor_states = []
        evaluations = []
        moves = []
        material_advantage_data = []
        move_count = 0
        training_data = []

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = self.get_move(white, board)
            else:
                move = self.get_move(black, board)

            board.push(move)
            tensor_state = board_to_tensor(board)
            tensor_states.append(tensor_state)
            # Assume some method to determine the target value for this state
            target_value = self.determine_target_value(board)
            training_data.append((tensor_state, target_value))
            moves.append(str(move))

            evaluation = self.engine.analyse(board, chess.engine.Limit(time=0.1))
            score = evaluation["score"].white().score(mate_score=10000)
            evaluations.append(score)

            material_advantage = self.calculate_material_advantage(board)
            material_advantage_data.append(material_advantage)

            move_count += 1

        self.save_tensors(tensor_states, game_num)

        # Determine the outcome of the game and assign it to the training data
        outcome = self.determine_game_outcome(board)
        training_data = [(tensor_state, outcome) for tensor_state, _ in training_data]

        game_result = {
            "winner": self.get_winner(board.result()),
            "loser": self.get_loser(board.result()),
            "white_pieces": str(white),
            "black_pieces": str(black),
            "moves": ", ".join(moves),
            "evaluations": evaluations,
            "material_advantage": material_advantage_data,
            "move_count": move_count
        }
        return game_result, training_data

    def calculate_material_advantage(self, board):
        advantage = 0
        for piece_type in PIECE_VALUES:
            advantage += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
            advantage -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
        return advantage

    def get_move(self, player, board):
        if isinstance(player, AlphaPawn):
            return player.choose_move(board)
        else:
            # Decide whether to apply a human-like behavior
            if random.random() < 0.1:  # 20% chance to apply human-like behavior
                # Randomly choose which human-like method to apply
                method_choice = random.choice(['limit_depth', 'random_top_moves', 'simulate_error'])

                if method_choice == 'limit_depth':
                    depth = random.choice([2, 3, 4])  # Randomly limit depth
                    result = self.engine.play(board, chess.engine.Limit(depth=depth))

                elif method_choice == 'random_top_moves':
                    info = self.engine.analyse(board, chess.engine.Limit(depth=10))
                    top_moves = info['pv'][:3]  # Get top 3 moves
                    move = random.choice(top_moves)  # Choose one randomly
                    result = self.engine.play(board, chess.engine.Limit(moves=[move]))

                elif method_choice == 'simulate_error':
                    if random.random() < 0.05:  # Chance of making a random move
                        legal_moves = list(board.legal_moves)
                        move = random.choice(legal_moves)
                    else:
                        result = self.engine.play(board, chess.engine.Limit(time=0.1))
                    return move

            else:
                # Play a standard move
                result = self.engine.play(board, chess.engine.Limit(time=0.1))

            return result.move

    def get_winner(self, result):
        return "White" if result == "1-0" else "Black" if result == "0-1" else "Draw"

    def get_loser(self, result):
        return "Black" if result == "1-0" else "White" if result == "0-1" else "Draw"

    def determine_game_outcome(self, board):
        # Checks the game outcome and returns a value representing it
        if board.is_checkmate():
            # Return 1.0 if White wins, -1.0 if Black wins
            return 1.0 if board.turn == chess.BLACK else -1.0
        elif board.is_game_over():
            # Return 0 for a draw
            return 0.0
    
    def determine_target_value(self, board):
        # Simple material advantage calculation
        piece_values = PIECE_VALUES
        material_advantage = 0
        for piece_type, value in piece_values.items():
            white_pieces = len(board.pieces(piece_type, chess.WHITE))
            black_pieces = len(board.pieces(piece_type, chess.BLACK))
            material_advantage += (white_pieces - black_pieces) * value

        return material_advantage


    def write_to_csv(self, file_name):
        """
        Write the results of the games to a CSV file.

        Parameters:
        - file_name: The name of the CSV file to write to.
        """
        with open(file_name, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "winner",
                    "loser",
                    "white_pieces",
                    "black_pieces",
                    "moves",
                    "evaluations",
                    "material_advantage",
                    "move_count"
                ],
            )
            writer.writeheader()
            for game_result in self.results:
                game_result['material_advantage'] = ', '.join(map(str, game_result['material_advantage']))
                writer.writerow(game_result)

    def load_openings(self, file_path):

        with open(file_path, "r") as file:
            return [line.strip() for line in file.readlines()]

    def save_tensors(self, tensor_states, game_num):
        tensor_file_path = f"../assets/chess_position_tensors/game_{game_num}_run_{self.current_run_number}_{datetime.now().strftime('%Y%m%d')}.pt"
        torch.save(tensor_states, tensor_file_path)
        print(f"Tensors saved for game {game_num} in run {self.current_run_number}")


def train_model(model, training_data):
    """
    Train the given CNN model using the specified training data.

    Parameters:
    - model: The CNN model to be trained.
    - training_data: Data to train the model on.
    """

    if not training_data:
        print("No training data available.")
        return
    # Convert training data to the appropriate format if necessary
    input_states, target_values = zip(*training_data)
    input_states = np.array(input_states)
    target_values = np.array(target_values)

    keras_model = model.get_model()  # Get the Keras model
    history = keras_model.fit(input_states, target_values, epochs=10, batch_size=32, validation_split=0.2)

    return history.history['loss'], history.history.get('val_loss', [])

def board_to_matrix(board):
    """
    Convert a chess board state to a matrix representation.

    Parameters:
    - board: The chess board state.

    Returns:
    - matrix (np.array): Matrix representation of the board.
    """
    piece_to_value = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }
    matrix = np.zeros((8, 8), dtype=int)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_to_value[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            matrix[chess.square_rank(square), chess.square_file(square)] = value

    return matrix


def board_to_tensor(board):
    """
    Convert a chess board state to a tensor representation suitable for CNN input.

    Parameters:
    - board: The chess board state.

    Returns:
    - tensor (torch.tensor): Tensor representation of the board.
    """
    matrix = board_to_matrix(board)
    return torch.tensor(matrix, dtype=torch.float32)


# Main execution
if __name__ == "__main__":
    # Configuration and instantiation of ChessSimulation and subsequent operations
    num_simulated_games = 1  # Adjust as needed
    ai_player = AlphaPawn()  # Your AI
    # Initialize the Mean Squared Error loss function
    loss_function = nn.MSELoss()

    stockfish_path = "../assets/stockfish-windows-x86-64-avx2.exe"  # Replace with your Stockfish path
    openings_database = "../assets/chess_openings.txt"

    simulation = ChessSimulation(num_simulated_games, "Stockfish", "Stockfish", stockfish_path, openings_database)
    training_data = simulation.run_simulation()

    print(training_data)

    # Assuming you have defined your model, optimizer, and loss function
    training_loss, validation_loss = train_model(simulation.cnn_model, training_data)

    graph_utils.plot_loss_over_epochs(training_loss, datetime.now().strftime('%Y%m%d'), validation_loss,)

    # Save the trained model if necessary
    simulation.cnn_model.save_model("./StrategosCNNModel/saved_model")

    print("Simulation complete. Tensors and game results saved.")
