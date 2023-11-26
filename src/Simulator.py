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


def board_to_matrix(board):
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
    matrix = board_to_matrix(board)
    return torch.tensor(matrix, dtype=torch.float32)


class ChessSimulation:
    run_counter = 0

    def __init__(self, num_games, player1, player2, stockfish_path, openings_file_path):
        self.num_games = num_games
        self.player1 = player1
        self.player2 = player2
        self.stockfish_path = stockfish_path
        self.results = []
        self.openings = self.load_openings(openings_file_path)
        ChessSimulation.run_counter += 1
        self.current_run_number = ChessSimulation.run_counter

    def run_simulation(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        date_str = datetime.now().strftime("%Y%m%d")

        for game_num in tqdm(range(self.num_games), desc="Simulating Games", unit="game"):
            white, black = (self.player1, self.player2) if game_num % 2 == 0 else (self.player2, self.player1)
            game_result = self.play_game(white, black)
            self.results.append(game_result)

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
    

    def play_game(self, white, black):
        board = chess.Board()
        opening = random.choice(self.openings)
        board.set_fen(opening)
        tensor_states = []
        evaluations = []
        moves = []
        material_advantage_data = []

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = self.get_move(white, board)
            else:
                move = self.get_move(black, board)

            board.push(move)
            tensor_states.append(board_to_tensor(board))
            moves.append(str(move))

            evaluation = self.engine.analyse(board, chess.engine.Limit(time=0.1))
            score = evaluation["score"].white().score(mate_score=10000)
            evaluations.append(score)

            material_advantage = self.calculate_material_advantage(board)
            material_advantage_data.append(material_advantage)

        game_result = {
            "winner": self.get_winner(board.result()),
            "loser": self.get_loser(board.result()),
            "white_pieces": str(white),
            "black_pieces": str(black),
            "moves": ", ".join(moves),
            "evaluations": evaluations,
            "material_advantage": material_advantage_data
        }
        return game_result

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
            result = self.engine.play(board, chess.engine.Limit(time=0.1))
            return result.move

    def get_winner(self, result):
        return "White" if result == "1-0" else "Black" if result == "0-1" else "Draw"

    def get_loser(self, result):
        return "Black" if result == "1-0" else "White" if result == "0-1" else "Draw"

    def write_to_csv(self, file_name):
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
                    "material_advantage"
                ],
            )
            writer.writeheader()
            for game_result in self.results:
                game_result['material_advantage'] = ', '.join(map(str, game_result['material_advantage']))
                writer.writerow(game_result)

    def load_openings(self, file_path):
        with open(file_path, "r") as file:
            return [line.strip() for line in file.readlines()]

# Main execution
if __name__ == "__main__":
    num_simulated_games = 500  # Adjust as needed
    ai_player = AlphaPawn()  # Your AI
    stockfish_path = "../assets/stockfish-windows-x86-64-avx2.exe"  # Replace with your Stockfish path
    openings_database = "../assets/chess_openings.txt"

    simulation = ChessSimulation(num_simulated_games, ai_player, "Stockfish", stockfish_path, openings_database)
    simulation.run_simulation()

    print("Simulation complete. Tensors and game results saved.")
