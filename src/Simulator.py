from alpha_pawn import AlphaPawn
import chess
import numpy as np
import torch
import chess.engine
import csv
from tqdm import tqdm
from datetime import date
import random


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
    def __init__(self, num_games, player1, player2, stockfish_path, openings_file_path):
        self.num_games = num_games
        self.player1 = player1
        self.player2 = player2
        self.stockfish_path = stockfish_path
        self.results = []
        self.openings = self.load_openings(openings_file_path)

    def run_simulation(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        for game_num in tqdm(
            range(self.num_games), desc="Simulating Games", unit="game"
        ):
            white, black = (
                (self.player1, self.player2)
                if game_num % 2 == 0
                else (self.player2, self.player1)
            )
            tensor_states, evaluations = self.play_game(white, black)
            self.save_tensors(tensor_states, game_num, evaluations)

        self.engine.quit()
        self.write_to_csv("../assets/chess_simulation_results.csv")

    def play_game(self, white, black):
        board = chess.Board()
        opening = random.choice(self.openings)
        board.set_fen(opening)
        tensor_states = []
        evaluations = []  # List to store Stockfish evaluations
        moves = []

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = self.get_move(white, board)
            else:
                move = self.get_move(black, board)

            board.push(move)
            tensor_states.append(board_to_tensor(board))
            moves.append(str(move))

            # Get Stockfish evaluation for the current board state
            evaluation = self.engine.analyse(board, chess.engine.Limit(time=0.1))
            score = (
                evaluation["score"].white().score(mate_score=10000)
            )  # Convert to a consistent perspective
            evaluations.append(score)

        game_result = {
            "winner": self.get_winner(board.result()),
            "loser": self.get_loser(board.result()),
            "white_pieces": str(white),
            "black_pieces": str(black),
            "moves": ", ".join(moves),
            "evaluations": evaluations,
        }
        self.results.append(game_result)

        return tensor_states, evaluations

    def get_move(self, player, board):
        if isinstance(player, AlphaPawn):
            return player.choose_move(board)
        else:
            result = self.engine.play(
                board, chess.engine.Limit(time=0.1)
            )  # Reduced time for faster simulation
            return result.move

    def get_winner(self, result):
        if result == "1-0":
            return "White"
        elif result == "0-1":
            return "Black"
        else:
            return "Draw"

    def get_loser(self, result):
        if result == "1-0":
            return "Black"
        elif result == "0-1":
            return "White"
        else:
            return "Draw"

    def save_tensors(self, tensor_states, game_num, evaluations):
        file_name = f"../assets/chess_position_tensors/chess_game_{game_num}_states_{str(date.today())}.pt"
        torch.save({"states": tensor_states, "evaluations": evaluations}, file_name)
        print(f"Saved {len(tensor_states)} tensors and evaluations to {file_name}")

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
                ],
            )
            writer.writeheader()
            for game_result in self.results:
                writer.writerow(game_result)

    def load_openings(self, file_path):
        with open(file_path, "r") as file:
            openings = [line.strip() for line in file.readlines()]
        return openings


if __name__ == "__main__":
    num_simulated_games = 5  # Adjust the number of games as needed
    ai_player = AlphaPawn()  # Your AI
    stockfish_path = "../assets/stockfish-windows-x86-64-avx2.exe"  # Replace with your Stockfish path
    openings_database = "../assets/chess_openings.txt"

    simulation = ChessSimulation(
        num_simulated_games, ai_player, "Stockfish", stockfish_path, openings_database
    )
    simulation.run_simulation()

    print("Simulation complete. Tensors and game results saved.")