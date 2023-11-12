from AlphaPawn import AlphaPawn
import chess.svg
import chess.engine
import csv
from tqdm import tqdm

class ChessSimulation:
    def __init__(self, num_games, player1, player2, stockfish_path):
        self.num_games = num_games
        self.player1 = player1
        self.player2 = player2
        self.stockfish_path = stockfish_path
        self.results = []

    def run_simulation(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        for game_num in tqdm(range(self.num_games), desc="Simulating Games", unit="game"):
            white, black = (self.player1, self.player2) if game_num < self.num_games / 2 else (self.player2, self.player1)
            self.play_game(white, black)
        self.engine.quit()

    def play_game(self, white, black):
        board = chess.Board()
        moves = []

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = self.get_move(white, board)
            else:
                move = self.get_move(black, board)

            board.push(move)
            moves.append(str(move))

        result = board.result()
        self.results.append({
            "winner": self.get_winner(result),
            "loser": self.get_loser(result),
            "white_pieces": str(white),
            "black_pieces": str(black),
            "moves": ', '.join(moves)
        })

    def get_move(self, player, board):
        if isinstance(player, AlphaPawn):
            return player.choose_move(board)
        else:
            result = self.engine.play(board, chess.engine.Limit(time=0.1))
            return result.move

    def get_winner(self, result):
        if result == '1-0':
            return 'White'
        elif result == '0-1':
            return 'Black'
        else:
            return 'Draw'

    def get_loser(self, result):
        if result == '1-0':
            return 'Black'
        elif result == '0-1':
            return 'White'
        else:
            return 'Draw'

    def write_to_csv(self, file_name):
        with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["winner", "loser", "white_pieces", "black_pieces", "moves"])
            writer.writeheader()
            for game_result in self.results:
                writer.writerow(game_result)

    def close_engine(self):
        self.engine.quit()

if __name__ == "__main__":
    num_simulated_games = 10
    ai_player = AlphaPawn()  # Your AI
    stockfish_path = "./stockfish-windows-x86-64-avx2.exe"  # Replace with your Stockfish path

    simulation = ChessSimulation(num_simulated_games, ai_player, "Stockfish", stockfish_path)
    simulation.run_simulation()
    simulation.write_to_csv("chess_simulation_results.csv")

    print("Simulation complete. Results saved to chess_simulation_results.csv")