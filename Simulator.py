from AlphaPawn import AlphaPawn
import chess.svg
import chess.engine
import csv
from tqdm import tqdm

class ChessSimulation:
    def __init__(self, num_games, player1, player2):
        self.num_games = num_games
        self.player1 = player1
        self.engine = chess.engine.SimpleEngine.popen_uci("./stockfish-windows-x86-64-avx2.exe")
        self.results = []

    def run_simulation(self):
        for _ in tqdm(range(self.num_games), desc="Simulating Games", unit="game"):
            self.play_game()

    def play_game(self):
        board = chess.Board()
        moves = []

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = self.player1.choose_move(board)
            else:
                result = self.engine.play(board, chess.engine.Limit(time=0.1))  # Stockfish move with a time limit
                move = result.move
            
            board.push(move)
            moves.append(str(move))

        result = board.result()
        self.results.append({
            "winner": self.get_winner(result),
            "loser": self.get_loser(result),
            "white_pieces": str(self.player1),
            "black_pieces": "Stockfish",
            "moves": ', '.join(moves)
        })

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

# Example usage
if __name__ == "__main__":
    num_simulated_games = 100
    ai_player = AlphaPawn()  # Your AI
    stockfish_path = "/stockfish-windows-x86-64-avx2.exe"  # Replace with your Stockfish path

    simulation = ChessSimulation(num_simulated_games, ai_player, stockfish_path)
    simulation.run_simulation()
    simulation.write_to_csv("chess_simulation_results.csv")
    simulation.close_engine()

    print("Simulation complete. Results saved to chess_simulation_results.csv")