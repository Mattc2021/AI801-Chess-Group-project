from mcts import MCTS
from cnn_chess_model import CNNChessModel

class AlphaPawn:
    def __init__(self):
        self.cnn_model = CNNChessModel()
        self.mcts = MCTS(self.cnn_model)

    def choose_move(self, board):
        return self.mcts.select_move(board)
