from mcts import MCTS
from cnn_chess_model import CNNChessModel

class AlphaPawn:
    """
    AlphaPawn is a chess-playing AI that utilizes a combination of Monte Carlo Tree Search (MCTS) 
    and a Convolutional Neural Network (CNN) model to choose the best move in a given chess board position.

    Attributes:
        cnn_model (CNNChessModel): An instance of the CNNChessModel class to evaluate chess board positions.
        mcts (MCTS): An instance of the MCTS class to perform the tree search for deciding moves.
    """

    def __init__(self):
        """
        Initializes the AlphaPawn instance by creating a CNN model for chess position evaluation 
        and a Monte Carlo Tree Search (MCTS) algorithm for move selection.
        """
        self.cnn_model = CNNChessModel()
        self.mcts = MCTS(self.cnn_model)

    def choose_move(self, board):
        """
        Selects the best chess move for the given board position.

        Args:
            board (Any): The current state of the chess board. The type depends on the chess library used.

        Returns:
            Any: The best move determined by the MCTS algorithm. The type depends on the chess library used.
        """
        return self.mcts.select_move(board)
