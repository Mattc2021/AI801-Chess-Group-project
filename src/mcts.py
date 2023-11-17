import random
import numpy as np
import chess
from utils import PIECE_VALUES
import tensorflow as tf
# Define MCTS algorithm
class MCTS:
    def __init__(self, cnn_model, exploration_factor=1.0, temperature=1.0):
        self.cnn_model = cnn_model
        self.exploration_factor = exploration_factor
        self.temperature = temperature

    def select_move(self, board):
        chosen_move_mcts = self.run_mcts(board)
        chosen_move_ab = self.alpha_beta_pruning(board, chosen_move_mcts)
        return chosen_move_ab

    def run_mcts(self, board):
        possible_moves = list(board.legal_moves)
        move_visits = {move: 0 for move in possible_moves}
        total_simulations = 5  # Number of simulations - adjust as needed

        for _ in range(total_simulations):
            self.simulate(board.copy(), move_visits)

        best_move_mcts = max(possible_moves, key=lambda move: move_visits[move])
        return best_move_mcts

    def simulate(self, board, move_visits):
        possible_moves = list(board.legal_moves)

        # Implement Softmax (Temperature parameter) for exploration
        action_probabilities = [move_visits[move] ** (1 / self.temperature) for move in possible_moves]
        total_prob = sum(action_probabilities)

        if total_prob == 0:
            # If all action_probabilities are zero, assign equal probabilities
            action_probabilities = [1 / len(possible_moves) for _ in possible_moves]
        else:
            action_probabilities = [prob / total_prob for prob in action_probabilities]

        chosen_move = np.random.choice(possible_moves, p=action_probabilities)

        # Update visit count for the chosen move
        move_visits[chosen_move] += 1

        # Perform rollout from the chosen move
        self.rollout(board, chosen_move)

    def rollout(self, board, move):
        rollout_board = board.copy()
        rollout_board.push(move)  # Apply the chosen move to the board

        while not rollout_board.is_game_over():
            possible_moves = list(rollout_board.legal_moves)
            random_move = random.choice(possible_moves)  # Choose a random move
            rollout_board.push(random_move)  # Apply the random move to the board

        # Once the game is over, you might want to evaluate the final state
        # This evaluation could involve the evaluate method or a different approach based on the game's rules

        # Return the evaluation of the final board state
        return self.evaluate(rollout_board)

    def alpha_beta_pruning(self, board, chosen_move):
        # Implement Alpha-Beta Pruning for move selection, using the MCTS-chosen move
        best_move_ab = None
        alpha = -float('inf')
        beta = float('inf')
        max_val = -float('inf')
        
        legal_moves = [chosen_move] if chosen_move else list(board.legal_moves)  # Start with the MCTS-chosen move if available
        for move in legal_moves:
            board.push(move)
            val = self.min_value(board, alpha, beta, 0)
            board.pop()
            if val > max_val:
                max_val = val
                best_move_ab = move
            alpha = max(alpha, max_val)
        return best_move_ab

    def process_board(self, board):
        # Convert the board state to a format suitable for the CNN input
        board_representation = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_value = PIECE_VALUES[piece.piece_type] * (1 if piece.color == board.turn else -1)
                board_representation.append(piece_value)
            else:
                board_representation.append(0)

        # Reshape the representation to fit the CNN input shape
        processed_board = tf.convert_to_tensor([board_representation], dtype=tf.float32)
        processed_board = tf.reshape(processed_board, (1, 8, 8, 1))
        
        return processed_board

    def max_value(self, board, alpha, beta, depth):
        # Maximizer function for Alpha-Beta Pruning
        if depth == 0 or board.is_game_over():
            # Return evaluation function or utility value
            return self.evaluate(board)
        
        val = -float('inf')
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            board.push(move)
            val = max(val, self.min_value(board, alpha, beta, depth + 1))
            board.pop()
            if val >= beta:
                return val
            alpha = max(alpha, val)
        return val

    def min_value(self, board, alpha, beta, depth):
        # Minimizer function for Alpha-Beta Pruning
        if depth == 0 or board.is_game_over():
            # Return evaluation function or utility value
            return self.evaluate(board)
        
        val = float('inf')
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            board.push(move)
            val = min(val, self.max_value(board, alpha, beta, depth + 1))
            board.pop()
            if val <= alpha:
                return val
            beta = min(beta, val)
        return val
    
    def evaluate(self, board):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }

        # Evaluation weights for different factors
        material_weight = 1.0
        mobility_weight = 0.5
        pawn_structure_weight = 0.3
        king_safety_weight = 0.2
        center_control_weight = 0.4

        score = 0

        # Material advantage
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    score += piece_values[piece.piece_type]
                else:
                    score -= piece_values[piece.piece_type]

        # Mobility - Number of legal moves available
        white_mobility = len(list(board.legal_moves))
        board.turn = chess.BLACK  # Switch turns to assess opponent's mobility
        black_mobility = len(list(board.legal_moves))
        board.turn = chess.WHITE  # Switch back to original turn
        mobility = white_mobility - black_mobility
        score += mobility_weight * mobility

        # Pawn structure - Evaluation based on pawn structure can be complex
        # Here, a simple evaluation based on the number of pawns present is used
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        pawn_structure = white_pawns - black_pawns
        score += pawn_structure_weight * pawn_structure

        # King safety - Distance of kings from the center
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        white_king_safety = abs(chess.square_file(white_king_square) - 4) + abs(chess.square_rank(white_king_square) - 4)
        black_king_safety = abs(chess.square_file(black_king_square) - 4) + abs(chess.square_rank(black_king_square) - 4)
        king_safety = black_king_safety - white_king_safety
        score += king_safety_weight * king_safety

        # Control of the center - Presence of pieces in the center
        center_control = 0
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    center_control += 1
                else:
                    center_control -= 1
        score += center_control_weight * center_control

        return score
