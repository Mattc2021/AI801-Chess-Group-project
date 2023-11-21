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
        total_simulations = 50  # Number of simulations - adjust as needed

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
                # Append piece values and color as features
                piece_value = PIECE_VALUES[piece.piece_type] * (1 if piece.color == board.turn else -1)
                one_hot_piece_type = tf.one_hot(piece.piece_type - 1, 6)  # Assuming 6 piece types
                one_hot_color = tf.one_hot(int(piece.color), 2)  # Assuming 2 colors (0 for black, 1 for white)
                features = [piece_value] + list(one_hot_piece_type.numpy()) + list(one_hot_color.numpy())
                board_representation.extend(features)
            else:
                # Placeholder for empty squares
                board_representation.extend([0] * 12)  # Fill with zeros for empty squares

        # Check if the size matches the intended shape before reshaping
        expected_size = 8 * 8 * 12
        actual_size = len(board_representation)
        print(f"Actual size of board representation: {actual_size}")

        if actual_size != expected_size:
            print(f"Expected size: {expected_size}, Actual size: {actual_size}")
            # Pad the board representation if its size doesn't match the expected size
            board_representation.extend([0] * (expected_size - actual_size))

        # Reshape the representation to fit the expected CNN input shape (8, 8, 12)
        processed_board = np.array(board_representation).reshape(8, 8, 12)
        processed_board = np.expand_dims(processed_board, axis=0)  # Add batch dimension
        processed_board = tf.convert_to_tensor(processed_board, dtype=tf.float32)

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
        cnn_weight = 0.6  # Adjust this weight for the CNN output

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

        # Convert the board state for CNN input
        processed_board = self.process_board(board)

        # Pass the processed board through the CNN model
        cnn_evaluation = self.cnn_model.predict(processed_board)

        # Use the CNN output in the evaluation
        print("CNN Evaluation:", cnn_evaluation)
        cnn_score = float(cnn_evaluation[0, 0])  # Accessing the specific value from the NumPy array




        # Combine the CNN evaluation with other evaluation factors using weights
        combined_score = (
            material_weight * score +
            mobility_weight * mobility +
            pawn_structure_weight * pawn_structure +
            king_safety_weight * king_safety +
            center_control_weight * center_control +
            cnn_weight * cnn_score  # Adjust the weight for the CNN output
        )

        return combined_score