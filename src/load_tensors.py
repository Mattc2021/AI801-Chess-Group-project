import torch


def load_tensors(file_name):
    data = torch.load(file_name)
    tensor_states = data["states"]
    evaluations = data["evaluations"]
    return tensor_states, evaluations


def display_tensor_info(tensor_states, evaluations):
    print(f"Total number of board states: {len(tensor_states)}")
    for i, (tensor, eval) in enumerate(zip(tensor_states, evaluations)):
        print(f"\nBoard State {i+1}:")
        print(tensor.numpy())  # Convert tensor to numpy array for easy viewing
        print(f"Stockfish Evaluation: {eval}")


if __name__ == "__main__":
    file_name = "../assets/chess_position_tensors/chess_game_4_states_2023-11-18.pt"  # Replace with your actual file name
    tensor_states, evaluations = load_tensors(file_name)
    display_tensor_info(tensor_states, evaluations)