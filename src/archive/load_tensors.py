import torch

def load_tensors(file_name):
    """
    Load tensors and evaluations from a file using PyTorch.

    Parameters:
    - file_name: The name of the file from which to load the data.

    Returns:
    A tuple containing the list of tensor states and their corresponding evaluations.
    """
    data = torch.load(file_name)  # Load the data from the specified file
    tensor_states = data["states"]  # Extract tensor states
    evaluations = data["evaluations"]  # Extract evaluations
    return tensor_states, evaluations


def display_tensor_info(tensor_states, evaluations):
    """
    Display information about tensor states and their evaluations.

    Parameters:
    - tensor_states: A list of tensor representations of chess board states.
    - evaluations: A list of evaluations corresponding to each tensor state.
    """
    print(f"Total number of board states: {len(tensor_states)}")
    for i, (tensor, eval) in enumerate(zip(tensor_states, evaluations)):
        print(f"\nBoard State {i+1}:")
        print(tensor.numpy())  # Convert the tensor to a numpy array for easier viewing
        print(f"Stockfish Evaluation: {eval}")


if __name__ == "__main__":
    # File name for the stored tensor data. Replace with the actual file name as needed.
    file_name = "../assets/chess_position_tensors/game_0_run_1.pt"
    tensor_states, evaluations = load_tensors(file_name)  # Load tensors and evaluations from the file
    display_tensor_info(tensor_states, evaluations)  # Display the loaded information
