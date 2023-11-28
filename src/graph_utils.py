import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Ensure the "graphs" directory exists for storing plots
graphs_dir = "../graphs"
os.makedirs(graphs_dir, exist_ok=True)

def save_plot(plt, filename, run_number, date_str):
    """
    Save the given matplotlib plot to a file in the graphs directory.

    Parameters:
    - plt: The matplotlib.pyplot module.
    - filename: Base name of the file to save the plot as.
    - run_number: An identifier for the run number of the experiment or simulation.
    - date_str: A date string to append to the filename for uniqueness.
    """
    unique_filename = f"{filename}_run{run_number}_{date_str}.png"
    path = os.path.join(graphs_dir, unique_filename)
    plt.savefig(path)
    print(f"Plot saved as {path}")
    plt.close()

def plot_win_loss_distribution(results_df, run_number, date_str):
    """
    Create and save a plot showing the distribution of wins and losses.

    Parameters:
    - results_df: DataFrame containing game results with a 'winner' column.
    - run_number: Run number of the experiment or simulation.
    - date_str: A date string for file naming.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x='winner', data=results_df)
    plt.title('Player Win-Loss Distribution')
    plt.xlabel('Player')
    plt.ylabel('Count')
    save_plot(plt, 'player_win_loss_distribution', run_number, date_str)

def plot_material_advantage_over_time(material_advantages_df, run_number, date_str):
    """
    Create and save a line plot showing material advantage over time.

    Parameters:
    - material_advantages_df: DataFrame with each row representing a game and a column for material advantages.
    - run_number: Run number of the experiment or simulation.
    - date_str: A date string for file naming.
    """
    plt.figure(figsize=(10, 6))

    for game_index, row in material_advantages_df.iterrows():
        game_advantages = row['material_advantage']
        move_numbers = list(range(1, len(game_advantages) + 1))
        sns.lineplot(x=move_numbers, y=game_advantages, label=f"Game {game_index + 1}")

    plt.title('Material Advantage Over Time')
    plt.xlabel('Move Number')
    plt.ylabel('Material Advantage')
    plt.legend()
    save_plot(plt, 'material_advantage_over_time', run_number, date_str)

def plot_position_evaluation_over_time(evaluations_df, run_number, date_str):
    """
    Create and save a line plot showing position evaluation over time for each game.

    Parameters:
    - evaluations_df: DataFrame with each row representing a game and a column for position evaluations.
    - run_number: Run number of the experiment or simulation.
    - date_str: A date string for file naming.
    """
    plt.figure(figsize=(10, 6))

    for game_index in range(len(evaluations_df)):
        game_evaluations = evaluations_df.iloc[game_index]['evaluations']
        if game_evaluations:
            sns.lineplot(data=game_evaluations, label=f"Game {game_index + 1}")
    
    plt.title('Position Evaluation Over Time')
    plt.xlabel('Move Number')
    plt.ylabel('Evaluation Score')
    plt.legend()
    save_plot(plt, 'position_evaluation_over_time', run_number, date_str)

def plot_loss_over_epochs(training_loss, date_str, validation_loss=None):
    epochs = range(1, len(training_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_loss, 'bo-', label='Training Loss')
    if validation_loss:
        plt.plot(epochs, validation_loss, 'r*-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    save_plot(plt, 'loss_over_epochs', date_str)