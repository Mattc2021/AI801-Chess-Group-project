import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Ensure the "graphs" directory exists
graphs_dir = "../graphs"
os.makedirs(graphs_dir, exist_ok=True)

def save_plot(plt, filename, run_number, date_str):
    unique_filename = f"{filename}_run{run_number}_{date_str}.png"
    path = os.path.join(graphs_dir, unique_filename)
    plt.savefig(path)
    print(f"Plot saved as {path}")
    plt.close()

def plot_win_loss_distribution(results_df, run_number, date_str):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='winner', data=results_df)
    plt.title('Player Win-Loss Distribution')
    plt.xlabel('Player')
    plt.ylabel('Count')
    save_plot(plt, 'player_win_loss_distribution', run_number, date_str)

def plot_material_advantage_over_time(material_advantages_df, run_number, date_str):
    plt.figure(figsize=(10, 6))

    # Iterate over each game in the DataFrame
    for game_index, row in material_advantages_df.iterrows():
        game_advantages = row['material_advantage']
        move_numbers = list(range(1, len(game_advantages) + 1))  # Create a move number list
        sns.lineplot(x=move_numbers, y=game_advantages, label=f"Game {game_index + 1}")

    plt.title('Material Advantage Over Time')
    plt.xlabel('Move Number')
    plt.ylabel('Material Advantage')
    plt.legend()
    save_plot(plt, 'material_advantage_over_time', run_number, date_str)

def plot_position_evaluation_over_time(evaluations_df, run_number, date_str):
    plt.figure(figsize=(10, 6))
    # Iterate over each game in the DataFrame
    for game_index in range(len(evaluations_df)):
        game_evaluations = evaluations_df.iloc[game_index]['evaluations']
        if game_evaluations:
            sns.lineplot(data=game_evaluations, label=f"Game {game_index + 1}")
    plt.title('Position Evaluation Over Time')
    plt.xlabel('Move Number')
    plt.ylabel('Evaluation Score')
    plt.legend()
    save_plot(plt, 'position_evaluation_over_time', run_number, date_str)