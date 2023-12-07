# AI801-Chess-Group-project
To run the beta version of the chess board install from requirements by running:

pip install -r requirements.txt

You can either do this with or without the use of a virtual environment.

To create or use a virtual environment run the command

python -m venv ./venv

Then navigate to the Scripts folder that has appeared in ./venv and run the activate.bat.
From there you should have a local python version solely for this project.


# To Run the Game

There are two ways you can run the game

  1) The first if you want to run the simulation to see how we were able to train the AI you should run the Simulator.py. Whne you run this you specify the amount of games with the variable "num_simulated_games". The more you specify the more games the model will run and train on
  2) If you want to play against our AI all you have to do is run main which is found in our src folder (for more accurate descriptions of the file structure look below) When you run main.py a tkinter window will display which allows us to play against the AI.

The AI is still very basic and will make mistakes, but the premise is there to create a great chess AI.
     
Team Member
--------------------
Coding Work

Matthew Chiaravalloti 
  1) Created the simulator to run and train the CNN
  2) Created the Chess GUI to be able to play our AI
  3) Created the Opening Database to the code
  4) Created the Graphs needed to display some descriptive data
  5) Commented the Code and structured the files
     
TJ (Timothy) Gallagher
  1) Developed MCTS algorith
  2) Developed the Alpha-Beta Pruning algorithms
  3) Developed the CNN models and engine

Jennifer Yin
  1) ???

File Structure

Ai801-Chess-Group-Project
----assets                          -This is where the data gets saved off into a csv, also the png's for the pieces are in this folder as well
--------chess_position_tensors      -This is where the tensors get saved off from the Simulator running
----graphs                          -This is where the graphs get saved off to describe game_lengths, loss_over_epochs, material_advantage_over_time, player_win_loss_distribution, position_evaluation_over_time
----src                             -This is where the main code is and the folder in which you should run the simulator.py or the main.py
--------archive                     -This is extra code that I would like to go back and fix, but code that is not related to the main code
--------StrategosCNNModel           -This is where the model gets saved and loaded from
------------assets                  -These are helpers for the StrategosCNNModel
------------variables               -These are helpers for the StrategosCNNModel





Copyright/Attribution Notice: 
JohnPablok's improved Cburnett chess set.
