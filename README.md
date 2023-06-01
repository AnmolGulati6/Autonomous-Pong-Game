# Autonomous-Pong-Game
This project is an implementation of the classic game Pong with an AI player trained using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm

# Features
- Play against an AI: You can play against an AI-controlled paddle in the game
- Train AI: The NEAT algorithm is used to train the AI to play Pong. You can run the training process to evolve the AI's performance over multiple generations
- Test the best AI: Once the training is complete, you can test the performance of the best AI model against a human player

# Prerequisites
- Python 3
- Pygame library
- NEAT-Python library

# Installation
- Clone the repository 
  - git clone https://github.com/AnmolGulati6/Autonomous-Pong-Game.git 
- pip install pygame
- pip install neat-python

# Usage
Open the playAI.py file and execute it using Python:
The game will start, and you will see the start screen with instructions on how to play
To play against the AI, press the Space key to start the game. Use the W and S keys to move the left paddle up and down, respectively
To train the AI, comment the self.show_start_screen() in the first function and uncomment the line # run_neat_algorithm(config) in the main section of the script. This will start the NEAT training process, and the AI will evolve over multiple generations.
To test the performance of the best AI model, comment the run_neat_algorithm(config), uncomment the self.show_start_screen() and line # test_best(config) in the main section of the script. This will load the best AI model from a saved file and test it against a human player.

# Configuration
The configuration for the NEAT algorithm is specified in the config.txt file. You can modify this file to adjust the parameters of the algorithm, such as population size, mutation rates, and network structure.



