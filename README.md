# rl-dst-egg-game

You can see the model play and learn by running: `python3 ./test.py`

Dependencies: Gymnasium, PyTorch, Numpy

The reinforcement learning model is designed to play the minigame, _An Eggy Tally_ from Klei Entertainment's _Don't Starve Together._

The game is somewhat similar to _2048,_ in which there are numbers contained in a 4x4 grid.
In _An Eggy Tally,_ a player's can combine adjacent numbers and add them together to reach 100, which is represented by an egg in the game.
A player's goal is to combine these numbers such that they are left with as many eggs on the grid as possible.
An example game: https://www.youtube.com/watch?v=KD2Ub4lyC9c (Credit to Lanzak K. on Youtube).

I recreated this game environment, then adapted it into a custom Gymnasium (formerly Gym) environment for a neural network to interact with.

The reinforcement learning model is based off PyTorch's "Reinforcement Learning (DQN) Tutorial" by Adam Paszke (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). 

Currently, the model only reaches a final reward state of about twice what it gets from random input. I am not sure why the model plateaus at that point (even while letting the model run for extended periods of time, combined with testing many different sets of hyperparameters).

The current reason I suspect is because the game is very focused on adding numbers to get EXACTLY 100. A move that would result in a 99 would greatly affect the outcome of the rest of how the game would be played. A move that results in a 101 is invalid, and cannot be made.

I would like to improve on this in the future, once I have learned more of the theory behind reinforcement learning, and especially once I have learned the best methods for data preprocessing, as it currently uses none.
