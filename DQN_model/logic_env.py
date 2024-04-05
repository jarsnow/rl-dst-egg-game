'''
The logic_env class is meant to turn the logic of the game into a workable class in the eyes of Gym.

The important attributes and functions that are needed by Gym are:

    - action_space : defines the valid inputs that the neural network is allowed to input into the game logic.

    - observation_space : defines the valid outputs that the game logic is outputting into the neural network.

    - reset() : called on a logic obj to prepare the game for a new episode. Essentially just starts the game over again (reset, duh).
        - Needs to return a tuple: (observation, info). 
            - Observation is the same format as observation_space
            - Info holds optional information that is meant for human digestion, and is not handled by Gym (to my knowledge).

    - step() : the main engine of the environment, which handles a single change in state of the game. 

'''


from logic import Logic

import gym
from gym import spaces
import pygame
import numpy as np

class LogicEnv(gym.Env):

    def __init__(self, render_mode=None,  width=4, height=4, num_min=1, num_max=20, num_goal=100, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.logic = Logic(width=width, height=height, num_min=num_min, num_max=num_max, num_goal=num_goal)

        self._action