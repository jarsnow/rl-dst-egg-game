'''
The logic_env class is meant to turn the logic of the game into a workable class in the eyes of Gym.

The important attributes and functions that are needed by Gym are:

    - action_space : defines the valid inputs that the neural network is allowed to input into the game logic.

    - observation_space : defines the valid outputs that the game logic is outputting into the neural network.

    - reset() : called on a logic obj to prepare the game for a new episode. Essentially just starts the game over again (reset, duh).
        - Needs to return a tuple: (observation, info). 
            - Observation is the current values of the logic game
            - Info holds optional information that is meant for human digestion, and is not handled by Gym (to my knowledge).

    - step() : the main engine of the environment, which handles a single change in state of the game. 
        - Changes the logic object for the rest of the episode
        - Required parameters:
            - action : is essentially the neural network's final answer to a given state.
                - A way to map the neural network's actions to usable actions in the game needs to be defined somewhere in the LogicEnv class. (ex: a dictionary) 
        - Needs to return a tuple that contains (observation, reward, terminated, False, info):
            - The new observation after taking the step
            - The reward given to the neural network for taking that action
            - Whether or not the game is over (and therefore the episode should be over as well)
            - Info holds optional information that is meant for human digestion, and is not handled by Gym (to my knowledge).

    
Less important attributes and functions are:

    - close(): called to end any open resources that are used by the environment (ex: closing out of an open pygame window)

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

        # outputs are represented ((0 - 100), (0 - 100) .... ) for 16 total
        self.observation_space = spaces.Box(low=1, high=100, shape=(4,4), dtype=np.int32)

        # inputs are represented (0,1,2 ... 63)
        self.action_space = spaces.Discrete(64)
    
    # assumes input is 0-4*len*width-1, or 0-63 for a normal 4x4 grid
    def get_playable_move_from_input(self, num_input):
        # IMPORTANT: this logic should be fixed later to accommodate non-4x4 grids!

        # the input should be as follows:
        # 0 : move tile at (0,0) left
        # 1 : move tile at (0,0) right
        # ... 
        # 4 : move tile at (0,1) left 
        # 18 : move tile at (1,0) up

        dir = num_input % 4
        tile_row = num_input // 16
        tile_col = (num_input % 16) // 4 

        to_row = tile_row
        to_col = tile_col

        match dir:
            case 0:
                # left
                to_col -= 1
            case 1:
                # right
                to_col += 1
            case 2:
                # up
                to_row -= 1
            case 3:
                # down
                to_row += 1

        return tile_row, tile_col, to_row, to_col
    
    def step(self, action):
        NO_GOAL_PUNISHMENT = -5
        # turn move from action into usable move
        reward_before = self.logic.get_current_reward()

        move = self.get_playable_move_from_input(action)

        if(not self.logic.is_valid_move(*move)):
           # what to do if the bot makes a bad move?
           pass 

        # actually make the move on the board
        self.logic.try_move_num(*move)

        reward_after = self.logic.get_current_reward()

        # reward is NO_GOAL_PUNISHMENT, unless it reaches a 100, or moves an existing 100 downwards.
        reward = (reward_after - reward_before) if (reward_after - reward_before != 0) else NO_GOAL_PUNISHMENT

        game_ended = self.logic.game_ended()
        observation = self.get_obs()
        info = self.get_info()

        #
        return observation, reward, game_ended, False, info

    # might add pygame visualization later, but not now.
    def close(self):
        pass

    def get_info(self):
        return str(self.logic)

    # TODO: check if this has to be casted to a box
    # it does not I believe
    def get_obs(self):
        return self.logic.get_nums_by_row() 
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.logic.reset()

        return self.get_obs(), self.get_info()

